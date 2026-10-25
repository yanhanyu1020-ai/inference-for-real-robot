"""Explicit v1/v2 readers. Raw closure is not training/alignment validity."""
from dataclasses import replace
from io import BytesIO
import json
import csv
from pathlib import Path
import pickle
import struct
import zlib

import numpy as np

from raw_image import ENCODING, decode_image

MAX_CHUNK_BYTES = 80*1024**2


def load_object(path):
    value=json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(value,dict):
        raise ValueError('expected JSON object: '+str(path))
    return value


def format_version(episode):
    path=Path(episode)/'meta.json'
    meta=load_object(path) if path.exists() else {}
    version=meta.get('format_version',1)
    if type(version) is not int or version not in (1,2):
        raise ValueError('unsupported episode format')
    if version==2 and meta.get('image_storage')!='mcap':
        raise ValueError('v2 requires MCAP image storage')
    return version


class BoundedInput:
    def __init__(self, stream): self.stream=stream
    def read(self, size):
        if size<0 or size>MAX_CHUNK_BYTES:
            raise ValueError('MCAP nested read exceeds size limit')
        return self.stream.read(size)


def chunk_records(chunk):
    from mcap.stream_reader import breakup_chunk
    if not 0 <= chunk.uncompressed_size <= MAX_CHUNK_BYTES:
        raise ValueError('MCAP uncompressed chunk exceeds size limit')
    if chunk.compression=='lz4':
        import lz4.frame
        decoder=lz4.frame.LZ4FrameDecompressor()
        data=decoder.decompress(chunk.data,max_length=MAX_CHUNK_BYTES+1)
        if len(data)>MAX_CHUNK_BYTES or not decoder.eof or decoder.unused_data:
            raise ValueError('invalid or oversized LZ4 chunk')
    elif chunk.compression=='':
        data=chunk.data
    else:
        raise ValueError('unsupported chunk codec: '+chunk.compression)
    if len(data)!=chunk.uncompressed_size:
        raise ValueError('chunk uncompressed size mismatch')
    return breakup_chunk(replace(chunk,compression='',data=data),validate_crc=True)


def iter_mcap_frames(path, camera):
    """Yield only complete CRC-verified chunks, in physical record order."""
    from mcap.stream_reader import StreamReader
    from mcap.records import Chunk,Channel,Message
    channels={}
    with Path(path).open('rb') as handle:
        reader=StreamReader(BoundedInput(handle),emit_chunks=True,validate_crcs=True,
                            record_size_limit=MAX_CHUNK_BYTES)
        for outer in reader.records:
            records=chunk_records(outer) if isinstance(outer,Chunk) else [outer]
            for record in records:
                if isinstance(record,Channel):
                    channels[record.id]=record
                if not isinstance(record,Message):
                    continue
                if not isinstance(outer,Chunk):
                    raise ValueError('raw images must be stored inside checked chunks')
                channel=channels.get(record.channel_id)
                if channel is None or channel.message_encoding!=ENCODING or channel.topic!='/camera/'+camera:
                    raise ValueError('unexpected MCAP channel/encoding')
                metadata,image=decode_image(record.data)
                if metadata['camera']!=camera:
                    raise ValueError('camera name mismatch')
                if record.log_time!=metadata['wall_time_ns'] or record.publish_time!=metadata['wall_time_ns']:
                    raise ValueError('MCAP reception timestamps disagree with payload')
                yield metadata,image
        verify_summary_crc(handle)


def verify_summary_crc(handle):
    """MCAP 1.3 StreamReader omits summary CRC; verify its bounded byte range.

    Footer is a 9-byte record header plus two uint64 offsets and uint32 CRC.
    CRC covers summary bytes and the footer through its two offsets (Writer.finish).
    Run after yielding checked chunks so recovery can preserve a damaged index's data.
    """
    handle.seek(0, 2)
    end=handle.tell()
    if end<45:
        raise ValueError('missing MCAP footer')
    footer_at=end-8-29
    handle.seek(footer_at)
    footer=handle.read(29)
    opcode,length,start,offset_start,expected=struct.unpack('<BQQQI',footer)
    if opcode!=2 or length!=20 or not (start==0 or 8<=start<=footer_at):
        raise ValueError('invalid MCAP footer offsets')
    if offset_start and not (start and start<=offset_start<=footer_at):
        raise ValueError('invalid MCAP summary offsets')
    if not expected:
        raise ValueError('missing MCAP summary CRC')
    handle.seek(start if start else footer_at)
    remaining=footer_at-(start if start else footer_at)
    crc=0
    while remaining:
        block=handle.read(min(1024*1024,remaining))
        if not block:
            raise ValueError('truncated MCAP summary')
        crc=zlib.crc32(block,crc)
        remaining-=len(block)
    if zlib.crc32(footer[:25],crc)!=expected:
        raise ValueError('MCAP summary CRC mismatch')


def iter_camera_frames(episode, camera):
    episode=Path(episode)
    if camera not in ('front','wrist'):
        raise ValueError('unknown camera')
    if format_version(episode)==1:
        from PIL import Image
        # Legacy files are trusted locally generated pickle files, never untrusted downloads.
        for index,path in enumerate(sorted((episode/'frames').glob('*.pkl')),1):
            with path.open('rb') as f: record=pickle.load(f)
            value=record['camera'][camera]
            with Image.open(BytesIO(value['rgb'])) as png:
                image=np.asarray(png.convert('RGB'))[:,:,::-1].copy()
            height,width=image.shape[:2]
            yield {'camera':camera,'serial':value['serial'],'seq':index,'reader_seq':None,
                'width':width,'height':height,'stride':width*3,'dtype':'uint8','pixel_format':'bgr8',
                'monotonic_ns':int(value['monotonic_ns']),'wall_time_ns':int(value['wall_time_ns']),
                'boot_id':None,'timestamp_source':'host_receive','device_timestamp_ns':None,
                'device_frame_number':None},image
        return
    last_seq,last_mono,boot,serial=0,None,None,None
    paths=sorted((episode/'raw/cameras').glob(camera+'_*.mcap'))
    if not paths:
        raise ValueError('no finalized camera shards: '+camera)
    for path in paths:
        for metadata,image in iter_mcap_frames(path,camera):
            if boot is not None and metadata['boot_id']!=boot:
                raise ValueError('boot changed within episode')
            if serial is not None and metadata['serial']!=serial:
                raise ValueError('camera serial changed within episode')
            if metadata['seq']!=last_seq+1 or (last_mono is not None and metadata['monotonic_ns']<=last_mono):
                raise ValueError('nonconsecutive sequence or non-increasing monotonic time')
            last_seq,last_mono=metadata['seq'],metadata['monotonic_ns']
            boot,serial=metadata['boot_id'],metadata['serial']
            yield metadata,image


def inspect_episode(episode):
    episode=Path(episode)
    version=format_version(episode)
    errors=[]
    streams={}
    try: summary=load_object(episode/'summary.json')
    except (OSError,ValueError) as exc:
        summary={}
        errors.append('summary: '+str(exc))
    boot=load_object(episode/'meta.json').get('boot_id') if version==2 else None
    readable=True
    for camera in ('front','wrist'):
        count=0
        first=last=None
        max_gap=0
        try:
            for metadata,_ in iter_camera_frames(episode,camera):
                if boot is not None and metadata['boot_id']!=boot:
                    raise ValueError('boot mismatch between cameras')
                boot=metadata['boot_id']
                if last is not None: max_gap=max(max_gap,metadata['monotonic_ns']-last['monotonic_ns'])
                if first is None: first=metadata
                last=metadata
                count+=1
        except Exception as exc:
            errors.append(camera+': '+str(exc))
            readable=False
        declared=summary.get('streams',{}).get(camera,{})
        stats={'frames':count,'first':first,'last':last,'max_gap_ns':max_gap,
            'effective_fps':(count-1)*1e9/(last['monotonic_ns']-first['monotonic_ns']) if count>1 else 0.0,
            'rejected':declared.get('rejected'),'write_errors':declared.get('write_errors')}
        streams[camera]=stats
        if version==2:
            for key in ('received','accepted','submitted','written','durable'):
                if type(declared.get(key)) is not int or declared[key]!=count:
                    errors.append(camera+': '+key+' count does not match verified frames')
            for key in ('rejected','write_errors','pending'):
                if type(declared.get(key)) is not int or declared[key]!=0:
                    errors.append(camera+': nonzero or missing '+key)
    raw_closed=bool(readable and summary.get('durable_complete') and
                    not list((episode/'raw/cameras').glob('*.active'))) if version==2 else readable
    if not raw_closed: errors.append('raw closure is unconfirmed')
    if version==2 and summary.get('quality_ok') is not True:
        errors.append('capture summary reports quality incomplete')
    return {'format_version':version,'raw_closed':raw_closed,'application_integrity':not errors,
        'alignment_valid':None,'sensor_integrity_unknown':True,'streams':streams,'errors':errors}


def summarize_timing(episode):
    """Fast stopped-episode timing index, not a replacement for full CRC validation."""
    episode=Path(episode)
    if format_version(episode)!=2: raise ValueError('v2 timing index required')
    report=load_object(episode/'summary.json')
    if report.get('durable_complete') is not True:
        raise ValueError('raw capture closure is not confirmed')
    output=[]
    for camera in ('front','wrist'):
        stream=report['streams'][camera]
        timing=stream['wall_summary']
        count=timing['count']
        if type(count) is not int or count<1 or any(stream.get(k)!=count for k in ('received','accepted','written','durable')):
            raise ValueError(camera+': timing counters do not match closed data')
        if any(stream.get(k)!=0 for k in ('rejected','write_errors','pending')):
            raise ValueError(camera+': capture reports rejected frames or write errors')
        for shard in stream['shards']:
            path=episode/'raw/cameras'/shard['path']
            if Path(shard['path']).name!=shard['path'] or path.stat().st_size!=shard['bytes']:
                raise ValueError('missing or changed camera shard')
        output.append({**timing,'name':'camera:'+camera,'timestamp_source':'camera.wall_time_ns'})
    return output


def validate_raw_episode(episode, config):
    from validate_episode import REQUIRED_RAW_FILES,ALLOW_EMPTY_RAW_FILES,CONTINUOUS_STATE_FILES
    episode=Path(episode)
    report=inspect_episode(episode)
    errors=report['errors']
    raw_stats={}
    warnings=['Raw integrity only; alignment and offline training pairing remain unvalidated.']
    starts=[s['first']['monotonic_ns'] for s in report['streams'].values() if s['first']]
    ends=[s['last']['monotonic_ns'] for s in report['streams'].values() if s['last']]
    for camera,stats in report['streams'].items():
        if stats['frames']<2: errors.append(camera+': at least two raw frames are required')
        if stats['effective_fps']<config['fps']*.8: errors.append(camera+': effective frame rate too low')
        if stats['max_gap_ns']>int(config.get('camera_stale_ms',500)*1e6): errors.append(camera+': excessive frame gap')
        if stats['first'] and stats['first']['serial']!=config['cameras'][camera]: errors.append(camera+': serial mismatch')
    for name in REQUIRED_RAW_FILES:
        try:
            first=last=None
            count=0
            max_gap=0
            with (episode/'raw'/name).open(newline='') as f:
                for row in csv.DictReader(f):
                    stamp=int(row['monotonic_ns'])
                    int(row['wall_time_ns'])  # Preserve wall steps; ordering uses monotonic only.
                    if last is not None:
                        if stamp<last: raise ValueError('nonmonotonic state sequence')
                        max_gap=max(max_gap,stamp-last)
                    if first is None: first=stamp
                    last=stamp
                    count+=1
            raw_stats[name]={'rows':count,'first_monotonic_ns':first,'last_monotonic_ns':last,'max_gap_ns':max_gap}
            if not count:
                (warnings if name in ALLOW_EMPTY_RAW_FILES else errors).append(name+': empty state stream')
            if count and starts and name in CONTINUOUS_STATE_FILES:
                if first>min(starts)+250_000_000 or last<max(ends)-250_000_000:
                    errors.append(name+': does not cover image interval within 250 ms')
                limit=config.get('piper_stale_ms',250) if name=='piper_state.csv' else config.get('go2_stale_ms',500)
                if max_gap>limit*1e6: errors.append(name+': excessive state gap')
        except (OSError,ValueError,KeyError) as exc:
            errors.append(name+': '+str(exc))
    report.update(ok=not errors,application_integrity=not errors,training_ready=False,
                  episode=str(episode),raw=raw_stats,warnings=warnings)
    return report
