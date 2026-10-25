"""Read-only summaries for capture data that lives on remote hosts.

The desktop package deliberately does not import ``rosbag`` or Unitree's
capture libraries.  The small Python programs below execute on their owning
host and return plain JSON, which keeps local runtime dependencies to the
standard library and makes the boundary easy to fake in tests.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import shlex
from typing import Callable, Any

from .models import CommandResult
from .remote import run_ssh


@dataclass(frozen=True)
class StreamSummary:
    name: str
    count: int
    first_ns: int
    last_ns: int
    timestamp_source: str = "unknown"
    host: str | None = None
    # None means legacy/unknown evidence, not a verified absence of gaps.
    max_gap_ns: int | None = None
    # Adjacent non-increasing timestamps in capture order (duplicates included).
    nonmonotonic_count: int | None = None
    producer_dropped: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def stream(self) -> str:
        """Compatibility alias used by callers that call a topic a stream."""
        return self.name

    @property
    def topic(self) -> str:
        return self.name

    @property
    def first_timestamp_ns(self) -> int:
        return self.first_ns

    @property
    def last_timestamp_ns(self) -> int:
        return self.last_ns


class InspectorError(RuntimeError):
    """Raised when a remote read-only inspection cannot complete."""


class InspectorConnectionError(InspectorError):
    """Raised when SSH did not establish a transport to the inspector host."""


def _remote_script(kind: str) -> str:
    common = r'''import glob,json,os,re,sys
def natural_key(path):
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)",path)]
def new_entry():
    return {"count":0,"first_ns":None,"last_ns":None,"max_gap_ns":0,"nonmonotonic_count":0}
def add_timestamp(entry,value):
    previous=entry.get("previous")
    if previous is not None:
        delta=value-previous
        entry["max_gap_ns"]=max(entry["max_gap_ns"],delta)
        entry["nonmonotonic_count"]+=int(delta<=0)
    entry["previous"]=value
    entry["count"]+=1
    entry["first_ns"]=value if entry["first_ns"] is None else min(entry["first_ns"],value)
    entry["last_ns"]=value if entry["last_ns"] is None else max(entry["last_ns"],value)
def summary(name,entry,source):
    return dict({key:entry[key] for key in ("count","first_ns","last_ns","max_gap_ns","nonmonotonic_count")},name=name,timestamp_source=source)
'''
    if kind == "p450":
        return common + r'''import rosbag
path=sys.argv[1]
paths=[path] if os.path.isfile(path) else sorted(glob.glob(os.path.join(path,"**","*.bag"),recursive=True),key=natural_key)
streams={}
for bag_path in paths:
    bag=rosbag.Bag(bag_path)
    try:
        for topic in sorted(bag.get_type_and_topic_info()[1]):
            entry=streams.setdefault(topic,dict(new_entry(),header_count=0,bag_count=0))
            for _,msg,t in bag.read_messages(topics=[topic]):
                stamp=getattr(getattr(msg,"header",None),"stamp",None); value=None
                if stamp is not None:
                    try: value=int(stamp.to_nsec())
                    except Exception:
                        try: value=int(stamp.secs)*1000000000+int(stamp.nsecs)
                        except Exception: value=None
                if value is not None and value > 0 and abs(value-int(t.to_nsec())) <= 1000000000:
                    if entry.get("previous_header") is not None and value < entry["previous_header"]:
                        raise ValueError("header timestamp moved backwards on "+topic)
                    entry["previous_header"]=value
                    entry["header_count"]+=1
                else:
                    value=int(t.to_nsec()); entry["bag_count"]+=1
                add_timestamp(entry,value)
    finally: bag.close()
out=[]
for name,entry in sorted(streams.items()):
    source="ros_header" if entry["bag_count"]==0 else ("bag_timestamp" if entry["header_count"]==0 else "mixed")
    if entry["count"]: out.append(summary(name,entry,source))
print(json.dumps(out,separators=(",",":")))'''
    if kind == "unitree":
        return common + r'''import csv,pickle
path=sys.argv[1]; out=[]; pkl_streams={}
meta_path=os.path.join(path,'meta.json')
metadata=json.load(open(meta_path)) if os.path.exists(meta_path) else {}
version=metadata.get('format_version',1)
if version==2:
    sys.path.insert(0,os.path.expanduser('~/heterovla-collection/onboard'))
    from episode_io import summarize_timing
    out.extend(summarize_timing(path))
    pkl_paths=[]
elif version==1:
    pkl_paths=sorted(glob.glob(os.path.join(path,"**","*.pkl"),recursive=True),key=natural_key)
else:
    raise ValueError('unsupported Unitree episode version')
for name in pkl_paths:
    try:
        # These files are trusted recorder output. Inspect encoded-image metadata
        # only; do not import camera libraries or decode PNG images.
        with open(name,"rb") as handle: record=pickle.load(handle)
        cameras=record["camera"]
        if not isinstance(cameras,dict) or not cameras: raise ValueError("missing camera metadata")
        for camera,metadata in cameras.items():
            value=metadata["wall_time_ns"]
            if isinstance(value,bool) or not isinstance(value,int) or value<=0:
                raise ValueError("invalid camera wall_time_ns")
            entry=pkl_streams.setdefault("pkl:"+str(camera),new_entry())
            add_timestamp(entry,value)
    except Exception as exc:
        raise ValueError("invalid recorder pkl "+name+": "+str(exc)) from exc
for name,entry in sorted(pkl_streams.items()):
    item=summary(name,entry,"camera.wall_time_ns")
    summary_path=os.path.join(path,'summary.json')
    if os.path.exists(summary_path):
        capture_summary=json.load(open(summary_path))
        dropped=capture_summary.get('frames_dropped')
        if dropped is not None:
            if type(dropped) is not int or dropped<0: raise ValueError('invalid frames_dropped')
            item['producer_dropped']=dropped
    out.append(item)
for name in sorted(glob.glob(os.path.join(path,"**","*.csv"),recursive=True)):
    entry=new_entry()
    with open(name,newline="") as handle:
        rows=csv.DictReader(handle)
        for row in rows:
            add_timestamp(entry,int(row["wall_time_ns"]))
    if entry["count"]: out.append(summary(os.path.relpath(name,path),entry,"wall_time_ns"))
print(json.dumps(out,separators=(",",":")))'''
    raise ValueError(f"unsupported inspector kind: {kind}")


def _parse_summaries(text: str, host: str) -> list[StreamSummary]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise InspectorError(f"{host} inspector returned malformed JSON") from exc
    if isinstance(payload, dict):
        payload = payload.get("streams", [])
    if not isinstance(payload, list):
        raise InspectorError(f"{host} inspector returned a non-list summary")
    summaries: list[StreamSummary] = []
    try:
        for item in payload:
            if not isinstance(item, dict):
                raise ValueError("stream entry is not an object")
            count = int(item["count"]); first = int(item["first_ns"]); last = int(item["last_ns"])
            if count <= 0 or first > last:
                raise ValueError("stream summary has invalid count or range")
            max_gap = item.get("max_gap_ns")
            nonmonotonic = item.get("nonmonotonic_count")
            producer_dropped = item.get("producer_dropped")
            if max_gap is not None:
                max_gap = int(max_gap)
                if max_gap < 0:
                    raise ValueError("negative maximum gap")
            if nonmonotonic is not None:
                nonmonotonic = int(nonmonotonic)
                if not 0 <= nonmonotonic < count:
                    raise ValueError("invalid nonmonotonic count")
            if producer_dropped is not None:
                if type(producer_dropped) is not int or producer_dropped < 0:
                    raise ValueError("invalid producer dropped count")
            summaries.append(StreamSummary(str(item["name"]), count, first, last,
                                           str(item.get("timestamp_source", "unknown")), host,
                                           max_gap, nonmonotonic, producer_dropped))
    except (KeyError, TypeError, ValueError) as exc:
        raise InspectorError(f"{host} inspector returned invalid stream summary") from exc
    return summaries


class _Inspector:
    kind: str

    def __init__(self, host: str | None = None, *, timeout_s: float = 30.0,
                 runner: Callable[..., CommandResult] | None = None) -> None:
        self.host = host or self.kind
        self.timeout_s = timeout_s
        self.runner = runner or run_ssh

    def summarize(self, session_dir: str) -> list[StreamSummary]:
        if not session_dir or not isinstance(session_dir, str):
            raise ValueError("session_dir must be a non-empty string")
        command = "python3 -c " + shlex.quote(_remote_script(self.kind)) + " " + shlex.quote(session_dir)
        if self.kind == 'p450':
            # Match the environment established by the existing p450_capture
            # entrypoint; non-interactive SSH does not source ROS automatically.
            setup = ('source /opt/ros/noetic/setup.bash && '
                     'if [ -f /home/amov/p450_experiment/devel/setup.bash ]; then '
                     'source /home/amov/p450_experiment/devel/setup.bash; fi && ')
            command = 'bash -c ' + shlex.quote(setup + command)
        result = self.runner(self.host, command, self.timeout_s)
        if not getattr(result, "ok", getattr(result, "returncode", 1) == 0):
            detail = getattr(result, "stderr", "") or getattr(result, "stdout", "") or "remote command failed"
            if getattr(result, "returncode", None) in {-1, 255}:
                raise InspectorConnectionError(f"{self.kind} inspector transport failed: {detail.strip()}")
            raise InspectorError(f"{self.kind} inspector failed: {detail.strip()}")
        return _parse_summaries(getattr(result, "stdout", ""), self.host)


class P450Inspector(_Inspector):
    kind = "p450"


class UnitreeInspector(_Inspector):
    kind = "unitree"
