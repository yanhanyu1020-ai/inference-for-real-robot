"""Small utilities for the V3 release; no dependency on previous datasets."""
import json


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def episode_path(root, info, index, camera=None):
    key = 'video_path' if camera else 'data_path'
    return root / info[key].format(episode_index=index, episode_chunk=index // info['chunks_size'], video_key=camera)
