"""Action chunk length is independent of the dataset sampling frequency."""
import json
from numbers import Integral
from pathlib import Path

# Project default explicitly selected by the user, independently of dataset fps.
DEFAULT_ACTION_HORIZON = 15


def validate_action_horizon(value):
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError('action_horizon must be a positive integer number of steps')
    return int(value)


def resolve_action_horizon(root, override=None):
    if override is not None:
        return validate_action_horizon(override)
    path = Path(root).expanduser().resolve()/'configs/action_chunk.json'
    value = json.loads(path.read_text())['action_horizon'] if path.exists() else DEFAULT_ACTION_HORIZON
    return validate_action_horizon(value)
