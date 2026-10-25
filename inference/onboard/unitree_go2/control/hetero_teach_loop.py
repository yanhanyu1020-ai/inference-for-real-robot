#!/usr/bin/env python3
"""Joystick dog + Piper drag-teach capture.

Thin wrapper so ``pgrep -f hetero_teach_loop.py`` does not match joystick
teleop.  Enters firmware drag-teach (MotionCtrl_1), then only reads joints;
left stick still drives the Go2 bridge.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from hetero_teleop_loop import main  # noqa: E402


if __name__ == "__main__":
    sys.argv[1:1] = ["--teach-arm"]
    main()
