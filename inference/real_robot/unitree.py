"""Unitree SDK2/Piper SDK/V4L2 adapter. Import has no hardware side effects."""
import fcntl
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np

from .core import Sensors, piper_sdk_target


def go2_normal_state_codes(config):
    """Explicit per-device compatibility; unspecified devices require code zero."""
    codes = config.get('go2_normal_state_codes', [0])
    if (not isinstance(codes, list) or not codes or
            any(type(code) is not int or not 0 <= code <= 0xffffffff for code in codes)):
        raise ValueError('go2_normal_state_codes must be a nonempty list of uint32 values')
    return frozenset(codes)


class UnitreeHardware:
    def __init__(self, config, execute=False):
        self.config, self.execute = config, execute
        self.go2_normal_codes = go2_normal_state_codes(config)
        self.sensors = Sensors('unitree')
        self.closed = threading.Event()
        self.io_lock = threading.RLock()
        self.processes, self.logs, self.cameras = [], [], {}
        self.piper = self.guard = None
        self.reader = None
        self.run = Path(tempfile.mkdtemp(prefix='heterovla-v3-'))
        self.motion_socket = self.run/'motion.sock'
        self.lock_file = open('/tmp/heterovla-v3-unitree.lock', 'a')
        fcntl.flock(self.lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if execute:
                conflicts = ('piper_chunk_loop.py', 'piper_stream_loop.py', 'hetero_teleop_loop.py',
                             'hetero_teach_loop.py', 'piper_gamepad_teleop.py', 'go2_velocity_bridge',
                             'go2_replay_bridge', 'replay_episode.py')
                for process in Path('/proc').glob('[0-9]*/cmdline'):
                    try:
                        command = process.read_bytes().replace(b'\0', b' ').decode(errors='replace')
                    except OSError:
                        continue
                    if any(name in command for name in conflicts):
                        raise RuntimeError('Another controller is running: PID '+process.parent.name)
            sys.path.insert(0, str(Path(__file__).parent/'vendor'))
            from hetero_pkl_recorder import CameraReader, feedback_piper
            from piper_safety import monitored_piper, SafetyGuard
            from piper_sdk import C_PiperInterface_V2
            self.piper = feedback_piper(monitored_piper(C_PiperInterface_V2))(
                config.get('can', 'can0'), start_sdk_joint_limit=True, start_sdk_gripper_limit=True)
            self.piper.ConnectPort(piper_init=False)
            if execute:
                self.guard = SafetyGuard(self.piper)
                self.guard.acknowledge()
                self.guard.start()
            for name, serial in config['cameras'].items():
                camera = CameraReader(name, serial, 640, 480, 30)
                self.cameras[name] = camera
                camera.start()
            self._spawn('v3_go2_state', [config.get('interface', 'eth0'), str(self.run)])
            if execute:
                self._spawn('v3_go2_motion', [config.get('interface', 'eth0'), str(self.motion_socket),
                                             '.25', '.25', '.5', '200', '3'])
            self.reader = threading.Thread(target=self._read, daemon=True)
            self.reader.start()
        except Exception:
            self.close()
            raise

    def _spawn(self, binary, args):
        folder = Path(self.config.get('build_dir', str(Path(__file__).parent/'build'))).expanduser()
        log = (self.run/(binary+'.log')).open('ab', buffering=0)
        self.logs.append(log)
        self.processes.append(subprocess.Popen([str(folder/binary), *args], stdout=log, stderr=log))

    def _read(self):
        started = time.monotonic()
        while not self.closed.wait(.01):
            self._read_once()
            if self.execute and self.sensors.error and time.monotonic()-started > 5:
                self.halt()
                return

    def _read_once(self):
        # A disabled robot or missing arm feedback must not hide live cameras.
        # Execution still rejects any error; preview reports each stream as received.
        errors = []
        try:
            if any(p.poll() is not None for p in self.processes):
                raise RuntimeError('Go2 bridge exited; inspect '+str(self.run))
            data = json.loads((self.run/'go2_snapshot.json').read_text())['sport_mode_state']
            if not data['valid']:
                raise RuntimeError('Missing Go2 DDS feedback')
            g = np.array([*data['position'], data['rpy'][2], data['body_height'], *data['rpy'][:2]])
            self.sensors.put('go2', data['monotonic_ns']/1e9, g)
            if data['error_code'] not in self.go2_normal_codes:
                raise RuntimeError('Go2 DDS error_code='+str(data['error_code']))
        except Exception as exc:
            errors.append('go2: '+str(exc))
        try:
            arm, stamp = self.arm_feedback()
            self.sensors.put('piper', stamp, arm)
        except Exception as exc:
            errors.append('piper: '+str(exc))
        for name, camera in self.cameras.items():
            try:
                if camera.error:
                    raise RuntimeError(camera.error)
                frame = camera.snapshot()
                if frame is not None:
                    self.sensors.put(name, frame['monotonic_ns']/1e9, frame['image'])
            except Exception as exc:
                errors.append(name+': '+str(exc))
        self.sensors.error = '; '.join(errors) or None

    def arm_feedback(self):
        joints, gripper, _, stamp, _ = self.piper.feedback_snapshot(timeout_ns=75_000_000)
        arm = [math.radians(float(getattr(joints.joint_state, 'joint_'+str(i)))/1000.) for i in range(1, 7)]
        arm.append(float(gripper.gripper_state.grippers_angle)/1_000_000.)
        return np.array(arm), stamp[0]/1e9

    def _go2(self, command):
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(.08)
            client.connect(str(self.motion_socket))
            client.sendall((command+'\n').encode())
            result = client.makefile('r').readline(512).split()
        if len(result) < 2 or result[:2] != ['OK', '0']:
            raise RuntimeError('Go2 command failed: '+str(result))

    def check_execution(self):
        if self.sensors.error:
            raise RuntimeError(self.sensors.error)
        if not self.execute or self.guard is None or self.guard.inhibited:
            raise RuntimeError('Piper execution unavailable or safety fault latched')
        from piper_safety import FEEDBACK_IDS, ENABLE_IDS
        if not self.piper.feedback_fresh(FEEDBACK_IDS+ENABLE_IDS, .25):
            raise RuntimeError('Missing Piper feedback/enable status')
        status = self.piper.GetArmStatus().arm_status
        if status.arm_status != 0 or status.ctrl_mode != 1 or not all(self.piper.GetArmEnableStatus()):
            raise RuntimeError('Piper must already be enabled, normal, and in CAN mode')
        with self.sensors.lock:
            _, go = self.sensors.streams['go2'].past(time.monotonic(), .075)
            if go[4] < .23:
                raise RuntimeError('Go2 is not standing')
        if not self.motion_socket.exists():
            raise RuntimeError('Go2 SportClient lease/socket not ready')
        self.arm_feedback()

    def send(self, row):
        with self.io_lock:
            self.check_execution()
            with self.guard.lock:
                if self.guard.inhibited:
                    raise RuntimeError(self.guard.fault)
                self.guard.active = True
                self.guard.last_tick = time.monotonic()
                joints, grip = piper_sdk_target(row[3:])
                self.guard.call('ModeCtrl', 0x01, 0x01, 15, 0)
                self.guard.call('JointCtrl', *joints)
                self.guard.call('GripperCtrl', grip, 1000, 0x01, 0)
            self._go2('MOVE %.8f %.8f %.8f' % tuple(row[:3]))

    def hold(self):
        with self.io_lock:
            self._go2('STOP')
            arm, _ = self.arm_feedback()
            joints, grip = piper_sdk_target(arm)
            with self.guard.lock:
                if self.guard.inhibited:
                    raise RuntimeError(self.guard.fault)
                self.guard.call('JointCtrl', *joints)
                self.guard.call('GripperCtrl', grip, 1000, 0x01, 0)
                self.guard.active = False

    def halt(self):
        if self.guard is not None:
            self.guard.trip('V3 execution stopped; inspect and restart gateway before resuming')
        if self.motion_socket.exists():
            self._go2('STOP')

    def close(self):
        self.closed.set()
        if self.guard is not None:
            self.guard.close()
        if self.motion_socket.exists():
            try:
                self._go2('QUIT')
            except Exception:
                pass
        for process in self.processes:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if self.reader is not None and self.reader is not threading.current_thread():
            self.reader.join(timeout=1)
        for camera in self.cameras.values():
            camera.stop()
        if self.piper is not None:
            self.piper.DisconnectPort()
        for log in self.logs:
            log.close()
        self.lock_file.close()
