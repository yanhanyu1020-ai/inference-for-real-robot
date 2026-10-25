"""Continuity checks around the existing gamepad controller, without CAN setup."""
import math
import time
import numpy as np
from scipy.spatial.transform import Rotation


def safe_controller(base, joint_only=True):
    class ContinuousTeleop(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.speed_factors = [.25, .5, 1.]
            self.speed_factor_index = 1
            self.movement_speeds = [10, 20, 30]
            self.movement_speed_index = 1
            self.up_level_mode = 'joint' if joint_only else 'pose'
            self.low_level_mode = 'joint'
            self._last_control_tick = time.monotonic()
            self._control_dt = .005
            self._had_gamepad = False
            self.command_inhibited = False
            self.safety_fault = None
            self.ik_error = None
            self._joint_input_active = np.zeros(6, dtype=bool)

        def update(self):
            if joint_only:
                self.up_level_mode = self.low_level_mode = 'joint'
                self.ik_error = None
            now = time.monotonic()
            self._control_dt = min(.1, max(.001, now - self._last_control_tick))
            self._last_control_tick = now
            # Upstream integrates per iteration. Scale by elapsed time so GPU
            # latency does not change the meaning of the joystick speed.
            self.translation_step = .04 * self._control_dt
            self.rotation_step = 10. * self._control_dt
            self.joint_angle_step = math.radians(20.) * self._control_dt
            super().update()

        def _toggle_up_level_mode(self):
            if joint_only:
                self.up_level_mode = 'joint'
                return
            return super()._toggle_up_level_mode()

        def _toggle_low_level_mode(self):
            if joint_only:
                self.low_level_mode = 'joint'
                return
            return super()._toggle_low_level_mode()

        def _update_joint_mode(self):
            left_x = self._apply_deadzone(self._get_axis_value('left_x'))
            left_y = self._apply_deadzone(self._get_axis_value('left_y'))
            right_x = self._apply_deadzone(self._get_axis_value('right_x'))
            right_y = self._apply_deadzone(self._get_axis_value('right_y'))
            hat = self._get_hat_value('dpad')
            active = np.array([left_x, left_y, right_y, hat[0], hat[1], right_x]) != 0
            released = self._joint_input_active & ~active
            if released.any():
                joints = self.interface.GetArmJointMsgs().joint_state
                actual = np.radians([float(getattr(joints, 'joint_' + str(i))) / 1000.
                                     for i in range(1, 7)])
                # Cancel only the released joints' outstanding targets, once.
                self.joint_angles[released] = actual[released]
            self._joint_input_active = active
            super()._update_joint_mode()
            if released.any() and not active.any():
                self._joint_to_pose()

        def _handle_joystick_events(self):
            connected = super()._handle_joystick_events()
            if self._had_gamepad and not connected and self.arm_enabled:
                self.command_inhibited = True
                self.safety_fault = '手柄断连：已禁止发送目标；重连后松开摇杆，按 Home 核对当前姿态后恢复'
            self._had_gamepad = bool(connected)
            return connected

        def _handle_button_events(self, events):
            if any(name in events for name in ('home', 'b', 'y', 'back')):
                self._joint_input_active[:] = False
            if self.command_inhibited:
                if 'home' in events:
                    self._resume_from_feedback()
                return
            super()._handle_button_events(events)

        def _resume_from_feedback(self):
            # Explicit Home acknowledgement after a dropout. Never replay the
            # pre-dropout target and never send a go-home/enable command here.
            try:
                axes = [self._get_axis_value(name) for name in
                        ('left_x', 'left_y', 'right_x', 'right_y')]
                if not all(math.isfinite(x) and abs(x) <= self.deadzone for x in axes):
                    raise ValueError('请先松开摇杆')
                if any(self._get_axis_value(n) > .05 for n in ('left_trigger', 'right_trigger')):
                    raise ValueError('请先松开扳机')
                if self._get_hat_value('dpad') != (0, 0):
                    raise ValueError('请先松开方向键')
                joints = self.interface.GetArmJointMsgs()
                gripper = self.interface.GetArmGripperMsgs()
                now = time.time()
                if not all(0 <= now - float(msg.time_stamp) <= .25 for msg in (joints, gripper)):
                    raise ValueError('CAN 反馈过期，不能恢复遥操')
                q = np.array([math.radians(float(getattr(joints.joint_state, 'joint_' + str(i))) / 1000.)
                              for i in range(1, 7)])
                if not np.isfinite(q).all() or any(not lo <= v <= hi for v, (lo, hi) in zip(q, self.joint_limits)):
                    raise ValueError('关节反馈无效')
                width = float(gripper.gripper_state.grippers_angle) / 1e6
                if not math.isfinite(width) or not 0 <= width <= self.gripper_max_width:
                    raise ValueError('夹爪反馈无效')
                self.joint_angles = q
                self.gripper_state = width / self.gripper_max_width * 100.
                self._joint_to_pose()
                self.command_inhibited = False
                self.safety_fault = self.ik_error = None
            except Exception as exc:
                self.safety_fault = str(exc)

        def _pose_to_joint(self, xyz, orientation):
            if joint_only:
                return
            previous = self.joint_angles.copy()
            quat = orientation.as_quat()
            wxyz = [quat[3], *quat[:3]]
            try:
                result = self.kinematic.solve_ik(xyz, wxyz, initial_guess=previous.tolist())
                if result is None:
                    raise ValueError('目标无连续逆解，请减小移动或调整姿态')
                q = np.asarray(result[:6], dtype=float)
                if q.shape != (6,) or not np.isfinite(q).all():
                    raise ValueError('逆解返回无效关节角')
                if any(not lo <= v <= hi for v, (lo, hi) in zip(q, self.joint_limits)):
                    raise ValueError('逆解超出关节限位')
                limit = math.radians(20.) * self._control_dt
                if np.any(np.abs(q - previous) > limit):
                    raise ValueError('逆解关节跳变，已保留上一目标')
                actual = np.asarray(self.kinematic.solve_fk(q.tolist()), dtype=float)
                if actual.shape != (7,) or not np.isfinite(actual).all():
                    raise ValueError('正运动学返回无效位姿')
                achieved = Rotation.from_quat([*actual[4:], actual[3]])
                if np.linalg.norm(actual[:3] - xyz) > .0011 or (achieved.inv() * orientation).magnitude() > .01:
                    raise ValueError('逆解位姿残差过大')
                self.joint_angles = q
                # Accumulate the requested Cartesian target: replacing it with
                # FK each tick can swallow steps below the IK solver tolerance.
                self.xyz_wxyz = np.concatenate((xyz, wxyz))
                self.xyz_rpy = np.concatenate((xyz, orientation.as_euler('xyz', degrees=True)))
                self.ik_error = None
            except Exception as exc:
                self.ik_error = str(exc)
    return ContinuousTeleop
