"""Replay IK branch flips without constructing a CAN or joystick interface."""
import numpy as np
from scipy.spatial.transform import Rotation
from types import SimpleNamespace


def controller(candidate):
    from safe_teleop import safe_controller
    class Base:
        pass
    ctl = safe_controller(Base, joint_only=False)()
    ctl.joint_angles = np.zeros(6)
    ctl.joint_limits = [(-3, 3)] * 6
    ctl.xyz_wxyz = np.array([0., 0., 0., 1., 0., 0., 0.])
    ctl.xyz_rpy = np.zeros(6)
    ctl._control_dt = .1
    ctl.ik_error = None
    ctl.seeds = []
    def solve(xyz, wxyz, initial_guess=None):
        ctl.seeds.append(initial_guess)
        return candidate
    ctl.kinematic = SimpleNamespace(solve_ik=solve,
        solve_fk=lambda q: [0., 0., 0., 1., 0., 0., 0.])
    return ctl


def test_real_recorded_wrist_branch_flip_is_rejected():
    ctl = controller([0, 0, 0, 1.104, 0, -1.104])
    ctl._pose_to_joint(np.zeros(3), Rotation.identity())
    assert np.array_equal(ctl.joint_angles, np.zeros(6))
    assert ctl.ik_error


def test_joint_only_default_blocks_both_mode_switches_and_ik():
    from safe_teleop import safe_controller
    class Base:
        def update(self):
            self._toggle_up_level_mode()
            self._toggle_low_level_mode()
            assert self.up_level_mode == self.low_level_mode == 'joint'
    ctl = safe_controller(Base)()
    assert ctl.up_level_mode == ctl.low_level_mode == 'joint'
    ctl._pose_to_joint(None, None)  # No solver or pose state is needed.
    ctl.up_level_mode = ctl.low_level_mode = 'pose'
    ctl.ik_error = 'old error'
    ctl.update()
    assert ctl.up_level_mode == ctl.low_level_mode == 'joint'
    assert ctl.ik_error is None


def test_continuous_solution_uses_previous_joint_seed():
    ctl = controller([.002, 0, 0, .003, 0, -.003])
    ctl._pose_to_joint(np.zeros(3), Rotation.identity())
    assert ctl.joint_angles[3] == .003
    assert ctl.seeds == [[0., 0., 0., 0., 0., 0.]]
    assert ctl.ik_error is None


def test_nonfinite_solution_is_rejected():
    ctl = controller([0, 0, float('nan'), 0, 0, 0])
    ctl._pose_to_joint(np.zeros(3), Rotation.identity())
    assert np.isfinite(ctl.joint_angles).all()
    assert ctl.ik_error


def test_disconnect_latches_even_after_receiver_returns():
    from safe_teleop import safe_controller
    class Base:
        def _handle_joystick_events(self):
            return self.now_connected
        def _handle_button_events(self, events):
            raise AssertionError('must not send Home/go-home while inhibited')
    ctl = safe_controller(Base)()
    ctl._had_gamepad = True
    ctl.arm_enabled = True
    ctl.now_connected = False
    assert not ctl._handle_joystick_events()
    ctl.now_connected = True
    assert ctl._handle_joystick_events()
    assert ctl.command_inhibited
    ctl._handle_button_events({'y': True})
    assert ctl.command_inhibited


def test_reconnect_requires_fresh_feedback_and_does_not_replay_old_target():
    import time
    ctl = controller([0] * 6)
    ctl.command_inhibited = True
    ctl.deadzone = .1
    ctl.gripper_max_width = .07
    ctl._get_axis_value = lambda name: 0.
    ctl._get_hat_value = lambda name: (0, 0)
    ctl._joint_to_pose = lambda: None
    joints = SimpleNamespace(time_stamp=0, joint_state=SimpleNamespace(**{'joint_'+str(i): 1000 for i in range(1,7)}))
    grip = SimpleNamespace(time_stamp=0, gripper_state=SimpleNamespace(grippers_angle=35000))
    ctl.interface = SimpleNamespace(GetArmJointMsgs=lambda: joints, GetArmGripperMsgs=lambda: grip)
    ctl._resume_from_feedback()
    assert ctl.command_inhibited
    joints.time_stamp = grip.time_stamp = time.time()
    ctl._resume_from_feedback()
    assert not ctl.command_inhibited
    np.testing.assert_allclose(ctl.joint_angles, np.radians([1] * 6))
    assert ctl.gripper_state == 50.
