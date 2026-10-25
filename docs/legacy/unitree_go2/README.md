# Unitree Go2 onboard software

[中文说明](README_zh.md)

> **Execution target:** the Jetson computer physically installed on the
> Unitree Go2 (`Ubuntu 20.04`, `aarch64`). This directory is versioned in the
> repository checkout on h102, but its programs are deployed to and executed
> on the robot. They are not run on h102.

## Current component: SDK2 telemetry recorder

The recorder directly subscribes to Unitree SDK2 DDS topics:

- `rt/wirelesscontroller`
- `rt/sportmodestate`
- `rt/lowstate`

It records controller input, body position and velocity, IMU data, joint state,
estimated torque, motor temperature, foot force, and the raw wireless-controller
payload. Every sample receives both monotonic and wall-clock nanosecond
timestamps.

The recorder is receive-only: it creates no command publisher and does not
control the robot.

## Robot-side requirements

- Ubuntu 20.04 on aarch64
- Unitree SDK2 2.0.0 source tree, default:
  `/home/unitree/unitree_sdk2-main`
- Robot DDS network available through `eth0`
- CMake and a C++17 compiler

## Deployment

Run the deployment script from an operator workstation that can SSH to the
robot:

```bash
GO2_HOST=unitree ./scripts/deploy_unitree.sh
```

The default robot-side installation directory is:

```text
/home/unitree/heterovla-recorder
```

The deployment script writes the repository commit to
`source_git_commit.txt`. New episodes copy this value into their metadata so a
dataset can be traced back to the recorder source revision.

## Recording one episode

On the robot:

```bash
ssh unitree
cd /home/unitree/heterovla-recorder
./go2_capture_ctl.sh start task_001 "stand up and lie down"
```

Operate the robot with its physical remote controller, then stop:

```bash
./go2_capture_ctl.sh stop
```

Check status at any time:

```bash
./go2_capture_ctl.sh status
```

Raw data is stored on the robot under:

```text
/home/unitree/heterovla-data/raw/<episode_id>/
```

Episode names are immutable: the control script refuses to overwrite an
existing directory.

## Output files

- `wireless_controller.csv`: joystick axes and key bitmask
- `sport_mode_state.csv`: body pose, velocity, gait state, IMU and foot force
- `low_state.csv`: high-rate IMU, joint, motor, power and raw controller state
- `instruction.txt`: task instruction
- `start_time.txt`, `stop_time.txt`: wall-clock boundaries
- `source_git_commit.txt`: recorder source revision, when deployed from Git
- `summary.json`: duration and message counts
- `recorder.log`: runtime health and cumulative message counts

These asynchronous raw streams must be preserved. Resampling and conversion
for LeRobot/openpi are server-side jobs and must not run in the real-time
recording callbacks.
