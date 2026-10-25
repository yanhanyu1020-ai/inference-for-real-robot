# HeteroVLA

Data-efficient vision-language-action learning for heterogeneous robot
coordination. The initial platform combines a Unitree Go2 mobile base, an
AgileX Piper arm, and an aerial or external vision module.

[中文说明](README_zh.md)

## Execution environments

This repository contains source code for multiple computers. A source file may
be checked out on the GPU server for version control without being executable
there.

| Environment | Target system | Responsibilities | Repository paths |
| --- | --- | --- | --- |
| Robot onboard runtime | Unitree Go2 Jetson, Ubuntu 20.04, aarch64 | DDS/SDK data acquisition, robot adapters, local safety, action execution | `onboard/unitree_go2/`, `robot_gateway/` |
| GPU server runtime | A800 (meta-eon-a800-006), x86_64, NVIDIA GPUs | Dataset conversion, training, policy serving, evaluation | `heterovla/`, `data_tools/`, `scripts/` |
| Operator workstation | A computer that can SSH to the robot and the GPU server | Deployment, episode control, data transfer | `scripts/` |

**Code under `onboard/unitree_go2/` and `robot_gateway/` is deployed to and
runs on the robot's Jetson. It is not a GPU-server runtime component.** The GPU
server only keeps a repository checkout of that source so it can be reviewed,
versioned, and pushed to GitHub.

## Repository layout

- `onboard/unitree_go2`: software whose execution target is the Go2 onboard
  Jetson (command agent, Go2/Piper command executors, recorder)
- `robot_gateway`: Piper streaming executors (`piper_stream_loop.py` for
  policy-server streaming, `piper_chunk_loop.py` for chunked remote control)
- `heterovla`: protocol definitions and control broker (`control_protocol.py`,
  `control_server.py`)
- `scripts`: deployment and operator utilities (`robotctl.sh`, `control_up.sh`,
  `mock_vla_server.py`, `send_go2_trajectory.py`, `send_hetero_demo.py`)
- `tests`: protocol tests
- `docs`: operator documentation (`control_channel_zh.md`,
  `trajectory_control_zh.md` for writing trajectories end-to-end)
- `third_party/openpi`: upstream openpi integration (planned)

## Current status

The remote control and streaming chain is working end to end:

```
A800 (mock VLA / future pi0.5 policy server, ws://127.0.0.1:8765)
        │  action chunks (openpi msgpack protocol, via SSH tunnel 18765)
        ▼
Go2 Jetson: robot_command_agent (persistent, ws://127.0.0.1:18766 broker)
        │  streams ModeCtrl+JointCtrl+GripperCtrl at 200 Hz
        ▼
Piper arm: joints + gripper
```

Go2 velocity streaming works via a persistent C++ bridge that holds the
robot's control lease (see below).

### Network setup

Two SSH tunnels on the Go2 Jetson forward the GPU server's services to
local ports (also done by `control_up.sh`, with a watchdog that restarts
the tunnel when the connection drops):

```bash
ssh -F ~/heterovla-control/ssh_config -fNT \
  -L 127.0.0.1:18766:127.0.0.1:8766 \   # control broker
  -L 127.0.0.1:18765:127.0.0.1:8765 \   # policy server
  heterovla-a800
```

One-command onboard bring-up: `~/heterovla-control/control_up.sh` (CAN,
tunnel watchdog with both forwards, agent with a clean environment).

### Operator commands (from A800)

```bash
bash scripts/robotctl.sh <command>
```

- Go2: `go2.stand_up`, `go2.stand_down`, `go2.stop`
- Piper: `piper.enable`, `piper.disable`, `piper.go_zero` (safe-home pose),
  `piper.open_gripper`, `piper.close_gripper`, `piper.stop`
- Streaming: `system.stream_start/stop/status` (runs `piper_stream_loop.py`
  against the policy server), `system.control_start/stop/status` (persistent
  chunk executor), `piper.action_chunk` with a JSON `actions` payload
- Go2 velocity: `system.go2_control_start/stop/status` (persistent C++
  bridge), `system.go2_stream_start/stop/status` (policy streaming),
  `go2.action_chunk` with a JSON `actions` payload
- Safety: `system.stop_all`
- Capture: `system.record_start/stop/status`, joystick teleop
  (`system.teleop_*`), drag-teach (`system.teach_*`; Piper read-only,
  left stick still drives the dog)

Note: `robotctl.sh` strips http(s)/all proxy environment variables before
connecting to the local broker (websockets 15 honours proxy env vars and
would otherwise send the local connection through the git proxy).

### Mock policy server (A800)

```bash
python3 scripts/mock_vla_server.py                 # arm: 7-dim trajectory
python3 scripts/mock_vla_server.py --robot dog     # dog: 3-dim [vx,vy,vyaw]
```

### Piper arm streaming

Mock VLA on A800 -> websocket -> `piper_stream_loop.py` on the Jetson,
which executes each action row at 200 Hz:

```bash
# A800
bash scripts/robotctl.sh system.stream_start       # start the streamer
bash scripts/robotctl.sh system.stream_status      # tail its log
bash scripts/robotctl.sh system.stream_stop
```

### Go2 locomotion: end-to-end flow (recommended from A800)

Pipeline:

```
A800: send_go2_trajectory.py / robotctl
        │  go2.action_chunk  [vx, vy, vyaw] (control broker 18766↔8766)
        ▼
Jetson: go2_chunk_loop.py (forwards one row every 50 ms)
        │  unix socket → MOVE / STAND_UP / STOP
        ▼
Jetson: go2_velocity_bridge (persistent; holds SportClient lease)
        ▼
Go2 body motion
```

#### 0. Prerequisites

1. **Turn the handheld remote off** (otherwise it holds the lease; Move can
   return success while the dog ignores it — errors 3205/3207).
2. Control server on A800 is up:
   `bash scripts/control_server_ctl.sh status` (listening on `127.0.0.1:8766`).
3. Jetson tunnels + agent are online (`~/heterovla-control/control_up.sh` on
   the dog). On A800:
   ```bash
   bash scripts/robotctl.sh system.status
   # expect: hostname=ubuntu execute=True; logs show robot connected: unitree-go2-01
   ```
4. Note: `system.control_start` is for the **Piper arm**; the dog needs
   **`system.go2_control_start`**.

#### 1. Start the Go2 control stack (lease bridge + chunk executor)

On **A800**, from the repo root:

```bash
bash scripts/robotctl.sh system.go2_control_start
bash scripts/robotctl.sh system.go2_control_status
# expect: bridge_pid=... chunk_pid=... (neither is none)
```

This starts `go2_velocity_bridge` (lease holder) and `go2_chunk_loop.py` on
the Jetson. Do **not** run `./go2_velocity_bridge` on A800 (that binary lives
on the dog under `~/heterovla-control/`).

#### 2. Stand up first (`send_go2_trajectory.py` does not auto-stand)

Velocity chunks are only `[vx,vy,vyaw]`; Move while lying down does nothing.
Stand through the **lease-holding bridge**. After `go2_control_start`,
`robotctl go2.stand_up` / `go2.stand_down` write to `go2.sock` and must not
spawn a second `go2_command_executor`.

```bash
# A800 (bridge already running)
bash scripts/robotctl.sh go2.stand_up
# or on the Jetson
python3 ~/heterovla-control/go2_send.py --stand-up
# or clean env: ~/heterovla-control/go2ctl.sh stand_up
```

#### 3. Push a velocity trajectory from A800

Each row is 3-D `[vx, vy, vyaw]` (m/s, rad/s), 1..5000 rows, default 50 ms/row.
`ok: true / chunk delivered` only means the chunk was **handed to the onboard
executor**; watch the dog for the full duration (e.g. `--seconds 2` → ~2 more
seconds). Confirm with status:

```bash
# Clearest smoke test: constant forward for 2 s (~0.6 m)
python3 scripts/send_go2_trajectory.py --vx 0.3 --seconds 2

# Built-in demo (forward hold → stop → yaw → stop → back → stop, ~2 s each;
# --ensure-control also ensures the control stack is up)
python3 scripts/send_go2_trajectory.py --ensure-control

# Custom JSON (same shape as piper.action_chunk, 3-D velocities)
python3 scripts/send_go2_trajectory.py --file dog_traj.json
# equivalent: bash scripts/robotctl.sh go2.action_chunk "$(cat dog_traj.json)"

# Immediately check onboard logs (Move failed? chunk ran full duration?)
bash scripts/robotctl.sh system.go2_control_status
```

Hard clamps: +-0.5 m/s, +-1 rad/s; the bridge STOPs after 400 ms with no new
command.

#### 3b. Go2 + Piper coordinated demo (from A800)

The dog velocity channel and the arm joint channel are independent and can
run at the same time. The default scene below is **serial**: the arm starts
only after walking finishes and the lie-down hold completes (the
chunk-received log ``after=stand_down`` is not "done").

1. Forward 2 s (`vx=0.3`)
2. Yaw right ~90 deg (open-loop, `vyaw=-0.5` rad/s × π s; slip will bias the angle)
3. Lie down (`go2.stand_down` through the **lease-holding bridge**)
4. Scripted Piper grasp (reach → close → lift; not vision)

Redeploy by **pulling** the new `go2_chunk_loop.py` on the Jetson (A800
cannot resolve hostname `unitree`). Do **not** use `robotctl go2.stand_up`
while the bridge holds the lease (it spawns `go2_command_executor`, times
out at 12 s, and the dog does not move).

```bash
# on the dog
scp -F ~/heterovla-control/ssh_config \
  heterovla-a800:~/heterovla/onboard/unitree_go2/control/go2_chunk_loop.py \
  ~/heterovla-control/go2_chunk_loop.py

# then on A800 (reloads the chunk script; no agent restart required)
bash scripts/robotctl.sh system.go2_control_stop
bash scripts/robotctl.sh system.go2_control_start
```

```bash
python3 scripts/send_hetero_demo.py --ensure-control
python3 scripts/send_hetero_demo.py --no-stand-up
python3 scripts/send_hetero_demo.py --skip-arm
python3 scripts/send_hetero_demo.py --skip-dog
python3 scripts/send_hetero_demo.py --dog-file dog_traj.json --arm-file grasp.json
```

E-stop: `bash scripts/robotctl.sh system.stop_all`. Stop `system.stream_start`
before this demo (`system.control_start` and the streamer are mutually exclusive).

#### 4. Stop / tear down

```bash
bash scripts/robotctl.sh go2.stop          # via separate executor; prefer bridge below
# Jetson: python3 ~/heterovla-control/go2_send.py --stop
# Jetson: python3 ~/heterovla-control/go2_send.py --stand-down
bash scripts/robotctl.sh system.go2_control_stop   # stop chunk + bridge (release lease)
bash scripts/robotctl.sh system.stop_all           # emergency stop everything
```

#### Policy streaming (optional, mock VLA → `go2_stream_loop.py`)

```bash
# A800
python3 scripts/mock_vla_server.py --robot dog
bash scripts/robotctl.sh system.go2_control_start
bash scripts/robotctl.sh system.go2_stream_start
bash scripts/robotctl.sh system.go2_stream_status
bash scripts/robotctl.sh system.go2_stream_stop
```

### Go2 velocity control (two terminals, Jetson-only debug)

Without A800 `robotctl`, on the dog directly:

Terminal 1 - lease holder (keep running):
```bash
cd ~/heterovla-control && ./go2_velocity_bridge eth0 go2.sock
```

Terminal 2 - send commands:
```bash
python3 ~/heterovla-control/go2_send.py --stand-up
python3 ~/heterovla-control/go2_send.py --vx 0.5 --seconds 2
python3 ~/heterovla-control/go2_send.py --stand-down
python3 ~/heterovla-control/go2_send.py --stop
```

Monitor: `go2_monitor eth0` (`rt/wirelesscontroller` + `rt/sportmodestate`).
SDK binaries need a clean environment (sourcing cyclonedds_ws/ROS →
`free(): invalid pointer`); use `~/heterovla-control/go2ctl.sh stand_up`.

## Engineering notes (Piper arm, learned the hard way)

1. **Gripper position control needs the arm in motion mode 0x01**, re-issued
   every frame via `ModeCtrl(0x01, 0x01, speed, 0x00)`, plus a sustained
   200 Hz stream at effort 3000 (3 N·m). Without it the gripper ignores
   commands (effort ~0) and appears dead.
2. **ModeCtrl speed matters for tracking changing targets**: speed 20 cannot
   keep up with a moving target, the controller reports
   `REACH_TARGET_POS_FAILED` and auto-disables the arm. Use speed 100.
3. **Exact zero pose sits on joint limits** (J2 min 0 deg, J3 max 0 deg).
   Commanding exact zeros triggers the same failure and drops the arm.
   `piper.go_zero` targets a safe offset home `[0, 0.5, -0.5, 0, 0.5, 0]` deg.
   Same for the gripper: never command 0 mm (physical closed stop reads
   ~0.2 mm); use >= 1 mm.
4. **Single writer rule**: only one process may stream Piper commands at a
   time. Two concurrent 200 Hz streams fight and can drop or jam the arm.
   `system.stream_start` / `system.control_start` reject each other.
5. **Controller state can be stale** (motors report enabled while the
   controller is limp in STANDBY). The streaming executors recover via a
   disable->enable cycle; a liveness check aborts if the arm does not respond
   within 30 steps.
6. **Enable once is enough at normal poses**: after a stream exits cleanly the
   arm keeps holding; do not re-enable an already healthy arm.

## Engineering notes (Go2, learned the hard way)

1. **Motion requires the control lease.** `SportClient(true)` +
   `WaitLeaseApplied()`; without a lease Move() returns rc=0 but the dog
   ignores it. Stance commands (StandUp/StandDown) need the lease too once
   the lease service is active (error 3205 "Request denied by lease"; 3207
   "Lease already exists" when another holder - e.g. the remote - has it).
   One persistent process must hold the lease (two-terminal pattern).
2. **Turn the handheld remote off for SDK control** - it takes the lease and
   its lock state gates locomotion.
3. **Clean environment required for SDK binaries** (interactive shells that
   source cyclonedds_ws/ROS crash with `free(): invalid pointer`).
4. **The remote's raw inputs** are on `rt/wirelesscontroller` (lx/ly/rx/ry/
   keys); dog state on `rt/sportmodestate` (mode stays 0/IDLE even while
   walking - do not gate on mode, read velocity feedback instead).
5. **Stand via the bridge, not a second executor.** After `go2_control_start`
   the lease is on `go2_velocity_bridge`. The updated agent routes
   `robotctl go2.stand_up` / `go2.stand_down` through the bridge; an old
   agent still spawns `go2_command_executor` and can conflict. You can also
   use `go2_send.py --stand-up`. `send_go2_trajectory.py` does not auto-stand;
   `send_hetero_demo.py` stands by default.
6. **`chunk delivered` ≠ motion finished.** A800 returns immediately; the
   robot keeps running at 50 ms/row. Use `--vx 0.3 --seconds 2` as the
   clearest smoke test; `system.control_start` is the arm, not the dog.

## Data policy

Raw episodes, videos, rosbag files, model checkpoints, credentials, device
addresses, and compiled binaries must not be committed. The private repository
contains source code and documentation, not collected robot data or secrets.
