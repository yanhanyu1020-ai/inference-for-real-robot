#!/usr/bin/env python3
"""Outbound command-channel agent for the Go2 onboard computer.

Dry-run is the default.  Hardware execution requires an explicit ``--execute``.
"""

import argparse
import asyncio
import json
import logging
import os
import socket
import subprocess
import sys
import time

import websockets


PROTOCOL_VERSION = 1
LOG = logging.getLogger("heterovla.robot_agent")


class RobotAgent:
    def __init__(self, args):
        self.args = args
        self.last_command = None

    async def run_forever(self):
        delay = 1
        while True:
            try:
                await self._session()
                delay = 1
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                LOG.warning("control connection lost: %s", exc)
                if self.args.execute:
                    await self._stop_all("control connection lost")
                await asyncio.sleep(delay)
                delay = min(delay * 2, 15)

    async def _session(self):
        LOG.info("connecting to %s", self.args.uri)
        async with websockets.connect(
            self.args.uri, open_timeout=10,             ping_interval=10, ping_timeout=10,
            max_size=1024 * 1024,
        ) as websocket:
            await websocket.send(json.dumps({
                "type": "register",
                "protocol": PROTOCOL_VERSION,
                "role": "robot",
                "robot_id": self.args.robot_id,
                "token": self.args.token,
                "hostname": socket.gethostname(),
                "execute": self.args.execute,
            }))
            response = json.loads(await websocket.recv())
            if response.get("type") != "registered":
                raise RuntimeError(response.get("error", "registration failed"))
            LOG.info("registered as %s (execute=%s)", self.args.robot_id, self.args.execute)
            heartbeat = asyncio.create_task(self._heartbeat(websocket))
            try:
                async for payload in websocket:
                    message = json.loads(payload)
                    if message.get("type") != "command":
                        continue
                    result = await self._execute(message)
                    await websocket.send(json.dumps(result))
            finally:
                heartbeat.cancel()

    async def _heartbeat(self, websocket):
        while True:
            await asyncio.sleep(5)
            await websocket.send(json.dumps({
                "type": "heartbeat", "robot_id": self.args.robot_id,
                "monotonic": time.monotonic(), "execute": self.args.execute,
            }))

    async def _execute(self, message):
        request_id = message.get("id", "")
        command = message.get("command", "")
        self.last_command = command
        started = time.monotonic()
        try:
            if command == "system.ping":
                output = "pong"
            elif command == "system.status":
                output = "hostname=%s execute=%s last_command=%s" % (
                    socket.gethostname(), self.args.execute, self.last_command)
            elif not self.args.execute:
                output = "dry-run: would execute %s" % command
                LOG.info(output)
            elif command == "system.stop_all":
                output = await self._stop_all("operator command")
            elif command == "system.stream_start":
                output = await self._stream_start()
            elif command == "system.stream_stop":
                output = await self._stream_stop()
            elif command == "system.stream_status":
                output = await self._stream_status()
            elif command == "system.control_start":
                output = await self._control_start()
            elif command == "system.control_stop":
                output = await self._control_stop()
            elif command == "system.control_status":
                output = await self._control_status()
            elif command == "piper.action_chunk":
                output = await self._action_chunk(message.get("arguments", {}))
            elif command == "go2.action_chunk":
                output = await self._go2_action_chunk(message.get("arguments", {}))
            elif command == "system.go2_control_start":
                output = await self._go2_control_start()
            elif command == "system.go2_control_stop":
                output = await self._go2_control_stop()
            elif command == "system.go2_control_status":
                output = await self._go2_control_status()
            elif command == "system.go2_stream_start":
                output = await self._go2_stream_start()
            elif command == "system.go2_stream_stop":
                output = await self._go2_stream_stop()
            elif command == "system.go2_stream_status":
                output = await self._go2_stream_status()
            elif command == "system.record_start":
                output = await self._record_start(message.get("arguments", {}))
            elif command == "system.record_stop":
                output = await self._record_stop()
            elif command == "system.record_status":
                output = await self._record_status()
            elif command == "system.teleop_start":
                output = await self._teleop_start()
            elif command == "system.teleop_stop":
                output = await self._teleop_stop()
            elif command == "system.teleop_status":
                output = await self._teleop_status()
            elif command == "system.teach_start":
                output = await self._teach_start()
            elif command == "system.teach_stop":
                output = await self._teach_stop()
            elif command == "system.teach_status":
                output = await self._teach_status()
            elif command in ("go2.stand_up", "go2.stand_down", "go2.stop"):
                output = await self._go2_pose_command(command)
            elif command.startswith("go2."):
                output = await self._run([
                    self.args.go2_executor, self.args.network_interface,
                    command.split(".", 1)[1],
                ])
            elif command.startswith("piper."):
                output = await self._run([
                    self.args.python, self.args.piper_executor,
                    "--can", self.args.piper_can, command.split(".", 1)[1],
                ])
            else:
                raise ValueError("command is not supported by this agent")
            return {"type": "result", "id": request_id, "ok": True,
                    "output": output, "elapsed_s": time.monotonic() - started,
                    "dry_run": not self.args.execute}
        except Exception as exc:
            LOG.exception("command failed: %s", command)
            return {"type": "result", "id": request_id, "ok": False,
                    "error": str(exc), "elapsed_s": time.monotonic() - started,
                    "dry_run": not self.args.execute}

    async def _stop_all(self, reason):
        LOG.warning("stopping Go2 and Piper: %s", reason)
        results = []
        try:
            results.append(await self._record_stop())
        except Exception as exc:
            results.append("record_stop failed: %s" % exc)
        try:
            results.append(await self._teleop_stop())
        except Exception as exc:
            results.append("teleop_stop failed: %s" % exc)
        try:
            results.append(await self._teach_stop())
        except Exception as exc:
            results.append("teach_stop failed: %s" % exc)
        try:
            results.append(await self._go2_control_stop())
        except Exception as exc:
            results.append("go2_control_stop failed: %s" % exc)
        try:
            results.append(await self._go2_stream_stop())
        except Exception as exc:
            results.append("go2_stream_stop failed: %s" % exc)
        # Do not spawn go2_command_executor here: it grabs a fresh SportClient
        # lease and often aborts (3205/3207 / FATAL), which blocks the remote
        # after the bridge is already stopped.  Bridge/chunk STOP is enough.
        try:
            results.append(await self._run([
                self.args.python, self.args.piper_executor, "--can",
                self.args.piper_can, "stop",
            ]))
        except Exception as exc:
            results.append("%s failed: %s" % (self.args.piper_executor, exc))
        return "; ".join(results)

    async def _run(self, argv):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._run_blocking, argv)

    def _run_blocking(self, argv):
        completed = subprocess.run(
            argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            universal_newlines=True, timeout=self.args.command_timeout,
        )
        output = completed.stdout.strip()
        if completed.returncode != 0:
            raise RuntimeError("exit %d: %s" % (completed.returncode, output))
        return output or "ok"

    STREAM_PATTERN = "piper_stream_loop.py"
    CHUNK_PATTERN = "piper_chunk_loop.py"
    TELEOP_PATTERN = "hetero_teleop_loop.py"
    TEACH_PATTERN = "hetero_teach_loop.py"

    def _stream_pid(self):
        return self._pgrep_first(self.STREAM_PATTERN)

    def _chunk_pid(self):
        return self._pgrep_first(self.CHUNK_PATTERN)

    def _teleop_pid(self):
        return self._pgrep_first(self.TELEOP_PATTERN)

    def _teach_pid(self):
        return self._pgrep_first(self.TEACH_PATTERN)

    def _pgrep_first(self, pattern):
        try:
            out = subprocess.run(
                ["pgrep", "-f", pattern],
                capture_output=True, text=True, timeout=5,
            ).stdout.strip()
        except subprocess.TimeoutExpired:
            return None
        lines = [line for line in out.splitlines() if line != str(os.getpid())]
        return lines[0] if lines else None

    def _pid_alive(self, pid):
        try:
            os.kill(int(pid), 0)
        except (OSError, ValueError):
            return False
        return True

    async def _kill_and_wait(self, pid, timeout=2.0):
        """TERM, then KILL if the process is still alive.

        The velocity bridge can hang in SportClient::StopMove on SIGTERM
        (unitree RecurrentThread abort), which keeps the sport lease.
        """
        if not pid:
            return "missing pid"
        subprocess.run(["kill", "-TERM", str(pid)], timeout=5)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not self._pid_alive(pid):
                return "stopped pid %s" % pid
            await asyncio.sleep(0.1)
        subprocess.run(["kill", "-KILL", str(pid)], timeout=5)
        await asyncio.sleep(0.15)
        if self._pid_alive(pid):
            return "failed to kill pid %s" % pid
        return "killed pid %s after TERM hang" % pid

    async def _stream_start(self):
        """Start the policy streamer locally.  Single-writer rule: reject if
        any Piper command stream is already running (two concurrent streams
        fight and can drop or jam the arm)."""
        pid = self._stream_pid()
        if pid:
            return "stream already running (pid %s)" % pid
        if self._chunk_pid():
            return "chunk executor running; stop it first (system.control_stop)"
        if self._teleop_pid():
            return "teleop running; stop it first (system.teleop_stop)"
        if self._teach_pid():
            return "teach capture running; stop it first (system.teach_stop)"
        here = os.path.dirname(os.path.realpath(__file__))
        log_file = os.path.join(here, "stream.log")
        # Use --python (system interpreter with piper_sdk), not the agent venv.
        argv = [
            self.args.python, self._gateway_script("piper_stream_loop.py"),
            "--host", "127.0.0.1", "--port", "18765",
        ]
        with open(log_file, "w") as handle:
            proc = subprocess.Popen(
                argv, stdout=handle, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        return "stream started (pid %d, log %s)" % (proc.pid, log_file)

    async def _stream_stop(self):
        pid = self._stream_pid()
        if not pid:
            return "no stream running"
        subprocess.run(["kill", "-TERM", pid], timeout=5)
        return "stop signal sent to pid %s (arm holds last pose)" % pid

    async def _stream_status(self):
        pid = self._stream_pid()
        here = os.path.dirname(os.path.realpath(__file__))
        log_file = os.path.join(here, "stream.log")
        tail = ""
        try:
            with open(log_file) as handle:
                tail = "".join(handle.readlines()[-8:]).strip()
        except OSError:
            tail = "(no log yet)"
        return "pid=%s\n%s" % (pid or "none", tail)

    async def _control_start(self):
        pid = self._chunk_pid()
        if pid:
            return "chunk executor already running (pid %s)" % pid
        if self._stream_pid():
            return "streamer running; stop it first (system.stream_stop)"
        if self._teleop_pid():
            return "teleop running; stop it first (system.teleop_stop)"
        if self._teach_pid():
            return "teach capture running; stop it first (system.teach_stop)"
        here = os.path.dirname(os.path.realpath(__file__))
        log_file = os.path.join(here, "chunk.log")
        # Same interpreter as piper.go_zero.  Agent venv has no piper_sdk.
        argv = [
            self.args.python, self._gateway_script("piper_chunk_loop.py"),
            "--socket", os.path.join(here, "chunk.sock"),
            "--episode-pointer", self._active_episode_path(),
        ]
        with open(log_file, "w") as handle:
            proc = subprocess.Popen(
                argv, stdout=handle, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        return "chunk executor started (pid %d, log %s)" % (proc.pid, log_file)

    async def _control_stop(self):
        pid = self._chunk_pid()
        if not pid:
            return "no chunk executor running"
        subprocess.run(["kill", "-TERM", pid], timeout=5)
        return "stop signal sent to pid %s (arm holds last pose)" % pid

    async def _control_status(self):
        pid = self._chunk_pid()
        here = os.path.dirname(os.path.realpath(__file__))
        log_file = os.path.join(here, "chunk.log")
        tail = ""
        try:
            with open(log_file) as handle:
                tail = "".join(handle.readlines()[-8:]).strip()
        except OSError:
            tail = "(no log yet)"
        return "pid=%s\n%s" % (pid or "none", tail)

    async def _action_chunk(self, arguments):
        pid = self._chunk_pid()
        if not pid:
            raise RuntimeError("chunk executor not running (system.control_start first)")
        sock_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "chunk.sock")
        payload = json.dumps(arguments).encode()
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            try:
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                sock.connect(sock_path)
                sock.sendall(payload)
                sock.close()
                return "chunk delivered (%d rows)" % len(arguments.get("actions", []))
            except (FileNotFoundError, ConnectionRefusedError):
                await asyncio.sleep(0.05)
        raise RuntimeError("chunk executor socket not ready")

    GO2_BRIDGE_PATTERN = "go2_velocity_bridge"
    GO2_CHUNK_PATTERN = "go2_chunk_loop.py"
    GO2_STREAM_PATTERN = "go2_stream_loop.py"

    def _go2_bridge_pid(self):
        return self._pgrep_first(self.GO2_BRIDGE_PATTERN)

    def _go2_chunk_pid(self):
        return self._pgrep_first(self.GO2_CHUNK_PATTERN)

    def _go2_stream_pid(self):
        return self._pgrep_first(self.GO2_STREAM_PATTERN)

    def _here(self):
        return os.path.dirname(os.path.realpath(__file__))

    def _gateway_script(self, name):
        # Deployment flattens gateway files beside the agent; the checkout keeps
        # their implementations in inference/onboard/robot_gateway.
        deployed = os.path.join(self._here(), name)
        if os.path.isfile(deployed):
            return deployed
        return os.path.normpath(os.path.join(
            self._here(), "..", "..", "robot_gateway", name))

    def _spawn(self, argv, log_name):
        log_file = os.path.join(self._here(), log_name)
        with open(log_file, "a") as handle:
            proc = subprocess.Popen(
                argv, stdout=handle, stderr=subprocess.STDOUT,
                start_new_session=True, cwd=self._here(),
            )
        return proc.pid, log_file

    async def _go2_control_start(self):
        """Start the lease-holding C++ bridge and the chunk executor."""
        messages = []
        bridge_pid = self._go2_bridge_pid()
        if bridge_pid:
            messages.append("bridge already running (pid %s)" % bridge_pid)
        else:
            sock = os.path.join(self._here(), "go2.sock")
            bridge = os.path.join(self._here(), "go2_velocity_bridge")
            if not os.path.isfile(bridge):
                raise RuntimeError("missing %s; deploy/build first" % bridge)
            # Clean env: ROS/CycloneDDS LD_LIBRARY_PATH fights the SDK libddsc.
            env = {
                "HOME": os.environ.get("HOME", ""),
                "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
                "USER": os.environ.get("USER", "unitree"),
            }
            log_file = os.path.join(self._here(), "go2_bridge.log")
            with open(log_file, "a") as handle:
                proc = subprocess.Popen(
                    [bridge, self.args.network_interface, sock],
                    stdout=handle, stderr=subprocess.STDOUT,
                    start_new_session=True, env=env, cwd=self._here(),
                )
            messages.append("bridge started (pid %d, log %s)" % (proc.pid, log_file))

        chunk_pid = self._go2_chunk_pid()
        if chunk_pid:
            messages.append("go2 chunk already running (pid %s)" % chunk_pid)
        elif self._teleop_pid():
            messages.append("teleop running; not starting go2 chunk")
        elif self._teach_pid():
            messages.append("teach capture running; not starting go2 chunk")
        else:
            if self._go2_stream_pid():
                return "go2 streamer running; stop it first (system.go2_stream_stop)"
            python = self.args.python
            argv = [
                python, os.path.join(self._here(), "go2_chunk_loop.py"),
                "--episode-pointer", self._active_episode_path(),
            ]
            pid, log_file = self._spawn(argv, "go2_chunk.log")
            messages.append("go2 chunk started (pid %d, log %s)" % (pid, log_file))
        return "; ".join(messages)

    async def _go2_control_stop(self):
        messages = []
        for label, pid in (
            ("go2 chunk", self._go2_chunk_pid()),
            ("go2 bridge", self._go2_bridge_pid()),
        ):
            if not pid:
                messages.append("no %s running" % label)
                continue
            messages.append("%s %s" % (label, await self._kill_and_wait(pid)))
        return "; ".join(messages)

    async def _go2_control_status(self):
        here = self._here()
        parts = [
            "bridge_pid=%s chunk_pid=%s"
            % (self._go2_bridge_pid() or "none", self._go2_chunk_pid() or "none")
        ]
        for name in ("go2_bridge.log", "go2_chunk.log"):
            path = os.path.join(here, name)
            try:
                with open(path) as handle:
                    tail = "".join(handle.readlines()[-6:]).strip()
            except OSError:
                tail = "(no log yet)"
            parts.append("--- %s ---\n%s" % (name, tail))
        return "\n".join(parts)

    async def _go2_stream_start(self):
        pid = self._go2_stream_pid()
        if pid:
            return "go2 stream already running (pid %s)" % pid
        if not self._go2_bridge_pid():
            return "go2 bridge not running; start it first (system.go2_control_start)"
        chunk_pid = self._go2_chunk_pid()
        notes = []
        if chunk_pid:
            # Stream and chunk both write MOVE to the same bridge; keep one writer.
            subprocess.run(["kill", "-TERM", chunk_pid], timeout=5)
            await asyncio.sleep(0.3)
            notes.append("stopped go2 chunk pid %s" % chunk_pid)
        argv = [
            self.args.python, os.path.join(self._here(), "go2_stream_loop.py"),
            "--host", "127.0.0.1", "--port", "18765",
        ]
        pid, log_file = self._spawn(argv, "go2_stream.log")
        msg = "go2 stream started (pid %d, log %s)" % (pid, log_file)
        if notes:
            msg = "; ".join(notes + [msg])
        return msg

    async def _go2_stream_stop(self):
        pid = self._go2_stream_pid()
        if not pid:
            return "no go2 stream running"
        subprocess.run(["kill", "-TERM", pid], timeout=5)
        return "stop signal sent to go2 stream pid %s" % pid

    async def _go2_stream_status(self):
        pid = self._go2_stream_pid()
        log_file = os.path.join(self._here(), "go2_stream.log")
        try:
            with open(log_file) as handle:
                tail = "".join(handle.readlines()[-8:]).strip()
        except OSError:
            tail = "(no log yet)"
        return "pid=%s\n%s" % (pid or "none", tail)

    async def _go2_pose_command(self, command):
        """Stand/stop via the lease-holding bridge when it is up.

        ``go2_command_executor`` takes its own SportClient lease and conflicts
        with the velocity bridge (3205/3207 / FATAL).  Route through go2.sock
        whenever the bridge is running so A800 ``robotctl go2.stand_down`` is
        safe after ``system.go2_control_start``.
        """
        line = {
            "go2.stand_up": "STAND_UP",
            "go2.stand_down": "STAND_DOWN",
            "go2.stop": "STOP",
        }[command]
        if not self._go2_bridge_pid():
            return await self._run([
                self.args.go2_executor, self.args.network_interface,
                command.split(".", 1)[1],
            ])
        sock_path = os.path.join(self._here(), "go2.sock")
        loop = asyncio.get_running_loop()

        def _send():
            deadline = time.monotonic() + 3.0
            last_error = None
            while time.monotonic() < deadline:
                try:
                    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                    sock.settimeout(2.0)
                    sock.connect(sock_path)
                    sock.sendall((line + "\n").encode())
                    sock.close()
                    return "bridge %s" % line
                except (FileNotFoundError, ConnectionRefusedError, OSError) as exc:
                    last_error = exc
                    time.sleep(0.05)
            raise RuntimeError(
                "bridge socket not ready (%s): %s" % (sock_path, last_error))

        return await loop.run_in_executor(None, _send)

    async def _go2_action_chunk(self, arguments):
        pid = self._go2_chunk_pid()
        if not pid:
            raise RuntimeError(
                "go2 chunk executor not running (system.go2_control_start first)")
        if not self._go2_bridge_pid():
            raise RuntimeError(
                "go2 bridge not running (system.go2_control_start first)")
        sock_path = os.path.join(self._here(), "go2_chunk.sock")
        payload = json.dumps(arguments).encode()
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            try:
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                sock.connect(sock_path)
                sock.sendall(payload)
                sock.close()
                return "go2 chunk delivered (%d rows)" % len(arguments.get("actions", []))
            except (FileNotFoundError, ConnectionRefusedError):
                await asyncio.sleep(0.05)
        raise RuntimeError("go2 chunk executor socket not ready")

    async def _teleop_start(self):
        """Joystick teleop: remote sticks → dog bridge + Piper CAN.

        Exclusive with Piper stream/chunk (single CAN writer) and with
        go2_chunk_loop (that process would log idle zeros into go2_cmd.csv).
        The velocity bridge stays up so the lease is held; turn the remote
        ON only after the bridge has the lease, so sticks are an input
        device and do not steal sport control.
        """
        pid = self._teleop_pid()
        if pid:
            return "teleop already running (pid %s)" % pid
        script = os.path.join(self._here(), "hetero_teleop_loop.py")
        if not os.path.isfile(script):
            raise RuntimeError("missing %s; pull_control_on_go2.sh first" % script)
        notes = []
        stream_pid = self._stream_pid()
        if stream_pid:
            subprocess.run(["kill", "-TERM", stream_pid], timeout=5)
            notes.append("stopped piper stream pid %s" % stream_pid)
        chunk_pid = self._chunk_pid()
        if chunk_pid:
            subprocess.run(["kill", "-TERM", chunk_pid], timeout=5)
            notes.append("stopped piper chunk pid %s" % chunk_pid)
        go2_stream_pid = self._go2_stream_pid()
        if go2_stream_pid:
            subprocess.run(["kill", "-TERM", go2_stream_pid], timeout=5)
            notes.append("stopped go2 stream pid %s" % go2_stream_pid)
        go2_chunk_pid = self._go2_chunk_pid()
        if go2_chunk_pid:
            subprocess.run(["kill", "-TERM", go2_chunk_pid], timeout=5)
            notes.append("stopped go2 chunk pid %s" % go2_chunk_pid)
        teach_pid = self._teach_pid()
        if teach_pid:
            subprocess.run(["kill", "-TERM", teach_pid], timeout=5)
            notes.append("stopped teach capture pid %s" % teach_pid)
        if notes:
            await asyncio.sleep(0.4)
        if not self._go2_bridge_pid():
            notes.append("warning: go2 bridge not running (dog sticks ignored)")
        argv = [
            self.args.python, script,
            "--can", self.args.piper_can,
            "--network-interface", self.args.network_interface,
            "--bridge-sock", os.path.join(self._here(), "go2.sock"),
            "--episode-pointer", self._active_episode_path(),
        ]
        log_file = os.path.join(self._here(), "teleop.log")
        with open(log_file, "w") as handle:
            proc = subprocess.Popen(
                argv, stdout=handle, stderr=subprocess.STDOUT,
                start_new_session=True, cwd=self._here(),
            )
        await asyncio.sleep(0.7)
        if proc.poll() is not None:
            tail = ""
            try:
                with open(log_file) as handle:
                    tail = handle.read()[-2000:].strip()
            except OSError:
                tail = "(no teleop.log)"
            raise RuntimeError(
                "teleop exited immediately (code %s):\n%s"
                % (proc.returncode, tail)
            )
        msg = "teleop started (pid %d, log %s)" % (proc.pid, log_file)
        if notes:
            msg = "; ".join(notes + [msg])
        return msg

    async def _teleop_stop(self):
        pid = self._teleop_pid()
        if not pid:
            return "no teleop running"
        subprocess.run(["kill", "-TERM", pid], timeout=5)
        return "stop signal sent to teleop pid %s (arm holds last pose, dog STOP)" % pid

    async def _teleop_status(self):
        pid = self._teleop_pid()
        log_file = os.path.join(self._here(), "teleop.log")
        tail = ""
        try:
            with open(log_file) as handle:
                tail = "".join(handle.readlines()[-10:]).strip()
        except OSError:
            tail = "(no log yet)"
        return "pid=%s\n%s" % (pid or "none", tail)

    async def _teach_start(self):
        """Drag-teach capture: left stick → dog bridge; Piper drag-teach.

        Stops Piper/Go2 chunk writers (they would ModeCtrl(0x01) and fight
        drag-teach) but keeps the velocity bridge so the dog stays on the
        same clock as the arm CSV.  The capture process enables the arm and
        sends MotionCtrl_1 drag-teach once so standby brakes open.
        """
        pid = self._teach_pid()
        if pid:
            return "teach capture already running (pid %s)" % pid
        script = os.path.join(self._here(), "hetero_teach_loop.py")
        if not os.path.isfile(script):
            raise RuntimeError("missing %s; pull_control_on_go2.sh first" % script)
        notes = []
        stream_pid = self._stream_pid()
        if stream_pid:
            subprocess.run(["kill", "-TERM", stream_pid], timeout=5)
            notes.append("stopped piper stream pid %s" % stream_pid)
        chunk_pid = self._chunk_pid()
        if chunk_pid:
            subprocess.run(["kill", "-TERM", chunk_pid], timeout=5)
            notes.append("stopped piper chunk pid %s" % chunk_pid)
        teleop_pid = self._teleop_pid()
        if teleop_pid:
            subprocess.run(["kill", "-TERM", teleop_pid], timeout=5)
            notes.append("stopped teleop pid %s" % teleop_pid)
        go2_stream_pid = self._go2_stream_pid()
        if go2_stream_pid:
            subprocess.run(["kill", "-TERM", go2_stream_pid], timeout=5)
            notes.append("stopped go2 stream pid %s" % go2_stream_pid)
        go2_chunk_pid = self._go2_chunk_pid()
        if go2_chunk_pid:
            subprocess.run(["kill", "-TERM", go2_chunk_pid], timeout=5)
            notes.append("stopped go2 chunk pid %s" % go2_chunk_pid)
        if notes:
            await asyncio.sleep(0.4)
        if not self._go2_bridge_pid():
            notes.append("warning: go2 bridge not running (dog sticks ignored)")
        argv = [
            self.args.python, script,
            "--can", self.args.piper_can,
            "--network-interface", self.args.network_interface,
            "--bridge-sock", os.path.join(self._here(), "go2.sock"),
            "--episode-pointer", self._active_episode_path(),
        ]
        log_file = os.path.join(self._here(), "teach.log")
        with open(log_file, "w") as handle:
            proc = subprocess.Popen(
                argv, stdout=handle, stderr=subprocess.STDOUT,
                start_new_session=True, cwd=self._here(),
            )
        await asyncio.sleep(0.7)
        if proc.poll() is not None:
            tail = ""
            try:
                with open(log_file) as handle:
                    tail = handle.read()[-2000:].strip()
            except OSError:
                tail = "(no teach.log)"
            raise RuntimeError(
                "teach capture exited immediately (code %s):\n%s"
                % (proc.returncode, tail)
            )
        msg = "teach capture started (pid %d, log %s)" % (proc.pid, log_file)
        if notes:
            msg = "; ".join(notes + [msg])
        return msg

    async def _teach_stop(self):
        pid = self._teach_pid()
        if not pid:
            return "no teach capture running"
        subprocess.run(["kill", "-TERM", pid], timeout=5)
        return (
            "stop signal sent to teach pid %s "
            "(exits software drag-teach; press the arm button if it is still "
            "in hardware teach before replay, dog STOP)"
            % pid
        )

    async def _teach_status(self):
        pid = self._teach_pid()
        log_file = os.path.join(self._here(), "teach.log")
        tail = ""
        try:
            with open(log_file) as handle:
                tail = "".join(handle.readlines()[-25:]).strip()
        except OSError:
            tail = "(no log yet)"
        return "pid=%s\n%s" % (pid or "none", tail)

    def _active_episode_path(self):
        return os.path.join(self.args.recorder_home, "active_episode")

    def _capture_ctl(self):
        return os.path.join(self.args.recorder_home, "go2_capture_ctl.sh")

    def _read_active_episode(self):
        path = self._active_episode_path()
        try:
            with open(path) as handle:
                value = handle.read().strip()
        except OSError:
            return None
        return value or None

    async def _record_start(self, arguments):
        episode_id = arguments.get("episode_id")
        instruction = arguments.get("instruction") or ""
        capture = self._capture_ctl()
        if os.path.isfile(capture):
            argv = ["bash", capture, "start", episode_id]
            if instruction:
                argv.append(instruction)
            return await self._run(argv)
        if self._read_active_episode():
            raise RuntimeError(
                "recording already active: %s" % self._read_active_episode())
        episode_dir = os.path.join(self.args.data_root, episode_id)
        if os.path.exists(episode_dir):
            raise RuntimeError("refusing to overwrite existing episode: %s" % episode_dir)
        os.makedirs(episode_dir)
        with open(os.path.join(episode_dir, "instruction.txt"), "w") as handle:
            handle.write(instruction + "\n")
        with open(os.path.join(episode_dir, "start_time.txt"), "w") as handle:
            handle.write(time.strftime("%Y-%m-%dT%H:%M:%S%z") + "\n")
        os.makedirs(self.args.recorder_home, exist_ok=True)
        with open(self._active_episode_path(), "w") as handle:
            handle.write(episode_dir + "\n")
        return (
            "recording started without DDS recorder (missing %s): %s"
            % (capture, episode_dir)
        )

    async def _record_stop(self):
        capture = self._capture_ctl()
        episode_dir = self._read_active_episode()
        if os.path.isfile(capture):
            try:
                output = await self._run(["bash", capture, "stop"])
            except RuntimeError as exc:
                if "not running" in str(exc).lower():
                    try:
                        os.remove(self._active_episode_path())
                    except OSError:
                        pass
                    if episode_dir is None:
                        return "no recording running"
                    output = (
                        "recorder was not running; cleared leftover episode "
                        "pointer: %s" % episode_dir
                    )
                else:
                    raise
        elif episode_dir is None:
            return "no recording running"
        else:
            with open(os.path.join(episode_dir, "stop_time.txt"), "w") as handle:
                handle.write(time.strftime("%Y-%m-%dT%H:%M:%S%z") + "\n")
            try:
                os.remove(self._active_episode_path())
            except OSError:
                pass
            output = "stopped recording: %s" % episode_dir
        # Chunk loops notice the pointer is gone on the next 5–50 ms tick
        # and write command-based replay JSON.  Then rebuild the dog replay
        # from sport pose deltas so stick-release zeros are not used.
        await asyncio.sleep(0.4)
        if episode_dir and os.path.isdir(episode_dir):
            try:
                sys.path.insert(0, self._here())
                from episode_log import write_go2_replay  # noqa: E402
                measured = write_go2_replay(episode_dir)
                if measured:
                    output = "%s; go2_replay from sport_mode_state" % output
            except Exception as exc:
                LOG.error("failed to rebuild go2_replay from sport state: %s", exc)
            names = []
            for name in (
                "go2_cmd.csv", "go2_replay.json",
                "piper_cmd.csv", "piper_state.csv", "piper_replay.json",
                "sport_mode_state.csv", "wireless_controller.csv", "stick.json",
            ):
                path = os.path.join(episode_dir, name)
                if os.path.isfile(path):
                    names.append(name)
            if names:
                output = "%s; files: %s" % (output, " ".join(names))
        return output

    async def _record_status(self):
        episode_dir = self._read_active_episode()
        capture = self._capture_ctl()
        dds = "dds=no"
        if os.path.isfile(capture):
            try:
                dds_status = await self._run(["bash", capture, "status"])
                dds = "dds=%s" % dds_status.splitlines()[0]
            except Exception as exc:
                dds = "dds=error: %s" % exc
        if episode_dir is None:
            return "stopped; %s" % dds
        files = []
        for name in (
            "go2_cmd.csv", "go2_replay.json",
            "piper_cmd.csv", "piper_state.csv", "piper_replay.json",
            "sport_mode_state.csv", "wireless_controller.csv", "low_state.csv",
            "stick.json",
        ):
            path = os.path.join(episode_dir, name)
            if os.path.isfile(path):
                files.append("%s=%dB" % (name, os.path.getsize(path)))
        listing = " ".join(files) if files else "(no control csv yet; is chunk running?)"
        return "recording %s; %s; %s" % (episode_dir, dds, listing)


def parse_args():
    here = os.path.dirname(os.path.realpath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", default=os.environ.get(
        "HETEROVLA_CONTROL_URI", "ws://127.0.0.1:8766"))
    parser.add_argument("--robot-id", default="unitree-go2-01")
    parser.add_argument("--token", default=os.environ.get("HETEROVLA_CONTROL_TOKEN", ""))
    parser.add_argument("--execute", action="store_true",
                        help="actually command hardware; omitted means dry-run")
    parser.add_argument("--network-interface", default="eth0")
    parser.add_argument("--piper-can", default="can0")
    parser.add_argument("--command-timeout", type=float, default=12.0)
    parser.add_argument("--go2-executor", default=os.path.join(here, "go2_command_executor"))
    parser.add_argument("--piper-executor", default=os.path.join(here, "piper_command_executor.py"))
    parser.add_argument("--python", default=os.environ.get(
        "PIPER_PYTHON",
        os.environ.get("HETEROVLA_PYTHON", "/usr/bin/python3")))
    parser.add_argument(
        "--recorder-home",
        default=os.environ.get(
            "GO2_RECORDER_HOME",
            os.path.expanduser("~/heterovla-recorder")),
    )
    parser.add_argument(
        "--data-root",
        default=os.environ.get(
            "GO2_DATA_ROOT",
            os.path.expanduser("~/heterovla-data/raw")),
    )
    return parser.parse_args()

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(RobotAgent(parse_args()).run_forever())


if __name__ == "__main__":
    main()
