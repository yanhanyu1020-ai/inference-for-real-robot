"""ROS1 P450 observations and ENU velocity setpoints; no arming/mode/takeoff calls."""
import fcntl
import math
import threading
import time

import numpy as np

from .core import Sensors, finite, world_velocity


def quaternion_rpy(q):
    x, y, z, w = finite(q, (4,))
    norm = math.sqrt(x*x+y*y+z*z+w*w)
    if norm < .5 or norm > 1.5:
        raise ValueError('Invalid P450 orientation quaternion')
    x, y, z, w = x/norm, y/norm, z/norm, w/norm
    return [math.atan2(2*(w*x+y*z), 1-2*(x*x+y*y)),
            math.asin(max(-1., min(1., 2*(w*y-z*x)))),
            math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))]


class P450Hardware:
    def __init__(self, config, execute=False):
        import rospy
        import cv2
        from geometry_msgs.msg import PoseStamped, TwistStamped
        from sensor_msgs.msg import CompressedImage
        from mavros_msgs.msg import State, ExtendedState
        from prometheus_msgs.msg import UAVState
        self.rospy, self.cv2, self.TwistStamped = rospy, cv2, TwistStamped
        self.config, self.execute = config, execute
        self.sensors = Sensors('p450')
        self.closed = threading.Event()
        self.lock = threading.RLock()
        self.command = np.zeros(4)
        self.deadline = 0.
        self.fault = None
        self.mode = None
        self.mode_time = 0.
        self.topology_time = 0.
        self.publisher = None
        self.subscribers = []
        self.callback_errors = {}
        self.thread = self.monitor = None
        self.lock_file = open('/tmp/heterovla-v3-p450.lock', 'a')
        fcntl.flock(self.lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rospy.init_node('heterovla_v3_gateway', anonymous=False, disable_signals=True)
        if rospy.get_param('/use_sim_time', False):
            raise RuntimeError('Real robot gateway requires wall-clock ROS time')
        self.ns = config.get('namespace', '/uav1').rstrip('/')
        self.velocity_topic = self.ns+'/mavros/setpoint_velocity/cmd_vel'
        try:
            for topic, cls, callback in [
                (self.ns+'/mavros/local_position/pose', PoseStamped, self._pose),
                (self.ns+'/prometheus/state', UAVState, self._flags),
                (self.ns+'/mavros/extended_state', ExtendedState, self._landed),
                (self.ns+'/mavros/state', State, self._mode),
                (config.get('image_topic', self.ns+'/camera/color/image_raw/compressed'), CompressedImage, self._image),
            ]:
                self.subscribers.append(rospy.Subscriber(topic, cls, self._guard(callback), queue_size=1, buff_size=4*1024*1024))
            if execute:
                self._check_topology()
                self.publisher = rospy.Publisher(self.velocity_topic, TwistStamped, queue_size=1)
                self.thread = threading.Thread(target=self._publish, daemon=True)
                self.monitor = threading.Thread(target=self._monitor, daemon=True)
                self.thread.start()
                self.monitor.start()
        except Exception:
            self.close()
            raise

    def _guard(self, callback):
        def receive(message):
            try:
                callback(message)
                self.callback_errors.pop(callback.__name__, None)
            except Exception as exc:
                # Drop stale startup messages. During execution a bad stream stops output;
                # otherwise the missing fresh sample remains visible in preview health.
                self.callback_errors[callback.__name__] = str(exc)
                if time.monotonic() < self.deadline:
                    self.sensors.error = str(exc)
                    self.halt()
        return receive

    def _stamp(self, message, max_age):
        age = (self.rospy.Time.now()-message.header.stamp).to_sec()
        if not math.isfinite(age) or not -.02 <= age <= max_age or message.header.stamp.to_sec() <= 0:
            raise RuntimeError('ROS source timestamp is missing/stale; check ROS clock and topic latency')
        return time.monotonic()-max(0., age)

    def _pose(self, message):
        stamp = self._stamp(message, .075)
        p, q = message.pose.position, message.pose.orientation
        self.sensors.put('pose', stamp, np.array([p.x, p.y, p.z, *quaternion_rpy([q.x, q.y, q.z, q.w])]))

    def _flags(self, message):
        self.sensors.put('flags', self._stamp(message, .15),
                         np.array([message.connected, message.armed, message.odom_valid], dtype=float))

    def _landed(self, message):
        self.sensors.put('landed', self._stamp(message, .25), int(message.landed_state))

    def _mode(self, message):
        with self.lock:
            self.mode_time = self._stamp(message, 1.5)
            self.mode = (bool(message.connected), bool(message.armed), str(message.mode))

    def _image(self, message):
        stamp = self._stamp(message, .15)
        image = self.cv2.imdecode(np.frombuffer(message.data, dtype=np.uint8), self.cv2.IMREAD_COLOR)
        if image is None or image.shape != (480, 640, 3):
            raise ValueError('P450 camera must publish 640x480 BGR/RGB compressed images')
        self.sensors.put('uav', stamp, image)

    def _check_topology(self):
        frame = self.rospy.get_param(self.ns+'/mavros/setpoint_velocity/mav_frame', 'LOCAL_NED')
        if frame != 'LOCAL_NED':
            raise RuntimeError('Expected MAVROS velocity mav_frame=LOCAL_NED; refusing an ambiguous frame')
        code, _, state = self.rospy.get_master().getSystemState()
        if code != 1:
            raise RuntimeError('Cannot verify ROS command ownership')
        topics = {self.velocity_topic, self.ns+'/mavros/setpoint_velocity/cmd_vel_unstamped',
                  self.ns+'/mavros/setpoint_raw/local', self.ns+'/mavros/setpoint_position/local',
                  self.ns+'/mavros/setpoint_raw/attitude', self.ns+'/mavros/setpoint_attitude/attitude'}
        for topic, nodes in state[0]:
            others = [node for node in nodes if node != self.rospy.get_name()]
            if topic in topics and others:
                raise RuntimeError('Another flight controller owns %s: %s' % (topic, others))
        self.topology_time = time.monotonic()

    def _monitor(self):
        while not self.closed.wait(.5):
            try:
                self._check_topology()
            except Exception as exc:
                self.fault = str(exc)
                self.halt()
                return

    def check_execution(self):
        if not self.execute or self.fault or self.sensors.error:
            raise RuntimeError(self.fault or self.sensors.error or 'P450 is observe-only')
        if time.monotonic()-self.topology_time > 2:
            raise RuntimeError('ROS command-ownership check expired')
        with self.lock:
            if self.mode != (True, True, 'OFFBOARD') or time.monotonic()-self.mode_time > 1.5:
                raise RuntimeError('P450 must already be connected, armed and in OFFBOARD mode')
        with self.sensors.lock:
            _, pose = self.sensors.streams['pose'].past(time.monotonic(), .075)
            _, flags = self.sensors.streams['flags'].past(time.monotonic(), .15)
            _, landed = self.sensors.streams['landed'].past(time.monotonic(), .25)
            origin = self.sensors.origin
            if not np.all(flags == 1) or landed != 2:
                raise RuntimeError('P450 must have valid odometry and be in flight')
            lo, hi = self.config.get('altitude_bounds_m', [.3, 2.0])
            if not lo <= pose[2] <= hi or origin is None:
                raise RuntimeError('P450 outside altitude envelope or episode origin missing')
            if np.linalg.norm(pose[:2]-origin[:2]) > self.config.get('radius_m', 1.0):
                raise RuntimeError('P450 outside episode horizontal envelope')

    def send(self, row):
        self.check_execution()
        with self.lock:
            if self.fault:
                raise RuntimeError(self.fault)
            self.command = finite(row, (4,)).copy()
            self.deadline = time.monotonic()+.1

    def _publish(self):
        # Keep the OFFBOARD stream alive with zero velocity between bounded chunks.
        while not self.closed.wait(.02) and not self.rospy.is_shutdown():
            command = np.zeros(4)
            try:
                with self.lock:
                    if not self.fault and time.monotonic() < self.deadline:
                        self.check_execution()
                        command = self.command.copy()
                if np.any(command):
                    with self.sensors.lock:
                        _, pose = self.sensors.streams['pose'].past(time.monotonic(), .075)
                    command[:2] = world_velocity(command[:2], pose[5])
                message = self.TwistStamped()
                message.header.stamp = self.rospy.Time.now()
                message.header.frame_id = 'map'
                message.twist.linear.x, message.twist.linear.y, message.twist.linear.z = map(float, command[:3])
                message.twist.angular.z = float(command[3])
                self.publisher.publish(message)
            except Exception as exc:
                self.fault = str(exc)
                self.deadline = 0.
                zero = self.TwistStamped()
                zero.header.stamp = self.rospy.Time.now()
                self.publisher.publish(zero)

    def hold(self):
        with self.lock:
            self.command[:] = 0
            self.deadline = 0.

    def halt(self):
        with self.lock:
            self.fault = self.fault or 'P450 stop latched; restart gateway after inspection'
            self.command[:] = 0
            self.deadline = 0.

    def close(self):
        self.halt()
        if self.publisher is not None:
            for _ in range(5):
                zero = self.TwistStamped()
                zero.header.stamp = self.rospy.Time.now()
                self.publisher.publish(zero)
                time.sleep(.02)
        self.closed.set()
        for thread in (self.thread, self.monitor):
            if thread is not None:
                thread.join(timeout=1)
        for sub in self.subscribers:
            sub.unregister()
        if self.publisher is not None:
            self.publisher.unregister()
        self.lock_file.close()
