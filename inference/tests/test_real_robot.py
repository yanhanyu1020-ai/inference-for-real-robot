"""CPU contracts for live V3 framing, units, action expiry and bounded execution."""
import base64
import json
import os
from pathlib import Path
import tempfile
import time
import threading
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from inference.real_robot.core import (History, Sensors, DT, action_plan, decode_observation,
                                       heading, piper_sdk_target, world_velocity)
from inference.real_robot.gateway import Gateway
from inference.real_robot.gateway import handler
from inference.real_robot.controller import Device, run_cycle
from inference.real_robot.p450 import quaternion_rpy
from inference.real_robot.unitree import UnitreeHardware, go2_normal_state_codes
from inference.real_robot.diagnostic import observation as diagnostic_observation
from inference.real_robot.diagnostic import validate_action_downlink
from http.server import ThreadingHTTPServer

ARM = np.array([0., .4, -.4, 0., 0., 0., .02])


def packet(kind, stamp=None):
    stamp = time.monotonic()-.002 if stamp is None else stamp
    bgr = np.full((480, 640, 3), (10, 20, 30), dtype=np.uint8)
    ok, encoded = cv2.imencode('.jpg', bgr)
    assert ok
    data = base64.b64encode(encoded).decode()
    names = ['front', 'wrist'] if kind == 'unitree' else ['uav']
    state = [0, 0, .3, 0, 0, 0, 1, 0, 0, 0, *ARM] if kind == 'unitree' else [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 2]
    return dict(kind=kind, stamp=stamp, state=state, images={n:data for n in names},
                image_stamps={n:stamp for n in names}, state_stamps=[stamp])


class FakeSensors:
    error = None
    def __init__(self, kind): self.kind = kind
    def snapshot(self, diagnostic=False):
        result=packet(self.kind)
        result.update(diagnostic_only=diagnostic,readiness_issues=[])
        return result
    def reset(self): pass


class FakeHardware:
    def __init__(self, kind):
        self.sensors = FakeSensors(kind)
        self.sent, self.holds, self.stops = [], 0, 0
        self.block = 0
    def check_execution(self):
        if self.stops: raise RuntimeError('Stopped')
    def send(self, row):
        self.sent.append((time.monotonic(), row.copy()))
        if self.block: time.sleep(self.block)
    def hold(self): self.holds += 1
    def halt(self): self.stops += 1
    def close(self): pass


class DirectDevice(Device):
    def __init__(self, gateway):
        super().__init__('http://unused', 'test-token')
        self.gateway = gateway
    def call(self, endpoint, body=None):
        return {'/status': self.gateway.status, '/observation': self.gateway.observe,
                '/diagnostic': self.gateway.diagnose,
                '/validate_actions': lambda: self.gateway.validate_actions(body),
                '/reset': self.gateway.reset, '/stop': self.gateway.stop,
                '/prepare': lambda: self.gateway.prepare(body),
                '/commit': lambda: self.gateway.commit(body)}[endpoint]()


class TestLiveSchema(unittest.TestCase):
    def test_units_and_heading(self):
        np.testing.assert_allclose(heading([0, 1], np.pi/2), [1, 0], atol=1e-10)
        np.testing.assert_allclose(world_velocity([1, 0], np.pi/2), [0, 1], atol=1e-10)
        joints, grip = piper_sdk_target([np.pi/2, 0, 0, 0, 0, 0, .02])
        self.assertEqual(joints[0], 90000)
        self.assertEqual(grip, 20000)
        np.testing.assert_allclose(quaternion_rpy([0, 0, 2**-.5, 2**-.5]), [0, 0, np.pi/2], atol=1e-10)

    def test_causal_velocity_uses_pose_difference_and_current_heading(self):
        history = History()
        for t in np.linspace(0, .2, 21):
            history.add(float(t), np.array([0., 2*t, 0., np.pi/2]))
        np.testing.assert_allclose(history.motion(.15, 3), [2, 0, 0], atol=1e-10)
        history.add(.3, np.array([999., 999., 0., 0.]))
        np.testing.assert_allclose(history.motion(.15, 3), [2, 0, 0], atol=1e-10)
        with self.assertRaises(RuntimeError): history.past(.5, .075)

    def test_state30_image_slots_and_rgb(self):
        now = time.monotonic()
        result = decode_observation(packet('unitree', now-.01), packet('p450', now-.01),
                                    {'unitree':(0,.001), 'p450':(0,.001)}, 'task', now=now)
        self.assertEqual(result['state'].shape, (30,))
        self.assertEqual(len(result['image']), 3)
        np.testing.assert_allclose(result['image']['base_0_rgb'][0,0], [30,20,10], atol=2)
        np.testing.assert_array_equal(result['state'][10:17], ARM.astype(np.float32))

    def test_bad_clock_skew_and_stale_frames(self):
        now = time.monotonic()
        for clock in [(0, .05), (float('nan'), .001)]:
            with self.assertRaises(RuntimeError):
                decode_observation(packet('unitree',now), packet('p450',now),
                                   {'unitree':clock,'p450':(0,.001)}, 'task', now)
        with self.assertRaises(RuntimeError):
            decode_observation(packet('unitree',now-.01), packet('p450',now-.15),
                               {'unitree':(0,.001),'p450':(0,.001)}, 'task', now)
        with self.assertRaises(RuntimeError):
            decode_observation(packet('unitree',now-1), packet('p450',now-1),
                               {'unitree':(0,.001),'p450':(0,.001)}, 'task', now)

    def test_action_shape_finite_and_physical_bounds(self):
        row = np.array([[.1, 0, 0, *ARM]])
        result = action_plan(row, 'unitree', ARM)
        self.assertAlmostEqual(result[0,0], .03)
        row[0,3] = .3
        result = action_plan(row, 'unitree', ARM)
        self.assertAlmostEqual(result[0,3], .35*DT)
        for bad in (np.full((1,10), np.nan), np.full((1,10), 1e10), np.zeros((1,14))):
            with self.assertRaises(ValueError): action_plan(bad, 'unitree', ARM)
        with self.assertRaises(ValueError): action_plan([[0,0,10,0]], 'p450')

    def test_live_state_builder_matches_training_frame_and_does_not_fill(self):
        sensor = Sensors('unitree')
        now = time.monotonic()
        sensor.origin = np.array([0,0,0,np.pi/2,.3,0,0])
        for t in np.linspace(now-.3, now, 31):
            sensor.put('go2',float(t),np.array([0,2*(t-(now-.3)),0,np.pi/2,.3,0,0]))
            sensor.put('piper',float(t),ARM)
        for name in ('front','wrist'):sensor.put(name,now,np.zeros((480,640,3),np.uint8))
        result=sensor.snapshot()
        self.assertAlmostEqual(result['state'][0],.6,places=5)
        np.testing.assert_allclose(result['state'][7:10],[2,0,0],atol=1e-8)
        with patch('inference.real_robot.core.time.monotonic',return_value=now+1):
            with self.assertRaises(RuntimeError):sensor.snapshot()


class TestGateway(unittest.TestCase):
    def setUp(self):
        self.hardware=FakeHardware('unitree')
        self.gateway=Gateway(self.hardware,execute=True)
        self.gateway.reset()
        self.addCleanup(self.gateway.close)

    def prepare(self):
        observation=self.gateway.observe()
        return self.gateway.prepare(dict(session=observation['session'],observation_id=observation['observation_id'],
                                         actions=[[0,0,0,*ARM]]*3))

    def test_prepare_has_no_motion_and_commit_is_single_use(self):
        plan=self.prepare()
        self.assertEqual(self.hardware.sent,[])
        start=time.monotonic()+.04
        self.gateway.commit(dict(plan_id=plan['plan_id'],start_at=start))
        with self.assertRaises(RuntimeError):self.gateway.commit(dict(plan_id=plan['plan_id'],start_at=start))
        self.gateway.worker.join(timeout=1)
        self.assertEqual(self.gateway.plan['status'],'done')
        self.assertEqual(self.gateway.plan['executed_steps'],3)
        self.assertGreaterEqual(self.hardware.sent[0][0],start)
        self.assertEqual(self.hardware.holds,1)

    def test_stale_unknown_and_wrong_episode_rejected(self):
        observation=self.gateway.observe()
        body=dict(session=observation['session'],observation_id=observation['observation_id'],actions=[[0,0,0,*ARM]])
        with self.assertRaises(ValueError):self.gateway.prepare(dict(body,session='wrong'))
        self.gateway.snapshots[observation['observation_id']]['stamp']-=2
        with self.assertRaises(RuntimeError):self.gateway.prepare(body)
        with self.assertRaises(ValueError):self.gateway.prepare(dict(body,observation_id='unknown'))
        self.assertEqual(self.hardware.sent,[])

    def test_scheduled_stop_prevents_execution(self):
        plan=self.prepare()
        self.gateway.commit(dict(plan_id=plan['plan_id'],start_at=time.monotonic()+.15))
        self.gateway.stop('test abort')
        self.gateway.worker.join(timeout=1)
        self.assertEqual(self.hardware.sent,[])
        self.assertEqual(self.gateway.plan['status'],'fault')

    def test_watchdog_stops_blocked_control_loop(self):
        self.hardware.block=.35
        plan=self.prepare()
        self.gateway.commit(dict(plan_id=plan['plan_id'],start_at=time.monotonic()+.03))
        self.gateway.worker.join(timeout=1)
        self.assertIsNotNone(self.gateway.fault)
        self.assertGreater(self.hardware.stops,0)
        self.assertEqual(len(self.hardware.sent),1)


class TestCoordinator(unittest.TestCase):
    def setUp(self):
        self.devices={}
        for kind in ('unitree','p450'):
            gateway=Gateway(FakeHardware(kind),execute=True)
            gateway.reset()
            self.addCleanup(gateway.close)
            self.devices[kind]=DirectDevice(gateway)
            self.devices[kind].calibrate()
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.output=Path(temporary.name)
        self.config={'execute_steps':3,'prompt':'task'}

    def policy(self, bad=False):
        class Policy:
            def infer(self, observation):
                assert observation['state'].shape==(30,)
                actions=np.zeros((15,14));actions[:,3:10]=ARM
                if bad:actions[0,0]=np.nan
                return actions
        return Policy()

    def test_full_cycle_returns_post_action_images(self):
        result=run_cycle(self.devices,self.policy(),self.config,self.output,execute=True)
        self.assertTrue(result['executed'])
        self.assertEqual(result['executed_steps'],3)
        self.assertEqual(set(p.name for p in (self.output/'latest').glob('*.jpg')),{'front.jpg','wrist.jpg','uav.jpg'})
        for device in self.devices.values():self.assertGreater(len(device.gateway.hardware.sent),0)

    def test_invalid_policy_output_stops_both_devices(self):
        with self.assertRaises(ValueError):run_cycle(self.devices,self.policy(True),self.config,self.output,execute=True)
        for device in self.devices.values():
            self.assertGreater(device.gateway.hardware.stops,0)
            self.assertEqual(device.gateway.hardware.sent,[])

    def test_failed_prepare_does_not_execute_peer(self):
        original = self.devices['p450'].call
        def fail(endpoint, body=None):
            if endpoint == '/prepare': raise RuntimeError('simulated device failure')
            return original(endpoint,body)
        self.devices['p450'].call=fail
        with self.assertRaises(RuntimeError):run_cycle(self.devices,self.policy(),self.config,self.output,execute=True)
        for device in self.devices.values():
            self.assertEqual(device.gateway.hardware.sent,[])
            self.assertGreater(device.gateway.hardware.stops,0)

    def test_failed_commit_cancels_scheduled_peer(self):
        original = self.devices['p450'].call
        def fail(endpoint, body=None):
            if endpoint == '/commit': raise RuntimeError('simulated device failure')
            return original(endpoint,body)
        self.devices['p450'].call=fail
        with self.assertRaises(RuntimeError):run_cycle(self.devices,self.policy(),self.config,self.output,execute=True)
        worker=self.devices['unitree'].gateway.worker
        if worker is not None:worker.join(timeout=1)
        for device in self.devices.values():self.assertEqual(device.gateway.hardware.sent,[])


class TestUnitreeOnlyCoordinator(unittest.TestCase):
    policy = TestCoordinator.policy

    def setUp(self):
        TestCoordinator.setUp(self)
        observer = self.devices['p450']
        observer.gateway.execute = False
        self.observer_calls = []
        original = observer.call
        def call(endpoint, body=None):
            self.observer_calls.append(endpoint)
            if endpoint not in ('/status', '/diagnostic'):
                self.fail('Control request sent to observation-only P450: '+endpoint)
            return original(endpoint, body)
        observer.call = call
        def snapshot(diagnostic=False):
            self.assertTrue(diagnostic)
            result = packet('p450')
            result['state'][11] = 0
            result.update(diagnostic_only=True, readiness_issues=['P450 is disarmed'])
            return result
        observer.gateway.hardware.sensors.snapshot = snapshot
        for method in ('send', 'hold', 'halt', 'check_execution'):
            setattr(observer.gateway.hardware, method,
                    lambda *args: self.fail('P450 actuator method called'))

    def test_unitree_executes_with_real_disarmed_p450_state_and_three_images(self):
        class Policy:
            def infer(inner, observation):
                self.assertEqual(observation['state'][28], 0)
                self.assertEqual(len(observation['image']), 3)
                return self.policy().infer(observation)
        result = run_cycle(self.devices, Policy(), self.config, self.output,
                           execute=True, targets=('unitree',))
        self.assertTrue(result['executed'])
        self.assertEqual(result['execution_targets'], ['unitree'])
        self.assertEqual(result['observation_only'], ['p450'])
        self.assertEqual(set(result['execution']), {'unitree'})
        self.assertEqual(result['observer_readiness_issues']['p450'], ['P450 is disarmed'])
        self.assertTrue(self.devices['unitree'].gateway.hardware.sent)
        self.assertFalse(self.devices['p450'].gateway.hardware.sent)
        self.assertIsNone(self.devices['p450'].gateway.plan)
        self.assertEqual(set(p.name for p in (self.output/'latest').glob('*.jpg')),
                         {'front.jpg', 'wrist.jpg', 'uav.jpg'})

    def test_bad_prediction_stops_unitree_without_sending_p450_stop(self):
        with self.assertRaises(ValueError):
            run_cycle(self.devices, self.policy(True), self.config, self.output,
                      execute=True, targets=('unitree',))
        self.assertGreater(self.devices['unitree'].gateway.hardware.stops, 0)
        self.assertFalse(self.devices['unitree'].gateway.hardware.sent)
        self.assertNotIn('/stop', self.observer_calls)

    def test_p450_execution_enabled_is_rejected_before_model_or_motion(self):
        self.devices['p450'].gateway.execute = True
        try:
            with self.assertRaisesRegex(RuntimeError, 'Observation-only gateway'):
                run_cycle(self.devices, self.policy(), self.config, self.output,
                          execute=True, targets=('unitree',))
            self.assertFalse(self.devices['unitree'].gateway.hardware.sent)
        finally:
            self.devices['p450'].gateway.execute = False

    def test_unitree_prepare_failure_does_not_control_p450(self):
        original = self.devices['unitree'].call
        def fail(endpoint, body=None):
            if endpoint == '/prepare':
                raise RuntimeError('Unitree is not ready')
            return original(endpoint, body)
        self.devices['unitree'].call = fail
        with self.assertRaisesRegex(RuntimeError, 'Unitree is not ready'):
            run_cycle(self.devices, self.policy(), self.config, self.output,
                      execute=True, targets=('unitree',))
        self.assertFalse(self.devices['unitree'].gateway.hardware.sent)
        self.assertNotIn('/stop', self.observer_calls)


class TestIndependentPreview(unittest.TestCase):
    def test_ground_diagnostic_keeps_real_armed_flag_and_is_not_executable(self):
        sensor=Sensors('p450')
        now=time.monotonic()
        sensor.origin=np.zeros(6)
        for stamp in np.linspace(now-.3,now,31):
            sensor.put('pose',float(stamp),np.zeros(6))
            sensor.put('flags',float(stamp),np.array([1.,0.,1.]))
            sensor.put('landed',float(stamp),1)
        sensor.put('uav',now,np.zeros((480,640,3),np.uint8))
        with self.assertRaises(RuntimeError):sensor.snapshot()
        data=sensor.snapshot(diagnostic=True)
        self.assertEqual(data['state'][11],0)
        self.assertTrue(data['diagnostic_only'])
        self.assertTrue(data['readiness_issues'])
        other=packet('unitree');other['diagnostic_only']=True
        obs=diagnostic_observation({'unitree':other,'p450':data},'task')
        self.assertEqual(obs['state'].shape,(30,))
        self.assertEqual(obs['state'][28],0)
        hardware=FakeHardware('p450');hardware.sensors=sensor
        gateway=Gateway(hardware,execute=False)
        self.addCleanup(gateway.close)
        result=gateway.diagnose()
        self.assertNotIn('observation_id',result)
        self.assertEqual(len(gateway.snapshots),0)
        gateway.execute=True
        with self.assertRaises(RuntimeError):gateway.diagnose()

    def test_go2_fault_does_not_hide_arm_or_camera_feedback(self):
        hardware = UnitreeHardware.__new__(UnitreeHardware)
        hardware.go2_normal_codes = go2_normal_state_codes({'go2_normal_state_codes':[0,100]})
        hardware.sensors = Sensors('unitree')
        hardware.processes = []
        stamp = time.monotonic()
        class Camera:
            error = None
            def snapshot(self):
                return dict(monotonic_ns=int(stamp*1e9), image=np.zeros((480,640,3),np.uint8))
        hardware.cameras = {name:Camera() for name in ('front','wrist')}
        hardware.arm_feedback = lambda:(ARM,stamp)
        with tempfile.TemporaryDirectory() as directory:
            hardware.run = Path(directory)
            (hardware.run/'go2_snapshot.json').write_text(json.dumps({'sport_mode_state':{
                'valid':True,'error_code':1001,'position':[0,0,.05], 'rpy':[0,0,0],
                'body_height':.07,'monotonic_ns':int(stamp*1e9)}}))
            hardware._read_once()
            preview = hardware.sensors.preview()
            self.assertEqual(set(preview['images']), {'front','wrist'})
            self.assertEqual(preview['raw_sensors']['piper']['value'], ARM.tolist())
            self.assertIn('1001', preview['sensor_error'])
            hardware.sensors.origin = np.zeros(7)
            with self.assertRaises(RuntimeError):hardware.sensors.snapshot()


class TestGo2StateCompatibility(unittest.TestCase):
    def test_configuration_is_explicit_and_validated(self):
        self.assertEqual(go2_normal_state_codes({}), {0})
        self.assertEqual(go2_normal_state_codes({'go2_normal_state_codes':[0,100]}), {0,100})
        for codes in ([], None, '100', [True], [100.0], [-1], [2**32]):
            with self.subTest(codes=codes), self.assertRaises(ValueError):
                go2_normal_state_codes({'go2_normal_state_codes':codes})

    def test_only_configured_code_is_accepted_and_pose_and_age_still_checked(self):
        hardware = UnitreeHardware.__new__(UnitreeHardware)
        hardware.sensors = Sensors('unitree')
        hardware.processes = []
        hardware.cameras = {}
        now = time.monotonic()
        hardware.arm_feedback = lambda:(ARM,now)
        with tempfile.TemporaryDirectory() as directory:
            hardware.run = Path(directory)
            path = hardware.run/'go2_snapshot.json'
            data = dict(valid=True, error_code=100, position=[0,0,.3], rpy=[0,0,0],
                        body_height=.3, monotonic_ns=int(now*1e9))
            def read():
                hardware.sensors = Sensors('unitree')
                hardware.sensors.origin = np.zeros(7)
                path.write_text(json.dumps({'sport_mode_state':data}))
                hardware._read_once()
                for name in ('front','wrist'):
                    hardware.sensors.put(name,now,np.zeros((480,640,3),np.uint8))
            hardware.go2_normal_codes = go2_normal_state_codes({})
            read()
            self.assertIn('error_code=100', hardware.sensors.error)
            hardware.go2_normal_codes = go2_normal_state_codes({'go2_normal_state_codes':[0,100]})
            read()
            self.assertIsNone(hardware.sensors.error)
            for code in (1001,1002,99999):
                data['error_code'] = code
                read()
                self.assertIn('error_code='+str(code), hardware.sensors.error)
            data['error_code'] = 100
            data['body_height'] = .07
            read()
            self.assertIsNone(hardware.sensors.error)
            hardware.sensors.origin = np.zeros(7)
            with self.assertRaisesRegex(RuntimeError, 'Go2 must already be standing'):
                hardware.sensors.snapshot()
            data['body_height'] = .3
            read()
            with patch('inference.real_robot.core.time.monotonic', return_value=now+1):
                with self.assertRaises(RuntimeError):hardware.sensors.snapshot()


class TestActionValidation(unittest.TestCase):
    def setUp(self):
        self.devices={}
        for kind in ('unitree','p450'):
            gateway=Gateway(FakeHardware(kind),execute=False)
            self.addCleanup(gateway.close)
            self.devices[kind]=DirectDevice(gateway)

    def test_downlink_echo_and_new_images_without_actuator_calls_or_plan(self):
        rows=np.zeros((15,14));rows[:,3:10]=ARM
        for d in self.devices.values():
            for method in ('send','hold','halt','check_execution'):
                setattr(d.gateway.hardware,method,lambda *a: self.fail('Actuator method called'))
        with tempfile.TemporaryDirectory() as folder:
            result=validate_action_downlink(self.devices,rows,{'execute_steps':3},Path(folder),0)
            self.assertTrue(result['action_downlink_verified'])
            self.assertTrue((Path(folder)/'downlink_000.json').exists())
            for name,device in self.devices.items():
                receipt=result['action_receipts'][name]
                self.assertTrue(receipt['validation_passed'])
                self.assertFalse(receipt['executable'])
                self.assertIsNone(device.gateway.plan)
                self.assertFalse(device.gateway.snapshots)
                self.assertGreater(min(result['post_validation_image_stamps'][name].values()),receipt['received_at'])
                with self.assertRaises(RuntimeError):
                    device.gateway.commit(dict(plan_id=receipt['request_id'],start_at=time.monotonic()+.1))

    def test_invalid_physical_prediction_returns_rejection_without_fault_or_execution(self):
        gateway=self.devices['unitree'].gateway
        rows=[[0,0,0,9,*ARM[1:]]]
        receipt=gateway.validate_actions(dict(request_id='a'*32,actions=rows))
        self.assertTrue(receipt['received'])
        self.assertFalse(receipt['validation_passed'])
        self.assertIn('joint limits',receipt['error'])
        self.assertIsNone(gateway.fault)
        self.assertIsNone(gateway.plan)
        self.assertFalse(gateway.hardware.sent)

    def test_wrong_width_nonfinite_and_execution_mode_rejected(self):
        gateway=self.devices['p450'].gateway
        for rows in ([[0]*10],[[float('nan'),0,0,0]],[]):
            with self.assertRaises(ValueError):
                gateway.validate_actions(dict(request_id='b'*32,actions=rows))
        gateway.execute=True
        try:
            with self.assertRaises(RuntimeError):
                gateway.validate_actions(dict(request_id='b'*32,actions=[[0]*4]))
        finally:gateway.execute=False


@unittest.skipUnless(os.environ.get('HETEROVLA_LOOPBACK_TEST') == '1', 'Explicit local socket permission required')
class TestHttpLoopback(unittest.TestCase):
    def test_unitree_only_execution_over_http_never_controls_p450(self):
        devices = {}; servers = []; gateways = []; observer_requests = []
        try:
            for kind in ('unitree', 'p450'):
                hardware = FakeHardware(kind)
                if kind == 'p450':
                    class GroundedSensors(FakeSensors):
                        def snapshot(self, diagnostic=False):
                            if not diagnostic:
                                raise RuntimeError('P450 is disarmed')
                            result = super().snapshot(diagnostic=True)
                            result['state'][11] = 0
                            return result
                    hardware.sensors = GroundedSensors(kind)
                gateway = Gateway(hardware, execute=kind == 'unitree')
                gateways.append(gateway)
                gateway.reset()
                base = handler(gateway, 'test-token')
                if kind == 'p450':
                    class ObserverHandler(base):
                        def dispatch(self):
                            observer_requests.append((self.command, self.path))
                            super().dispatch()
                    base = ObserverHandler
                server = ThreadingHTTPServer(('127.0.0.1', 0), base)
                servers.append(server)
                threading.Thread(target=server.serve_forever, kwargs={'poll_interval': .01}, daemon=True).start()
                device = Device('http://127.0.0.1:'+str(server.server_port), 'test-token')
                device.calibrate()
                devices[kind] = device
            class Policy:
                def infer(inner, observation):
                    self.assertEqual(observation['state'][28], 0)
                    rows = np.zeros((15,14)); rows[:,3:10] = ARM
                    return rows
            with tempfile.TemporaryDirectory() as directory:
                result = run_cycle(devices, Policy(), {'prompt':'task', 'execute_steps':3},
                                   Path(directory), execute=True, targets=('unitree',))
                self.assertTrue(result['executed'])
                self.assertEqual(result['execution_targets'], ['unitree'])
            self.assertTrue(gateways[0].hardware.sent)
            self.assertFalse(gateways[1].hardware.sent)
            self.assertIsNone(gateways[1].plan)
            self.assertTrue(observer_requests)
            self.assertTrue(all(method == 'GET' and path in ('/status', '/diagnostic')
                                for method, path in observer_requests))
        finally:
            for server in servers:
                server.shutdown(); server.server_close()
            for gateway in gateways:
                gateway.close()

    def test_observe_only_action_validation_over_http(self):
        gateway=Gateway(FakeHardware('p450'),execute=False)
        server=ThreadingHTTPServer(('127.0.0.1',0),handler(gateway,'test-token'))
        threading.Thread(target=server.serve_forever,kwargs={'poll_interval':.01},daemon=True).start()
        try:
            device=Device('http://127.0.0.1:'+str(server.server_port),'test-token')
            receipt=device.call('/validate_actions',dict(request_id='c'*32,actions=[[.1,0,0,0]]))
            self.assertTrue(receipt['validation_passed'])
            self.assertEqual(receipt['received_actions'],[[.1,0,0,0]])
            self.assertFalse(device.call('/status')['execution_enabled'])
            with self.assertRaises(RuntimeError):
                Device(device.url,'wrong-token').call('/validate_actions',dict(request_id='c'*32,actions=[[0]*4]))
            self.assertFalse(gateway.hardware.sent)
            self.assertIsNone(gateway.plan)
        finally:
            server.shutdown();server.server_close();gateway.close()

    def test_two_http_gateways_complete_cycle_and_reject_bad_token(self):
        token='test-only-token-32-characters-long'
        devices={};servers=[];gateways=[]
        try:
            for kind in ('unitree','p450'):
                gateway=Gateway(FakeHardware(kind),execute=True);gateways.append(gateway)
                server=ThreadingHTTPServer(('127.0.0.1',0),handler(gateway,token));servers.append(server)
                threading.Thread(target=server.serve_forever,kwargs={'poll_interval':.01},daemon=True).start()
                device=Device('http://127.0.0.1:'+str(server.server_port),token)
                device.calibrate();device.call('/reset',{});devices[kind]=device
                with self.assertRaises(RuntimeError):Device(device.url,'wrong-token').call('/status')
                self.assertIsNone(gateway.fault)
            class Policy:
                def infer(self, observation):
                    rows=np.zeros((15,14));rows[:,3:10]=ARM
                    return rows
            with tempfile.TemporaryDirectory() as directory:
                result=run_cycle(devices,Policy(),{'prompt':'task','execute_steps':3},Path(directory),execute=True)
                self.assertTrue(result['executed'])
                self.assertEqual(result['executed_steps'],3)
                for name in ('front','wrist','uav'):
                    self.assertEqual(cv2.imread(str(Path(directory)/'latest'/(name+'.jpg'))).shape,(480,640,3))
        finally:
            for server in servers:server.shutdown();server.server_close()
            for gateway in gateways:gateway.close()


if __name__=='__main__':unittest.main()
