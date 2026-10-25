import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import dataset_v3
from chunk_settings import resolve_action_horizon


class FakeDataset:
    fps=30
    episode_data_index={'from':[0,4],'to':[4,10]}
    gap=False
    padded=False
    imputed=False
    bad_imputation=False

    def __init__(self,*args,**kwargs):
        self.offsets=kwargs['delta_timestamps']['action']
        self.horizon=len(self.offsets)

    def __getitem__(self,index):
        times=index/30+np.asarray(self.offsets)
        if self.gap:times[-1]+=1/30
        valid=np.ones((self.horizon,14),bool)
        imputed=np.zeros_like(valid)
        if self.imputed:
            valid[0,0]=False
            imputed[0,0]=True
        if self.bad_imputation:
            imputed[0,1]=True
        return {'action':np.zeros((self.horizon,14)), 'action_valid':valid, 'action_imputed':imputed,
            'observation.state':np.zeros(30),'state_valid':np.ones(30,bool),'state_imputed':np.zeros(30,bool),
            'action_is_pad':np.full(self.horizon,self.padded),
            'source_task_time':times.reshape(-1,1),'source_task_time_is_pad':np.zeros(self.horizon,bool)}


class ChunkTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        p=self.root/'lerobot/local/heterovla_v3_train/meta';p.mkdir(parents=True)
        (p/'info.json').write_text(json.dumps({'fps':30}))
        (p/'tasks.jsonl').write_text(json.dumps({'task_index':0,'task':'中文任务'}))
        (self.root/'configs').mkdir()
        (self.root/'configs/action_chunk.json').write_text(json.dumps({'action_horizon':3}))
        (self.root/'configs/task.json').write_text(json.dumps({
            'same_task_for_all_included_recordings':True,
            'english_translation':'Pick up the block.'}))
        self.patch=patch.object(dataset_v3,'LeRobotDataset',FakeDataset);self.patch.start()

    def tearDown(self):
        self.patch.stop();self.temp.cleanup()

    def test_chunk_and_fps_are_independent(self):
        ds=dataset_v3.JointV3Dataset(self.root)
        self.assertEqual(ds.horizon,3);self.assertEqual(ds.fps,30)
        np.testing.assert_allclose(ds.dataset.offsets,[0,1/30,2/30])
        self.assertEqual(ds.starts,[0,1,4,5,6,7])
        self.assertEqual(ds[-1]['action'].shape,(3,14))

    def test_override_and_short_episodes(self):
        ds=dataset_v3.JointV3Dataset(self.root,horizon=5)
        self.assertEqual(ds.starts,[4,5])
        with self.assertRaises(ValueError):dataset_v3.JointV3Dataset(self.root,horizon=7)

    def test_gaps_and_padding_are_rejected(self):
        ds=dataset_v3.JointV3Dataset(self.root)
        ds.dataset.gap=True
        with self.assertRaises(ValueError):ds[0]
        ds.dataset.gap=False;ds.dataset.padded=True
        with self.assertRaises(ValueError):ds[0]

    def test_invalid_horizons(self):
        for value in [0,-1,True,1.5,'50']:
            with self.assertRaises(ValueError):resolve_action_horizon(self.root,value)
        self.assertEqual(resolve_action_horizon(self.root,50),50)

    def test_model_receives_configured_translation(self):
        ds=dataset_v3.JointV3Dataset(self.root)
        sample={**ds[0], 'task':'中文任务', 'observation.state':np.zeros(30)}
        for camera in ('front','wrist','uav'):
            sample[f'observation.images.{camera}']=np.zeros((3,4,4))
        self.assertEqual(dataset_v3.as_openpi_sample(sample)['prompt'], 'Pick up the block.')
        self.assertEqual(sample['task'], '中文任务')

    def test_missing_translation_or_multiple_tasks_are_rejected(self):
        path=self.root/'configs/task.json'
        path.write_text(json.dumps({'same_task_for_all_included_recordings':True}))
        with self.assertRaisesRegex(ValueError,'english_translation'):
            dataset_v3.JointV3Dataset(self.root)
        path.write_text(json.dumps({'same_task_for_all_included_recordings':False,
                                    'english_translation':'Pick up the block.'}))
        with self.assertRaisesRegex(ValueError,'single-task'):
            dataset_v3.JointV3Dataset(self.root)

    def test_only_declared_validation_accepts_imputed_values(self):
        train=self.root/'lerobot/local/heterovla_v3_train/meta'
        val=self.root/'lerobot/local/heterovla_v3_val/meta'
        val.mkdir(parents=True)
        for p in train.iterdir():
            (val/p.name).write_bytes(p.read_bytes())
        (self.root/'configs/imputation.json').write_text(json.dumps({
            'method':'previous_valid_value','splits':['train','val']}))
        train_ds=dataset_v3.JointV3Dataset(self.root,'train')
        train_ds.dataset.imputed=True
        with self.assertRaises(ValueError):train_ds[0]
        val_ds=dataset_v3.JointV3Dataset(self.root,'val')
        val_ds.dataset.imputed=True
        self.assertTrue(val_ds[0]['action_imputed'][0,0])
        val_ds.dataset.bad_imputation=True
        with self.assertRaisesRegex(ValueError,'imputation mask'):val_ds[0]


if __name__=='__main__':unittest.main()
