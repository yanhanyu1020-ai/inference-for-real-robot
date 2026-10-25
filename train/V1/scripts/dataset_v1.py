"""Strict H-step LeRobot loader: no chunk crosses a source gap or episode end."""
from pathlib import Path
import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


class JointV1Dataset:
    def __init__(self, root, split="train", horizon=10):
        self.root=Path(root).expanduser().resolve()
        self.horizon=horizon
        self.dataset=LeRobotDataset(f"local/heterovla_v1_{split}",
            root=self.root/"lerobot/local"/f"heterovla_v1_{split}",video_backend="pyav",
            delta_timestamps={"action":[i/10 for i in range(horizon)],
                              "action_valid":[i/10 for i in range(horizon)]})
        self.starts=[]
        for start,end in zip(self.dataset.episode_data_index['from'],self.dataset.episode_data_index['to']):
            self.starts.extend(range(int(start),int(end)-horizon+1))
        if not self.starts:raise ValueError('No complete action chunks')

    def __len__(self):return len(self.starts)

    def __getitem__(self,index):
        x=self.dataset[self.starts[index]]
        if x['action_is_pad'].any() or not x['action_valid'].all():
            raise ValueError('Invalid or padded action reached strict loader')
        return x


def as_openpi_sample(sample):
    """LeRobot (CHW float 0..1) -> openpi raw observation (HWC uint8)."""
    image={}
    for name,slot in [('front','base_0_rgb'),('wrist','left_wrist_0_rgb'),('uav','right_wrist_0_rgb')]:
        a=np.asarray(sample[f'observation.images.{name}'])
        if a.ndim!=3:raise ValueError(f'Unexpected image shape: {a.shape}')
        if a.shape[0]==3:a=np.transpose(a,(1,2,0))
        if np.issubdtype(a.dtype,np.floating):a=np.rint(np.clip(a,0,1)*255).astype(np.uint8)
        image[slot]=a
    result={'state':np.asarray(sample['observation.state'],dtype=np.float32),
        'image':image,'image_mask':{k:np.bool_(True) for k in image},
        'prompt':sample['task']}
    if 'action' in sample:result['actions']=np.asarray(sample['action'],dtype=np.float32)
    return result


class OpenpiV1Dataset:
    def __init__(self,root,split='train',horizon=10):self.base=JointV1Dataset(root,split,horizon)
    def __len__(self):return len(self.base)
    def __getitem__(self,index):return as_openpi_sample(self.base[index])
