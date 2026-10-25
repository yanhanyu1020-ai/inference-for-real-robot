"""CPU check of configurable action windows; never instantiate or train a model."""
import argparse
import json
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader
from chunk_settings import resolve_action_horizon
from dataset_15hz import JointV215HzDataset, as_openpi_sample


def check(root, horizon):
    manifest=json.loads((root/'reports/episode_manifest.json').read_text())
    result={}
    for split in ['train','val']:
        episodes=[e for e in manifest if e['split']==split]
        usable=[e for e in episodes if e['frames']>=horizon]
        ds=JointV215HzDataset(root,split,horizon=horizon)
        expected=sum(e['frames']-horizon+1 for e in usable)
        assert len(ds)==expected
        starts=np.asarray(ds.starts)
        for start,end in zip(ds.dataset.episode_data_index['from'],ds.dataset.episode_data_index['to']):
            selected=starts[(starts>=int(start)) & (starts<int(end))]
            assert len(selected)==max(0,int(end)-int(start)-horizon+1)
            assert np.all(selected+horizon<=int(end))
        for index in sorted({0,len(ds)//2,len(ds)-1}):
            sample=as_openpi_sample(ds[index])
            assert sample['actions'].shape==(horizon,14) and sample['state'].shape==(30,)
        batch=next(iter(DataLoader(ds,batch_size=4,num_workers=0)))
        assert tuple(batch['action'].shape)==(4,horizon,14)
        all_sources={e['source_episode'] for e in episodes}
        used_sources={e['source_episode'] for e in usable}
        result[split]=dict(fps=ds.fps,action_horizon=horizon,windows=len(ds),
            sources=len(all_sources),sources_with_windows=len(used_sources),
            sources_without_windows=sorted(all_sources-used_sources),
            eligible_episodes=len(usable),short_episodes_excluded=len(episodes)-len(usable),
            batch_action_shape=list(batch['action'].shape),strict_boundaries_verified=True)
        print('CHUNK_VERIFIED',horizon,split,len(ds),len(used_sources),flush=True)
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-root',type=Path,default=Path(__file__).resolve().parents[4]/'heterovla-data/V2_15Hz_train35')
    ap.add_argument('--action-horizon',type=int,help='Override the default chunk length independently of fps')
    ap.add_argument('--check-horizons',type=int,nargs='+',default=[10,15,30,50])
    ap.add_argument('--report',type=Path)
    args=ap.parse_args();root=args.data_root.expanduser().resolve()
    default=resolve_action_horizon(root,args.action_horizon)
    horizons=sorted(set([default,*args.check_horizons]))
    results={str(h):check(root,h) for h in horizons}
    report=dict(action_horizon_configurable=True,default_action_horizon=default,
        configured_default_datasets=results[str(default)],checks=results,
        fps_independent_of_action_horizon=True,model_training_started=False)
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
