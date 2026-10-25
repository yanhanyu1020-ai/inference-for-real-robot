"""Send a real dataset observation as state14; never contact robot execution endpoints."""
import argparse
import json
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url',default='ws://127.0.0.1:8001')
    parser.add_argument('--split',choices=('train','val'),default='val')
    parser.add_argument('--index',type=int,default=0)
    args=parser.parse_args()
    from . import configuration as c
    c.bootstrap()
    from dataset_v3 import OpenpiV3Dataset
    from inference.real_robot.controller import Policy
    from inference.real_robot.relative14 import check_policy_metadata,state14_from_state30
    sample=OpenpiV3Dataset(c.DATA,args.split,15)[args.index]
    obs={key:value for key,value in sample.items() if key!='actions'}
    obs['state']=state14_from_state30(obs['state'])
    policy=Policy(args.url)
    try:
        check_policy_metadata(policy.metadata)
        start=time.monotonic()
        actions=policy.infer(obs)
        print(json.dumps(dict(passed=True,input_state_shape=list(obs['state'].shape),
            action_shape=list(actions.shape),elapsed_s=time.monotonic()-start,
            output_encoding=policy.metadata['returned_action_encoding'],robot_commands_sent=False)))
    finally:
        policy.close()


if __name__=='__main__':main()
