"""Recompute V3 state statistics from numerical Parquet columns, without loading videos."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def statistics(values, names):
    rows = []
    for index, name in enumerate(names):
        column = values[:, index]
        distinct, counts = np.unique(column, return_counts=True)
        row = dict(dimension=index+1, name=name, count=len(column),
                   mean=float(column.mean()), std=float(column.std(ddof=0)),
                   minimum=float(column.min()), maximum=float(column.max()),
                   q01=float(np.quantile(column, .01)), q99=float(np.quantile(column, .99)),
                   unique_count=len(distinct),
                   modal_value=float(distinct[np.argmax(counts)]),
                   modal_fraction=float(counts.max()/len(column)))
        if len(distinct) <= 8:
            row['value_counts'] = {str(float(v)): int(n) for v,n in zip(distinct,counts)}
        rows.append(row)
    return rows


def main():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,default=project.parent/'heterovla-data/aligned/V3')
    parser.add_argument('--output',type=Path,default=project/'outputs/analysis/v3_state_statistics')
    args = parser.parse_args()
    root = args.data_root/'lerobot/local/heterovla_v3_train'
    schema = json.loads((args.data_root/'configs/schema.json').read_text())
    info = json.loads((root/'meta/info.json').read_text())
    names = schema['state_names']
    horizon = schema['default_action_horizon']
    assert len(names)==30 and horizon==15 and info['features']['observation.state']['names']==names
    all_rows, starts, seen = [], [], set()
    files = sorted((root/'data').rglob('*.parquet'))
    for file in files:
        table = pq.read_table(file,columns=['observation.state','frame_index','episode_index'])
        values = np.asarray(table['observation.state'].to_pylist(),dtype=np.float64)
        assert values.shape==(len(table),30) and np.isfinite(values).all(), str(file)
        episodes = set(table['episode_index'].to_pylist())
        assert len(episodes)==1 and not episodes & seen, str(file)
        seen.update(episodes)
        assert table['frame_index'].to_pylist()==list(range(len(table))), str(file)
        assert len(table)>=horizon, str(file)
        all_rows.append(values)
        # JointV3Dataset starts each H15 sample at range(start, end-H+1).
        # The final H-1 rows are future action targets, not independent state inputs.
        starts.append(values[:len(values)-horizon+1])
    all_rows = np.concatenate(all_rows)
    starts = np.concatenate(starts)
    assert len(files)==info['total_episodes']==856
    assert len(all_rows)==info['total_frames']==45997
    assert len(starts)==34013
    norm_file = args.data_root/'assets/heterovla_v3/norm_stats.json'
    norm = json.loads(norm_file.read_text())['norm_stats']['state']
    frames = statistics(all_rows,names)
    windows = statistics(starts,names)
    comparison = {}
    for label, rows in [('all_training_frames',frames),('actual_H15_input_states',windows)]:
        comparison[label] = {field: float(np.max(np.abs(np.array([r[field] for r in rows])-norm[field])))
                             for field in ('mean','std','q01','q99')}
    result = dict(split='train',units='raw physical units, before normalization and padding',
                  weighting='equal weight per row, not equal weight per episode or recording',
                  note='H15 input states are the dataset sampling population, not an exact optimizer-step frequency audit',
                  episodes=len(files),all_training_frames=len(all_rows),H15_input_states=len(starts),
                  horizon=horizon,normalization_file=str(norm_file.resolve()),
                  normalization_sha256=hashlib.sha256(norm_file.read_bytes()).hexdigest(),
                  stored_norm_max_absolute_difference=comparison,
                  all_frame_statistics=frames,training_input_statistics=windows)
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'statistics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    for label, rows in [('all_frames',frames),('training_inputs',windows)]:
        columns=[key for key in rows[0] if key!='value_counts']
        with (args.output/(label+'.csv')).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=columns,extrasaction='ignore')
            writer.writeheader();writer.writerows(rows)
    print(json.dumps({k:result[k] for k in ('episodes','all_training_frames','H15_input_states',
          'stored_norm_max_absolute_difference')},indent=2))
    for row in windows:
        print('%2d %-24s mean=% .8f std=%.8f unique=%d mode=%.4f%%' %
              (row['dimension'],row['name'],row['mean'],row['std'],row['unique_count'],100*row['modal_fraction']))
    print('Report:',args.output)


if __name__=='__main__':main()
