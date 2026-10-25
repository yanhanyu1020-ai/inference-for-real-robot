"""Offline LR scan utilities. The optimization objective remains official 32D MSE."""
import ast
import inspect
import json
from pathlib import Path
import textwrap

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

GROUPS = {'go2': (0, 3), 'piper': (3, 10), 'uav': (10, 14)}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    temp.replace(path)


def derive_loss_elements():
    # Use the installed official implementation verbatim, changing only its final
    # dimension reduction. Fail closed if upstream changes this return expression.
    from openpi.models import pi0
    tree = ast.parse(textwrap.dedent(inspect.getsource(pi0.Pi0.compute_loss)))
    fn = tree.body[0]
    expected = ast.parse('jnp.mean(jnp.square(v_t - u_t), axis=-1)', mode='eval').body
    returns = [n for n in ast.walk(fn) if isinstance(n, ast.Return)]
    assert len(returns) == 1 and ast.dump(returns[0].value) == ast.dump(expected)
    returns[0].value = ast.parse('jnp.square(v_t - u_t)', mode='eval').body
    fn.name = 'loss_elements'
    fn.decorator_list = []
    fn.returns = None
    ast.fix_missing_locations(tree)
    scope = dict(vars(pi0))
    exec(compile(tree, '<official_pi0_unreduced_loss>', 'exec'), scope)
    return scope['loss_elements']


loss_elements = derive_loss_elements()


def scalar_metrics(per_dim):
    out = {'loss': float(np.mean(per_dim)), 'real14_loss': float(np.mean(per_dim[:14]))}
    out.update({f'{k}_loss': float(np.mean(per_dim[a:b])) for k, (a, b) in GROUPS.items()})
    out.update({f'action_{i:02d}_loss': float(v) for i, v in enumerate(per_dim[:14])})
    return out


def eval_elements(model_def, params, rng, observation, actions):
    model = nnx.merge(model_def, params)
    model.eval()
    return loss_elements(model, rng, observation, actions, train=False).mean(axis=1)


def cache_batches(directory):
    from openpi.models.model import Observation
    for path in sorted(Path(directory).glob('batch_*.npz')):
        with np.load(path) as file:
            values = {k: file[k] for k in file.files}
        n = int(values.pop('count'))
        offset = int(values.pop('offset'))
        data = {'image': {}, 'image_mask': {}}
        for key, value in values.items():
            if '/' in key:
                group, name = key.split('/', 1)
                data[group][name] = value
            else:
                data[key] = value
        actions = data.pop('actions')
        observation = Observation.from_dict(data)
        yield offset, n, (observation, actions)


def evaluate(eval_fn, params, directory, seed=42):
    total = np.zeros(32, dtype=np.float64)
    count = 0
    for offset, n, batch in cache_batches(directory):
        values = np.asarray(eval_fn(params, jax.random.fold_in(jax.random.key(seed), offset), *batch))
        if not np.isfinite(values).all():
            raise FloatingPointError('Non-finite validation/probe loss')
        total += values[:n].sum(axis=0, dtype=np.float64)
        count += n
    if not count:
        raise ValueError('Empty evaluation cache')
    return {**scalar_metrics(total / count), 'windows': count}


def early_stop_update(history, step, value):
    """Each evaluation must improve the best *preceding* value by >=1%."""
    previous_best = min((x['value'] for x in history), default=float('inf'))
    streak = history[-1]['streak'] if history else 0
    streak = (0 if value <= previous_best * 0.99 else streak + 1) if step > 500 else 0
    return {'step': step, 'value': value, 'streak': streak}


def overfit(history):
    # Concrete first-round heuristic: three successive increases in fixed val
    # loss and three decreases in fixed train-probe loss, >1% endpoint change.
    if len(history) < 4 or history[-1]['step'] < 500:
        return False
    last = history[-4:]
    v = [x['val'] for x in last]
    t = [x['train'] for x in last]
    return (all(a < b for a, b in zip(v, v[1:])) and
            all(a > b for a, b in zip(t, t[1:])) and
            v[-1] > v[0] * 1.01 and t[-1] < t[0] * 0.99)
