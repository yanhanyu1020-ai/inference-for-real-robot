"""Strict dense-to-LoRA initialization. Imports JAX only inside GPU execution."""
import hashlib
import json
import math
from pathlib import Path

from v3_config import HEADS, parameter_group, trainable_path

RUNTIME_SHA256 = {
    'src/openpi/models/model.py': '0d74bc1d8f4623ac3d8e543710b8bf75f9b95e588a3ebf6127c659e001385cff',
    'src/openpi/models/lora.py': 'ced28716950c34d84a1a6d0b03b1d9550000732721df884818080b6e78cb3289',
    'src/openpi/models/gemma.py': '4d7a8df96aa6d6303aa028bc35d528b8fba5d6b0c123e884871772ae94306048',
    'src/openpi/models/pi0.py': 'e9fbaae98dae1b5e886cf2f67cc69dcb25f6f67ef9b16fea7b82f464d9a9dfd6',
    'src/openpi/models/pi0_config.py': '8e472113b1ba8df99913af26a55a0c385d06abf1b9cd23e7d755f2e3441a72b0',
    'src/openpi/training/weight_loaders.py': 'a54ac3222f81f8c6233241bc16b6f369c9416cdaa83f7058f96f7ddcfde1a3b4',
    'src/openpi/shared/nnx_utils.py': '4da7111ba3011c4d1c381c9ed17257771df6a50ed89182647c3fb6f79208005e',
    'src/openpi/training/optimizer.py': 'cfb5d6f6aec61fbe961287b4015c652260628a0ed4682a0dbdc271b8e65431fa',
    'scripts/train.py': '22f98a06aaa6e68e0f725d727ab95398238d1cae8a46bb94e829f330758a9baf',
}
EXPECTED_TRAINABLE = {'paligemma_lora': 27869184, 'action_expert_lora': 22118400,
                      'heads': 2165792, 'total': 52153376}


def check_runtime_contract(openpi):
    observed = {name: hashlib.sha256((Path(openpi) / name).read_bytes()).hexdigest() for name in RUNTIME_SHA256}
    if observed != RUNTIME_SHA256:
        raise RuntimeError('OpenPI runtime differs from the audited V3 initialization/LoRA contract')
    return observed


def lora_factor(path):
    name = str(path[-1])
    if name == 'lora_a' or name.endswith('_lora_a'):
        return 'a'
    if name == 'lora_b' or name.endswith('_lora_b'):
        return 'b'
    if 'lora' in '/'.join(map(str, path)).lower():
        raise ValueError(f'Unrecognized LoRA parameter: {path}')
    return None


def validate_dense_contract(parent, expected):
    """No casts or subset merges: every dense leaf must survive exactly."""
    dense = {path: leaf for path, leaf in expected.items() if lora_factor(path) is None}
    if set(parent) != set(dense):
        raise ValueError(f'Dense parameter path mismatch: missing={set(dense)-set(parent)}, extra={set(parent)-set(dense)}')
    for path, value in parent.items():
        reference = dense[path]
        if tuple(value.shape) != tuple(reference.shape) or str(value.dtype) != str(reference.dtype) or str(value.dtype) != 'float32':
            raise ValueError(f'Dense shape/dtype mismatch: {path}: {value.shape}/{value.dtype} vs {reference.shape}/{reference.dtype}')
    if not any(lora_factor(path) for path in expected):
        raise ValueError('Target tree has no LoRA parameters')


def parameter_report(flat, trainable_keys, *, enforce_counts=True):
    trainable_keys = set(trainable_keys)
    if trainable_keys != {path for path in flat if trainable_path(path)}:
        raise ValueError('NNX training filter differs from the explicit LoRA/head whitelist')
    groups = {}
    for path, leaf in flat.items():
        value = leaf.value if hasattr(leaf, 'value') else leaf
        name = parameter_group(path)
        group = groups.setdefault(name, dict(parameters=0, trainable=0, leaves=0, dtypes=[], paths=[]))
        count = math.prod(value.shape)
        group['parameters'] += count
        group['trainable'] += count if path in trainable_keys else 0
        group['leaves'] += 1
        group['dtypes'] = sorted(set(group['dtypes']) | {str(value.dtype)})
        group['paths'].append('/'.join(map(str, path)))
    if any(group['dtypes'] != ['float32'] for group in groups.values()):
        raise ValueError('Every stored parameter, frozen or trainable, must remain float32')
    if any(groups.get(name, {}).get('trainable', 0) for name in ('vit', 'gemma_base', 'other')):
        raise ValueError('A ViT, Gemma base or unknown parameter is trainable')
    if not groups.get('vit', {}).get('parameters') or not groups.get('gemma_base', {}).get('parameters'):
        raise ValueError('ViT/Gemma base groups are missing')
    for name in ('paligemma_lora', 'action_expert_lora', *HEADS):
        if not groups.get(name, {}).get('trainable'):
            raise ValueError(f'Required trainable group is missing: {name}')
    total_trainable = sum(group['trainable'] for group in groups.values())
    if enforce_counts:
        actual = dict(paligemma_lora=groups['paligemma_lora']['trainable'],
                      action_expert_lora=groups['action_expert_lora']['trainable'],
                      heads=sum(groups[name]['trainable'] for name in HEADS), total=total_trainable)
        if actual != EXPECTED_TRAINABLE:
            raise ValueError(f'Unexpected LoRA/head counts: {actual}')
    return dict(groups=groups, total_parameters=sum(group['parameters'] for group in groups.values()),
                trainable_parameters=total_trainable, parameter_dtypes=['float32'],
                full_unfreeze=False, lora=True, vit_frozen=True, gemma_base_frozen=True,
                trainable_whitelist_exact=True)


def frozen_digest(params):
    """Streaming host checks at initialization/first update/final, never a model copy."""
    import jax
    import numpy as np
    digest = hashlib.sha256()
    for path, leaf in sorted(params.flat_state().items()):
        if trainable_path(path):
            continue
        value = np.asarray(jax.device_get(leaf.value))
        digest.update('/'.join(map(str, path)).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.dtype.str.encode())
        digest.update(memoryview(value).cast('B'))
    return digest.hexdigest()


def checkpoint_metadata_flat(metadata):
    """Orbax metadata only: normalize its params/VariableState value wrappers."""
    from collections.abc import Mapping
    flat = {}

    def visit(value, path=()):
        if hasattr(value, 'tree'):
            visit(value.tree, path)
        elif isinstance(value, Mapping):
            for key, child in value.items():
                visit(child, (*path, str(key)))
        elif hasattr(value, 'shape') and hasattr(value, 'dtype'):
            normalized = path[1:] if path[:1] == ('params',) else path
            normalized = normalized[:-1] if normalized[-1:] == ('value',) else normalized
            if normalized in flat:
                raise ValueError(f'Duplicate checkpoint metadata path: {normalized}')
            flat[normalized] = value
        else:
            raise ValueError(f'Unrecognized parameter metadata node: {path}: {type(value)}')

    visit(metadata)
    return flat


def cpu_shape_preflight(config, weights_path):
    """Trace real model/optimizer shapes on CPU; restore no tensor values."""
    import jax
    import numpy as np
    import orbax.checkpoint as ocp
    if jax.default_backend() != 'cpu':
        raise RuntimeError('V3 shape preflight must use the CPU backend')
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:1]), ('fsdp',))
    shape, _ = init_lora_train_state(config, jax.random.key(config.seed), mesh, resume=True)
    report = parameter_report(shape.params.flat_state(), shape.params.filter(config.trainable_filter).flat_state())
    checkpointer = ocp.PyTreeCheckpointer()
    try:
        metadata = checkpointer.metadata(str(weights_path))
    finally:
        checkpointer.close()
    parent = checkpoint_metadata_flat(metadata)
    from flax.traverse_util import flatten_dict
    validate_dense_contract(parent, flatten_dict(shape.params.to_pure_dict()))
    report.update(parent_dense_metadata_leaves=len(parent), dense_metadata_exact_match=True,
                  abstract_model_only=True, checkpoint_arrays_loaded=False, gpu_initialized=False)
    return report


def init_lora_train_state(config, init_rng, mesh, *, resume, weights_path=None, proof_path=None):
    """Own initializer: official state/sharding structure, no frozen bf16 cast."""
    import flax.nnx as nnx
    from flax.traverse_util import flatten_dict, unflatten_dict
    import jax
    import jax.numpy as jnp
    import numpy as np
    from openpi.models import model as model_lib
    from openpi.training import optimizer, sharding, utils as training_utils
    tx = optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng, partial_params=None):
        _, model_rng = jax.random.split(rng)
        model = config.model.create(model_rng)
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # Supply the COMPLETE tree, including newly initialized LoRA.
            # This avoids depending on partial-replacement semantics in NNX.
            complete = flatten_dict(state.to_pure_dict())
            loaded = flatten_dict(partial_params)
            if set(loaded) != {path for path in complete if lora_factor(path) is None}:
                raise ValueError('Dense partial initializer keys changed after validation')
            complete.update(loaded)
            state.replace_by_pure_dict(unflatten_dict(complete))
            model = nnx.merge(graphdef, state)
        params = nnx.state(model)
        # Official LoRA initializes BOTH factors randomly. Zero every B factor
        # explicitly, including FFN *_lora_b; keep A's normal initialization.
        params = params.map(lambda path, leaf: leaf.replace(jnp.zeros_like(leaf.value)) if lora_factor(path) == 'b' else leaf)
        return training_utils.TrainState(step=0, params=params, model_def=nnx.graphdef(model), tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)), ema_decay=None, ema_params=None)

    shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(shape, mesh, log=True)
    parameter_report(shape.params.flat_state(), shape.params.filter(config.trainable_filter).flat_state())
    if resume:
        return shape, state_sharding
    if weights_path is None:
        raise ValueError('Fresh V3 requires explicit parent dense checkpoint')
    parent = model_lib.restore_params(Path(weights_path), restore_type=np.ndarray)
    parent_flat = flatten_dict(parent)
    expected = flatten_dict(shape.params.to_pure_dict())
    validate_dense_contract(parent_flat, expected)
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    state = jax.jit(init, in_shardings=replicated, out_shardings=state_sharding)(init_rng, parent)
    jax.block_until_ready(state)
    a_count = b_count = dense_count = 0
    dense_sha = hashlib.sha256()
    for path, leaf in sorted(state.params.flat_state().items()):
        value = np.asarray(jax.device_get(leaf.value))
        factor = lora_factor(path)
        if factor == 'b':
            if np.any(value != 0):
                raise ValueError(f'Fresh LoRA B is nonzero: {path}')
            b_count += 1
        elif factor == 'a':
            if not np.isfinite(value).all() or not np.any(value != 0):
                raise ValueError(f'Fresh LoRA A is zero/nonfinite: {path}')
            a_count += 1
        else:
            if not np.array_equal(value, parent_flat[path]):
                raise ValueError(f'Parent dense parameter was not loaded exactly: {path}')
            dense_sha.update('/'.join(map(str, path)).encode())
            dense_sha.update(memoryview(value).cast('B'))
            dense_count += 1
    if not a_count or a_count != b_count:
        raise ValueError('LoRA A/B pairing counts differ')
    if proof_path:
        Path(proof_path).write_text(json.dumps(dict(dense_leaves=dense_count, dense_all_exactly_equal=True,
            dense_value_sha256=dense_sha.hexdigest(), lora_a_leaves=a_count, lora_b_leaves=b_count,
            all_lora_b_exact_zero=True, all_lora_a_nonzero=True, all_parameter_dtype='float32',
            frozen_cast_removed=True, optimizer_phase_step=0), indent=2)+'\n')
    return state, state_sharding
