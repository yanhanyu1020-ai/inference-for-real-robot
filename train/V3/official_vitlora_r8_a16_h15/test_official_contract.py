"""CPU-only contracts for the narrow official-trainer EMA correction.

Uses tiny nnx.State trees and a delegated AdamW step with the official EMA
expression. It does not build Pi0, load weights/data, or initialize a GPU.
"""

from __future__ import annotations

import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import dataclasses
import json

import configuration

configuration.bootstrap()

from flax import nnx, struct
import jax
import jax.numpy as jnp
import numpy as np
import optax


@struct.dataclass
class TinyTrainState:
    step: jax.Array
    params: nnx.State
    opt_state: object
    ema_params: nnx.State | None
    tx: object = struct.field(pytree_node=False)
    ema_decay: float | None = struct.field(pytree_node=False)


def _variable(value):
    return nnx.VariableState(nnx.Param, value)


def _params():
    # Include random bf16 numbers for which upstream old*.99 + old*.01
    # rounds differently than old. The correction must preserve every bit.
    frozen = jax.random.normal(jax.random.key(42), (16, 16)).astype(jnp.bfloat16)
    fp32 = jnp.arange(12, dtype=jnp.float32).reshape(3, 4) / 50
    return nnx.State({
        "PaliGemma": {
            "img": {
                "embedding": {"kernel": _variable(frozen)},
                "pos_embedding": _variable(frozen[0]),
                "Transformer": {
                    "encoderblock": {
                        "MultiHeadDotProductAttention_0": {
                            "query": {
                                "kernel": _variable(frozen),
                                "bias": _variable(frozen[1]),
                                "lora_a": _variable(fp32),
                                "lora_b": _variable(fp32.T * 0),
                            },
                        },
                        "LayerNorm_0": {
                            "scale": _variable(frozen[2]),
                            "bias": _variable(frozen[3]),
                        },
                    },
                },
            },
            "llm": {
                "layers": {
                    "attn": {"kernel": _variable(fp32)},
                    "action_expert": {"kernel": _variable(fp32 + 0.25)},
                },
            },
            # Prefix similarity must not accidentally freeze unrelated modules.
            "img2": {"kernel": _variable(fp32 + 0.75)},
        },
        "action_in_proj": {"kernel": _variable(fp32 + 1)},
        "action_out_proj": {"bias": _variable(fp32[0] + 2)},
    })


def _flat_values(state):
    return {key: value.value for key, value in state.flat_state().items()}


def _bits(value):
    value = np.asarray(jax.device_get(value))
    return value.reshape(-1).view(np.uint8)


def _equal_tree(left, right):
    assert jax.tree.structure(left) == jax.tree.structure(right)
    for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
        np.testing.assert_array_equal(_bits(a), _bits(b))


def delegated_step(config, rng, state, batch):
    """Small AdamW delegate mirroring the upstream state-update/EMA contract."""
    del batch
    trainable = state.params.filter(config.trainable_filter)
    # The values stand in for the model's gradients; using a state-shaped
    # tree exercises the same Optax and NNX plumbing as the official step.
    factor = jax.random.uniform(jax.random.fold_in(rng, state.step), ()) * 0.01 + 0.01
    gradients = jax.tree.map(lambda value: value * factor + factor, trainable)
    updates, opt_state = state.tx.update(gradients, state.opt_state, trainable)
    trainable = optax.apply_updates(trainable, updates)
    params = nnx.State.merge(trainable, state.params.filter(config.freeze_filter))
    result = dataclasses.replace(state, step=state.step + 1, params=params, opt_state=opt_state)
    if state.ema_decay is not None:
        result = dataclasses.replace(
            result,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new,
                state.ema_params, params,
            ),
        )
    info = {
        "loss": sum(jnp.sum(value * value) for value in jax.tree.leaves(trainable)),
        "grad_norm": optax.global_norm(gradients),
        "param_norm": optax.global_norm(params),
    }
    return result, info


def main():
    assert jax.default_backend() == "cpu"
    config = configuration.make_config(run_name="cpu_contract_only")
    params = _params()
    all_paths = set(params.flat_state())
    frozen = set(params.filter(config.freeze_filter).flat_state())
    trained = set(params.filter(config.trainable_filter).flat_state())
    expected_frozen = {
        path for path in all_paths
        if path[:2] == ("PaliGemma", "img") and path[-1] not in ("lora_a", "lora_b")
    }
    assert frozen == expected_frozen
    assert trained == all_paths - expected_frozen
    assert frozen.isdisjoint(trained)
    assert any(path[-1] == "lora_a" for path in trained)
    assert any(path[-1] == "lora_b" for path in trained)
    assert all(path in trained for path in all_paths if path[:2] != ("PaliGemma", "img"))

    tx = optax.adamw(learning_rate=1e-5, b1=0.9, b2=0.95, weight_decay=1e-10)
    state = TinyTrainState(
        step=jnp.asarray(0, jnp.int32), params=params,
        opt_state=tx.init(params.filter(config.trainable_filter)),
        ema_params=jax.tree.map(lambda value: value.copy(), params),
        tx=tx, ema_decay=0.99,
    )
    corrected = configuration.corrected_train_step(delegated_step)
    upstream_jit = jax.jit(lambda rng, state: delegated_step(config, rng, state, None))
    corrected_jit = jax.jit(lambda rng, state: corrected(config, rng, state, None))
    rng = jax.random.key(51)

    eager, eager_info = corrected(config, rng, state, None)
    eager_original, eager_original_info = delegated_step(config, rng, state, None)
    # Compare each dispatch mode with its own unwrapped delegate. Eager and
    # XLA may reduce bf16 norms differently; that is not an adapter change.
    _equal_tree(eager.params, eager_original.params)
    _equal_tree(eager.opt_state, eager_original.opt_state)
    _equal_tree(eager_info, eager_original_info)
    _equal_tree(
        eager.ema_params.filter(config.trainable_filter),
        eager_original.ema_params.filter(config.trainable_filter),
    )
    _equal_tree(eager.ema_params.filter(config.freeze_filter), params.filter(config.freeze_filter))

    frozen_initial = params.filter(config.freeze_filter)
    changed_upstream_frozen_ema = 0
    trainable_changed = False
    for _ in range(64):
        upstream, upstream_info = upstream_jit(rng, state)
        result, result_info = corrected_jit(rng, state)
        _equal_tree(result.params, upstream.params)
        _equal_tree(result.opt_state, upstream.opt_state)
        _equal_tree(result.step, upstream.step)
        _equal_tree(result_info, upstream_info)
        _equal_tree(
            result.ema_params.filter(config.trainable_filter),
            upstream.ema_params.filter(config.trainable_filter),
        )
        _equal_tree(result.params.filter(config.freeze_filter), frozen_initial)
        _equal_tree(result.ema_params.filter(config.freeze_filter), frozen_initial)
        changed_upstream_frozen_ema += sum(
            not np.array_equal(_bits(value), _bits(_flat_values(frozen_initial)[path]))
            for path, value in _flat_values(upstream.ema_params.filter(config.freeze_filter)).items()
        )
        trainable_changed = trainable_changed or any(
            not np.array_equal(_bits(value), _bits(_flat_values(params)[path]))
            for path, value in _flat_values(result.params.filter(config.trainable_filter)).items()
        )
        state = result
    assert changed_upstream_frozen_ema > 0, "Test must actually reproduce upstream frozen bf16 EMA drift"
    assert trainable_changed

    no_ema = dataclasses.replace(state, ema_decay=None, ema_params=None)
    raw, raw_info = upstream_jit(rng, no_ema)
    corrected_raw, corrected_raw_info = corrected_jit(rng, no_ema)
    _equal_tree(raw, corrected_raw)
    _equal_tree(raw_info, corrected_raw_info)
    assert corrected_raw.ema_params is None
    print(json.dumps({
        "passed": True, "backend": jax.default_backend(), "jit_steps": 64,
        "frozen_leaves": len(frozen), "trainable_leaves": len(trained),
        "filter_freezes_only_vit_dense": True,
        "vit_lora_and_all_non_vit_parameters_trainable": True,
        "frozen_raw_and_ema_bitwise_unchanged": True,
        "trainable_raw_ema_optimizer_metrics_match_delegate_bitwise": True,
        "upstream_frozen_ema_drift_reproduced": changed_upstream_frozen_ema > 0,
        "eager_and_jit_checked": True, "ema_disabled_passthrough_checked": True,
    }, indent=2))


if __name__ == "__main__":
    main()
