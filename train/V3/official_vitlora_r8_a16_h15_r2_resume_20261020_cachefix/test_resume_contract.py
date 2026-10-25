"""CPU-only round trip of the pinned official full training checkpoint.

This uses tiny arrays, the original regression model, the real TrainState,
official train_step/checkpoint functions, and the existing frozen EMA fix.
No Pi0 weights, dataset videos, or GPU are opened.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime
import functools
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import types

os.environ.update(JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="", WANDB_MODE="disabled")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--original-code",
        default="/meta_eon_cfs/home/hph/heterovla/train/V3/official_vitlora_r8_a16_h15_r2",
    )
    args = parser.parse_args()
    original = Path(args.original_code).resolve()
    sys.path.insert(0, str(original))
    import test_compilation_compat as tiny

    c = tiny.c
    jax, jnp, np = tiny.jax, tiny.jnp, tiny.np
    cfg = tiny.config_lib.TrainConfig(
        name="cpu_official_resume_contract", exp_name="cpu_only",
        freeze_filter=c.FreezeViTDense(), ema_decay=.99,
        optimizer=tiny.optimizer.AdamW(),
        lr_schedule=tiny.optimizer.CosineDecaySchedule(
            warmup_steps=500, peak_lr=1e-5, decay_steps=20000, decay_lr=1e-6),
        num_train_steps=20000, save_interval=1000, keep_period=5000,
        seed=42, resume=True, overwrite=False,
    )
    assert jax.default_backend() == "cpu"
    assert all(device.platform == "cpu" for device in jax.devices())
    tx = tiny.optimizer.create_optimizer(cfg.optimizer, cfg.lr_schedule)
    initial = tiny.fresh_state(cfg, tx)
    assert type(initial) is tiny.utils.TrainState
    frozen = jax.tree.map(lambda value: value.copy(), initial.params.filter(cfg.freeze_filter))
    official = c.official_trainer()
    corrected = c.corrected_train_step(official.train_step)
    update = jax.jit(functools.partial(corrected, cfg), donate_argnums=(1,))
    train_rng, _ = jax.random.split(jax.random.key(cfg.seed))
    batch = (
        tiny.model_lib.Observation(
            images={}, image_masks={},
            state=jnp.array([[1., .5], [-.2, .3]], dtype=jnp.float32)),
        jnp.array([[[.1, -.1]], [[.2, .05]]], dtype=jnp.float32),
    )

    def optimizer_counts(state):
        return [int(np.asarray(jax.device_get(value)))
                for path, value in jax.tree_util.tree_flatten_with_path(state.opt_state)[0]
                if path and isinstance(path[-1], jax.tree_util.GetAttrKey)
                and path[-1].name == "count"]

    def set_count(path, value):
        if (path and isinstance(path[-1], jax.tree_util.GetAttrKey)
                and path[-1].name == "count"):
            return jnp.asarray(12000, dtype=value.dtype)
        return value

    # Jump only the tiny fixture's counters, then perform a real official
    # update so LoRA B and the two Adam moments are demonstrably nonzero.
    state = dataclasses.replace(
        initial, step=jnp.asarray(12000, dtype=jnp.int32),
        opt_state=jax.tree_util.tree_map_with_path(set_count, initial.opt_state))
    assert optimizer_counts(state) == [12000, 12000]
    state, _ = update(train_rng, state, batch)
    jax.block_until_ready(state)
    assert int(state.step) == 12001
    assert optimizer_counts(state) == [12001, 12001]
    lora_b = [value.value for path, value in state.params.flat_state().items()
              if path[-1] == "lora_b"]
    assert lora_b and all(np.any(np.asarray(value)) for value in lora_b)
    assert any(np.any(np.asarray(value)) for path, value
               in jax.tree_util.tree_flatten_with_path(state.opt_state)[0]
               if getattr(path[-1], "name", None) not in ("count",))
    snapshot = jax.tree.map(lambda value: value.copy(), state)

    class TinyLoader:
        def data_config(self):
            return types.SimpleNamespace(norm_stats=None, asset_id=None)

    loader = TinyLoader()
    with tempfile.TemporaryDirectory(prefix="heterovla_cpu_resume_") as directory:
        checkpoint_root = Path(directory) / "checkpoints"
        manager, resuming = official._checkpoints.initialize_checkpoint_dir(
            checkpoint_root, keep_period=5000, overwrite=False, resume=False)
        assert resuming is False
        official._checkpoints.save_state(manager, state, loader, 12000)
        manager.wait_until_finished()
        saved = checkpoint_root / "12000"
        assert all((saved / name).is_file() for name in (
            "_CHECKPOINT_METADATA", "params/_METADATA", "train_state/_METADATA"))
        manager.close()

        manager, resuming = official._checkpoints.initialize_checkpoint_dir(
            checkpoint_root, keep_period=5000, overwrite=False, resume=True)
        assert resuming is True and manager.latest_step() == 12000
        shape = jax.eval_shape(lambda: tiny.fresh_state(cfg, tx))
        restored = official._checkpoints.restore_state(manager, shape, loader, step=12000)
        jax.block_until_ready(restored)
        tiny.assert_bits_equal(snapshot, restored)
        assert int(restored.step) == 12000 + 1
        assert optimizer_counts(restored) == [12001, 12001]
        tiny.assert_bits_equal(restored.params.filter(cfg.freeze_filter), frozen)
        tiny.assert_bits_equal(restored.ema_params.filter(cfg.freeze_filter), frozen)
        assert all(np.any(np.asarray(value.value)) for path, value
                   in restored.params.flat_state().items() if path[-1] == "lora_b")

        restarted_rng, _ = jax.random.split(jax.random.key(cfg.seed))
        tiny.assert_bits_equal(jax.random.key_data(train_rng), jax.random.key_data(restarted_rng))
        tiny.assert_bits_equal(
            jax.random.key_data(jax.random.fold_in(train_rng, state.step)),
            jax.random.key_data(jax.random.fold_in(restarted_rng, restored.step)))
        before_lr = float(cfg.lr_schedule.create()(optimizer_counts(state)[-1]))
        restored_lr = float(cfg.lr_schedule.create()(optimizer_counts(restored)[-1]))
        assert before_lr == restored_lr

        uninterrupted, expected_info = update(train_rng, state, batch)
        resumed, actual_info = update(restarted_rng, restored, batch)
        jax.block_until_ready(resumed)
        tiny.assert_bits_equal(uninterrupted, resumed)
        tiny.assert_bits_equal(expected_info, actual_info)
        assert int(resumed.step) == 12002
        assert optimizer_counts(resumed) == [12002, 12002]
        tiny.assert_bits_equal(resumed.params.filter(cfg.freeze_filter), frozen)
        tiny.assert_bits_equal(resumed.ema_params.filter(cfg.freeze_filter), frozen)
        manager.close()

    result = {
        "passed": True,
        "checked_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "backend": jax.default_backend(),
        "official_train_sha256": c.sha256(original / "official_source/scripts/train.py"),
        "test_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "state_class": f"{tiny.utils.TrainState.__module__}.{tiny.utils.TrainState.__name__}",
        "official_train_step_save_restore_used": True,
        "checkpoint_directory_step": 12000,
        "restored_actual_optimizer_step": 12001,
        "first_resumed_update_actual_step": 12002,
        "raw_ema_adam_step_bitwise_equal_after_restore": True,
        "resumed_next_update_bitwise_equal_to_uninterrupted": True,
        "nonzero_lora_b_preserved": True,
        "adam_counts_before_and_after": [[12001, 12001], [12002, 12002]],
        "lr_at_restored_count": restored_lr,
        "seed_split_and_fold_in_rng_identical": True,
        "frozen_raw_and_ema_exactly_unchanged": True,
        "gpu_used": False,
        "base_arrays_loaded": False,
        "videos_decoded": False,
    }
    output = Path(__file__).with_name("cpu_resume_contract.json")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
