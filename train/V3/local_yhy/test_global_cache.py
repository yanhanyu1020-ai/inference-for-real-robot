import os; os.environ.update(JAX_ENABLE_COMPILATION_CACHE="false", JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="", WANDB_MODE="disabled", PYTHONDONTWRITEBYTECODE="1")
"""CPU-only: global cache disable with the pinned official TrainState/update.

A separate, explicitly enabled CPU process first populates a persistent cache.
The actual regression process disables it before importing JAX and instruments
the pinned compiler cache-read entry point. No model weights/videos/GPU load.
"""
import argparse
import dataclasses
import datetime
import functools
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def probe(original, cache, *, warm=False):
    # Only the disposable warm-cache subprocess is allowed to enable caching.
    # The production-like probe never changes the global False flag.
    if warm:
        os.environ["JAX_ENABLE_COMPILATION_CACHE"] = "true"
    import jax
    import jaxlib
    from jax._src import compiler

    assert jax.config.jax_enable_compilation_cache is warm
    assert jax.default_backend() == "cpu"
    assert all(device.platform == "cpu" for device in jax.devices())
    jax.config.update("jax_compilation_cache_dir", str(cache))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
    # This is the actual pinned JAX 0.5.3 persistent executable-cache reader.
    assert hasattr(compiler, "_cache_read"), "Pinned compiler._cache_read missing"
    cache_reads = []
    old_read = compiler._cache_read

    def observed_read(*args, **kwargs):
        cache_reads.append(True)
        return old_read(*args, **kwargs)

    compiler._cache_read = observed_read
    try:
        sys.path.insert(0, str(original))
        import test_compilation_compat as tiny

        c, jnp, np = tiny.c, tiny.jnp, tiny.np
        cfg = tiny.config_lib.TrainConfig(
            name="cpu_global_cache_disabled", exp_name="cpu_only",
            freeze_filter=c.FreezeViTDense(), ema_decay=.99,
            optimizer=tiny.optimizer.AdamW(),
            lr_schedule=tiny.optimizer.CosineDecaySchedule(
                warmup_steps=500, peak_lr=1e-5, decay_steps=20000, decay_lr=1e-6),
            seed=42,
        )
        tx = tiny.optimizer.create_optimizer(cfg.optimizer, cfg.lr_schedule)
        initial = tiny.fresh_state(cfg, tx)
        assert type(initial) is tiny.utils.TrainState
        shape = jax.eval_shape(lambda: tiny.fresh_state(cfg, tx))
        batch = (
            tiny.model_lib.Observation(
                images={}, image_masks={},
                state=jnp.array([[1., .5], [-.2, .3]], dtype=jnp.float32)),
            jnp.array([[[.1, -.1]], [[.2, .05]]], dtype=jnp.float32),
        )
        batch_shape = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), batch)
        mesh = tiny.sharding.make_mesh(1)
        placement = tiny.sharding.fsdp_sharding(shape, mesh)
        replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        data_sharding = jax.sharding.NamedSharding(
            mesh, jax.sharding.PartitionSpec(tiny.sharding.DATA_AXIS))
        official = c.official_trainer()
        corrected = c.corrected_train_step(official.train_step)

        def jitted():
            return jax.jit(
                functools.partial(corrected, cfg),
                in_shardings=(replicated, placement, data_sharding),
                out_shardings=(placement, replicated), donate_argnums=(1,))

        rng, _ = jax.random.split(jax.random.key(cfg.seed))
        executable = tiny.lower_and_compile(jitted(), rng, shape, batch_shape)
        memory = executable.memory_analysis()
        assert memory is not None and memory.alias_size_in_bytes > 0, str(memory)

        def set_count(path, value):
            if path and getattr(path[-1], "name", None) == "count":
                return jnp.asarray(12000, dtype=value.dtype)
            return value

        def counts(state):
            return [int(np.asarray(jax.device_get(value)))
                    for path, value in jax.tree_util.tree_flatten_with_path(state.opt_state)[0]
                    if path and getattr(path[-1], "name", None) == "count"]

        frozen = jax.tree.map(lambda x: x.copy(), initial.params.filter(cfg.freeze_filter))
        state = dataclasses.replace(
            initial, step=jnp.asarray(12000, jnp.int32),
            opt_state=jax.tree_util.tree_map_with_path(set_count, initial.opt_state))
        state, _ = jitted()(rng, state, batch)
        jax.block_until_ready(state)
        assert int(state.step) == 12001 and counts(state) == [12001, 12001]
        assert all(np.any(np.asarray(value.value))
                   for path, value in state.params.flat_state().items() if path[-1] == "lora_b")
        reference = jax.tree.map(lambda x: x.copy(), state)
        to_compile = jax.tree.map(lambda x: x.copy(), state)
        direct, direct_info = jitted()(rng, reference, batch)
        actual, actual_info = executable(rng, to_compile, batch)
        jax.block_until_ready(actual)
        tiny.assert_bits_equal(direct, actual)
        tiny.assert_bits_equal(direct_info, actual_info)
        assert int(actual.step) == 12002 and counts(actual) == [12002, 12002]
        tiny.assert_bits_equal(actual.params.filter(cfg.freeze_filter), frozen)
        tiny.assert_bits_equal(actual.ema_params.filter(cfg.freeze_filter), frozen)
        assert not tiny.at.config.jaxtyping_disable
        assert jax.config.jax_enable_compilation_cache is warm
        if not warm:
            assert not cache_reads, f"Unexpected persistent cache reads: {len(cache_reads)}"
        return {
            "passed": True, "jax": jax.__version__, "jaxlib": jaxlib.__version__,
            "backend": jax.default_backend(), "cache_enabled": warm,
            "persistent_cache_reads": len(cache_reads),
            "memory": {name: getattr(memory, name) for name in (
                "argument_size_in_bytes", "output_size_in_bytes", "alias_size_in_bytes",
                "temp_size_in_bytes")},
            "official_train_sha256": c.sha256(original / "official_source/scripts/train.py"),
            "state_class": f"{tiny.utils.TrainState.__module__}.{tiny.utils.TrainState.__name__}",
            "official_train_step_used": True,
            "abstract_donation_lower_compile_passed": True,
            "next_raw_adam_ema_and_metrics_bitwise_equal_to_official_direct_jit": True,
            "optimizer_step_before_and_after": [12001, 12002],
            "adam_counts_before_and_after": [[12001, 12001], [12002, 12002]],
            "frozen_raw_and_ema_exactly_unchanged": True,
            "gpu_used": False, "base_arrays_loaded": False, "videos_decoded": False,
            "cache_files": sorted(p.name for p in Path(cache).rglob("*") if p.is_file()),
        }
    finally:
        compiler._cache_read = old_read


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-code", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--warm-cache", action="store_true")
    parser.add_argument("--cache")
    parser.add_argument("--output", default=str(Path(__file__).with_name("global_cache_cpu_result.json")))
    args = parser.parse_args()
    original = Path(args.original_code).resolve()
    if args.warm_cache:
        assert args.cache
        print(json.dumps(probe(original, Path(args.cache), warm=True)))
        return
    assert "jax" not in sys.modules, "JAX imported before the global disable guard"
    with tempfile.TemporaryDirectory(prefix="v3_official_global_cache_cpu_") as directory:
        command = [sys.executable, "-B", str(Path(__file__).resolve()),
                   "--warm-cache", "--cache", directory, "--original-code", str(original)]
        warmed = subprocess.run(command, text=True, capture_output=True, check=True)
        warm_row = json.loads(warmed.stdout)
        assert warm_row["cache_files"], "Warm subprocess did not populate persistent cache"
        disabled = probe(original, Path(directory))
    result = {
        "passed": True,
        "checked_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "test_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "global_cache_disabled_before_jax_import": True,
        "populated_cache_bypassed_without_reader_calls": True,
        "global_cache_flag_remained_false": True,
        "gpu_used": False, "warm_cache_cpu_process": warm_row,
        "global_disabled_cpu_process": disabled,
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
