"""CPU-only, cross-process probe of cached executable donation statistics."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

os.environ.update(JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")


def child(cache, enabled):
    import jax
    import jax.numpy as jnp
    import jaxlib

    jax.config.update("jax_compilation_cache_dir", cache)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
    jax.config.update("jax_enable_compilation_cache", enabled)
    from jax._src import compiler
    hits = []
    old_read = compiler._cache_read

    def read(*args, **kwargs):
        result = old_read(*args, **kwargs)
        hits.append(result[0] is not None)
        return result

    compiler._cache_read = read

    def donor(key, state):
        return ({"x": state["x"] * 1.01 + key[0].astype(jnp.float32),
                 "y": state["y"] - .01 * state["x"],
                 "ema": .99 * state["ema"] + .01 * state["x"]},
                jnp.mean(state["x"]))

    sig = {name: jax.ShapeDtypeStruct((262144,), jnp.float32) for name in ("x", "y", "ema")}
    fn = jax.jit(donor, donate_argnums=(1,))
    executable = fn.lower(jax.ShapeDtypeStruct((2,), jnp.uint32), sig).compile()
    memory = executable.memory_analysis()
    hlo = executable.as_text()
    header = hlo.splitlines()[0]
    values = {name: getattr(memory, name) for name in (
        "argument_size_in_bytes", "output_size_in_bytes", "alias_size_in_bytes", "temp_size_in_bytes")}
    # Prove donation actually aliases buffers when executing, independently of
    # the statistics or HLO header. The three outputs have identical shapes.
    state = {name: jnp.ones((262144,), dtype=jnp.float32) for name in ("x", "y", "ema")}
    old_pointers = {name: value.unsafe_buffer_pointer() for name, value in state.items()}
    new, _ = executable(jnp.array([0, 0], dtype=jnp.uint32), state)
    jax.block_until_ready(new)
    new_pointers = {name: value.unsafe_buffer_pointer() for name, value in new.items()}
    assert set(old_pointers.values()) == set(new_pointers.values())
    row = {"jax": jax.__version__, "jaxlib": jaxlib.__version__, "backend": jax.default_backend(),
           "cache_enabled": enabled, "cache_hits": hits,
           "memory": values, "hlo_header": header,
           "hlo_sha256": hashlib.sha256(hlo.encode()).hexdigest(),
           "execution_reused_all_donated_buffer_pointers": True,
           "cache_files": sorted(p.name for p in Path(cache).glob("*"))}
    print(json.dumps(row))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--cache")
    ap.add_argument("--disable-cache", action="store_true")
    args = ap.parse_args()
    if args.child:
        child(args.cache, not args.disable_cache)
        return
    rows = []
    with tempfile.TemporaryDirectory(prefix="heterovla_cpu_cache_probe_") as cache:
        for phase in ("cold", "warm", "disabled"):
            argv = [sys.executable, "-B", str(Path(__file__).resolve()), "--child", "--cache", cache]
            if phase == "disabled":
                argv.append("--disable-cache")
            result = subprocess.run(argv, text=True, capture_output=True, check=True)
            row = json.loads(result.stdout)
            row["phase"] = phase
            rows.append(row)
        assert any(rows[1]["cache_hits"]), rows[1]
        assert all(r["backend"] == "cpu" for r in rows)
        assert all(r["execution_reused_all_donated_buffer_pointers"] for r in rows)
        result = {"passed": True, "gpu_used": False, "rows": rows,
                  "alias_stats_lost_on_cache_hit": (
                      rows[0]["memory"]["alias_size_in_bytes"] > 0
                      and rows[1]["memory"]["alias_size_in_bytes"] == 0),
                  "disabled_cache_restores_alias_stats": (
                      rows[2]["memory"]["alias_size_in_bytes"] == rows[0]["memory"]["alias_size_in_bytes"])}
        output = Path(__file__).with_name("cache_memory_cpu_result.json")
        output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
