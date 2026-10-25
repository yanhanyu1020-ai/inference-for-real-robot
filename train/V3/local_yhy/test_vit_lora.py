"""CPU-only executable contract checks for the independent SigLIP LoRA extension."""

from __future__ import annotations

import os

# Set before any JAX import. This test must never initialize an accelerator.
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import json

import flax.linen as nn
from flax.traverse_util import flatten_dict, unflatten_dict
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import siglip
from openpi.training import sharding

import vit_lora


def _flat(tree):
    return flatten_dict(tree)


def _only_lora(path):
    return path[-1] in {"lora_a", "lora_b"}


def main():
    assert jax.default_backend() == "cpu"
    # Official activation constraints need a training mesh. A one-device CPU
    # mesh reproduces the same model operations without allocating a GPU.
    mesh = sharding.make_mesh(1)
    reports = []
    original = siglip.Encoder1DBlock
    image = jax.random.normal(jax.random.key(9), (2, 16, 16, 3))
    for scan in (False, True):
        for dtype in ("float32", "bfloat16"):
            siglip.Encoder1DBlock = original
            dense = siglip.Module(
                width=16, depth=2, mlp_dim=32, num_heads=2,
                patch_size=(4, 4), pool_type="none", scan=scan, dtype_mm=dtype,
            )
            with mesh:
                dense_params = dense.init(jax.random.key(0), image)["params"]
                dense_output = dense.apply({"params": dense_params}, image)[0]
            # install() is idempotent; reset only the saved installation marker
            # between independent test configurations, not in production code.
            vit_lora._INSTALLED = None
            vit_lora.install(8, 16)
            adapted = siglip.Module(
                width=16, depth=2, mlp_dim=32, num_heads=2,
                patch_size=(4, 4), pool_type="none", scan=scan, dtype_mm=dtype,
            )
            with mesh:
                new_params = adapted.init(jax.random.key(0), image)["params"]
            old_flat, new_flat = _flat(dense_params), _flat(new_params)
            assert set(old_flat) == {key for key in new_flat if not _only_lora(key)}
            assert all(new_flat[key].shape == value.shape for key, value in old_flat.items())
            # Strictly overlay the pretrained dense tree, as official weight
            # loading does, while retaining only new adapter initialization.
            new_flat.update(old_flat)
            params = unflatten_dict(new_flat)
            lora_a = [value for key, value in new_flat.items() if key[-1] == "lora_a"]
            lora_b = [value for key, value in new_flat.items() if key[-1] == "lora_b"]
            assert len(lora_a) == (6 if scan else 12)
            assert all(np.count_nonzero(value) > 0 for value in lora_a)
            assert all(np.count_nonzero(value) == 0 for value in lora_b)
            with mesh:
                output = adapted.apply({"params": params}, image)[0]
            np.testing.assert_array_equal(np.asarray(output), np.asarray(dense_output))

            # A zero B makes A's initial gradient zero by design; B must receive
            # finite nonzero gradient immediately. After a B update, A must also
            # receive gradient. This catches detached/dead low-rank branches.
            target = jax.random.normal(jax.random.key(41), output.shape)

            def loss(p):
                prediction = adapted.apply({"params": p}, image)[0].astype(jnp.float32)
                return jnp.mean((prediction - target) ** 2)

            with mesh:
                grads = _flat(jax.jit(jax.grad(loss))(params))
            assert all(np.isfinite(value).all() for value in grads.values())
            b_grads = [value for key, value in grads.items() if key[-1] == "lora_b"]
            assert all(np.count_nonzero(value) > 0 for value in b_grads)
            assert all(np.count_nonzero(value) == 0 for key, value in grads.items() if key[-1] == "lora_a")
            stepped_flat = {
                key: value - 0.1 * grads[key] if key[-1] == "lora_b" else value
                for key, value in new_flat.items()
            }
            with mesh:
                next_grads = _flat(jax.jit(jax.grad(loss))(unflatten_dict(stepped_flat)))
            assert all(np.count_nonzero(value) > 0 for key, value in next_grads.items() if key[-1] == "lora_a")
            # The extension must not touch standard Flax Dense/attention types.
            plain = nn.Dense(4)
            plain_keys = _flat(plain.init(jax.random.key(3), jnp.ones((2, 4)))["params"])
            assert not any(_only_lora(key) for key in plain_keys)
            reports.append({
                "scan": scan, "dtype": dtype, "dense_keys_preserved": True,
                "zero_delta_exact": True, "adapter_tensors": len(lora_a) + len(lora_b),
                "b_gradient_nonzero": True, "a_gradient_nonzero_after_b_update": True,
                "non_siglip_dense_untouched": True,
            })
    print(json.dumps({"passed": True, "backend": jax.default_backend(), "cases": reports}, indent=2))


if __name__ == "__main__":
    main()
