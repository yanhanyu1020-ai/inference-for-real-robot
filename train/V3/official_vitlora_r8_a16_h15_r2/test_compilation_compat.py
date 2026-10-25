"""CPU regression of lower/compile with the real official TrainState and step.

The model contains only tiny arrays. The official TrainState dataclass,
official train_step, AdamW, corrected frozen EMA wrapper, donation, abstract
inputs, and explicit lower/compile path are the production components.
"""
from __future__ import annotations

import os

os.environ.update(JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="", WANDB_MODE="disabled")

import dataclasses
import functools
import json

import configuration as c

c.bootstrap()

from flax import nnx
import jax
import jax.numpy as jnp
from jaxtyping import TypeCheckError
import numpy as np

from openpi.models import model as model_lib
from openpi.shared import array_typing as at
from openpi.training import config as config_lib, optimizer, sharding, utils

from compilation_compat import lower_and_compile


class SmallModel(model_lib.BaseModel):
    def __init__(self):
        super().__init__(action_dim=2, action_horizon=1, max_token_len=1)
        self.PaliGemma = nnx.Dict(
            img=nnx.Dict(
                kernel=nnx.Param(jnp.array([[.1, .03], [-.05, .08]], dtype=jnp.bfloat16)),
                lora_a=nnx.Param(jnp.array([[.1], [.2]], dtype=jnp.float32)),
                lora_b=nnx.Param(jnp.zeros((1, 2), dtype=jnp.float32)),
            ),
            llm=nnx.Dict(kernel=nnx.Param(jnp.array([[.3, .1], [-.1, .2]], dtype=jnp.float32))),
        )
        self.deterministic = True

    def compute_loss(self, rng, observation, actions, *, train=False):
        del rng, train
        image = self.PaliGemma.img
        kernel = image.kernel.value.astype(jnp.float32) + image.lora_a.value @ image.lora_b.value
        prediction = observation.state @ kernel + observation.state @ self.PaliGemma.llm.kernel.value
        return jnp.mean(jnp.square(prediction[:, None, :] - actions), axis=-1)

    def sample_actions(self, rng, observation, **kwargs):
        del rng, kwargs
        return jnp.zeros((observation.state.shape[0], 1, 2), dtype=jnp.float32)


def fresh_state(config, tx):
    graphdef, params = nnx.split(SmallModel())
    return utils.TrainState(
        step=jnp.asarray(0, dtype=jnp.int32), params=params, model_def=graphdef,
        opt_state=tx.init(params.filter(config.trainable_filter)), tx=tx,
        ema_decay=.99, ema_params=jax.tree.map(lambda x: x.copy(), params),
    )


def assert_bits_equal(left, right):
    assert jax.tree.structure(left) == jax.tree.structure(right)
    for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
        a = np.asarray(jax.device_get(a)); b = np.asarray(jax.device_get(b))
        np.testing.assert_array_equal(a.reshape(-1).view(np.uint8), b.reshape(-1).view(np.uint8))


def assert_typechecking_on(state):
    assert not at.config.jaxtyping_disable
    try:
        dataclasses.replace(state, step="invalid-step")
    except TypeCheckError:
        return
    raise AssertionError("Official TrainState runtime checking did not resume")


def main():
    assert jax.default_backend() == "cpu"
    assert not at.config.jaxtyping_disable
    cfg = config_lib.TrainConfig(
        name="cpu_real_trainstate_compilation_contract", exp_name="cpu_only",
        freeze_filter=c.FreezeViTDense(), ema_decay=.99,
        optimizer=optimizer.AdamW(),
        lr_schedule=optimizer.CosineDecaySchedule(warmup_steps=500, peak_lr=1e-5,
                                                  decay_steps=20000, decay_lr=1e-6),
    )
    # The optimizer object is static TrainState metadata. Keep the same one
    # across abstract/concrete states, as official init_train_state does.
    tx = optimizer.create_optimizer(cfg.optimizer, cfg.lr_schedule)
    state = fresh_state(cfg, tx)
    assert type(state) is utils.TrainState
    state_shape = jax.eval_shape(lambda: fresh_state(cfg, tx))
    batch = (
        model_lib.Observation(images={}, image_masks={},
                              state=jnp.array([[1., .5], [-.2, .3]], dtype=jnp.float32)),
        jnp.array([[[.1, -.1]], [[.2, .05]]], dtype=jnp.float32),
    )
    batch_shape = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), batch)
    mesh = sharding.make_mesh(1)
    placement = sharding.fsdp_sharding(state_shape, mesh)
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    official = c.official_trainer()
    step = c.corrected_train_step(official.train_step)

    def jitted_step():
        return jax.jit(functools.partial(step, cfg),
                       in_shardings=(replicated, placement, data_sharding),
                       out_shardings=(placement, replicated), donate_argnums=(1,))

    rng = jax.random.key(4)
    try:
        jitted_step().lower(rng, state_shape, batch_shape).compile()
    except TypeCheckError as error:
        original_error = str(error)
        assert "ArgInfo" in original_error, original_error
    else:
        raise AssertionError("Regression must reproduce the original ArgInfo typechecking failure")
    assert_typechecking_on(state)

    compiled = lower_and_compile(jitted_step(), rng, state_shape, batch_shape)
    assert_typechecking_on(state)
    # Direct jit is the official main-loop dispatch. Use independent states
    # because the production signature donates the state argument.
    direct, direct_info = jitted_step()(rng, fresh_state(cfg, tx), batch)
    actual, actual_info = compiled(rng, fresh_state(cfg, tx), batch)
    jax.block_until_ready(actual)
    assert int(actual.step) == 1
    assert_bits_equal(actual, direct)
    assert_bits_equal(actual_info, direct_info)
    assert_bits_equal(actual.params.filter(cfg.freeze_filter), state.params.filter(cfg.freeze_filter))
    assert_bits_equal(actual.ema_params.filter(cfg.freeze_filter), state.params.filter(cfg.freeze_filter))
    assert compiled.memory_analysis() is not None
    assert_typechecking_on(state)

    class DeliberateFailure(RuntimeError):
        pass

    class FailsLower:
        def lower(self, *args, **kwargs):
            assert at.config.jaxtyping_disable
            raise DeliberateFailure("lower failed")

    class FailsCompile:
        def lower(self, *args, **kwargs):
            assert at.config.jaxtyping_disable
            return self

        def compile(self):
            assert at.config.jaxtyping_disable
            raise DeliberateFailure("compile failed")

    for initial in (False, True):
        for failing in (FailsLower(), FailsCompile()):
            at.config.update("jaxtyping_disable", initial)
            try:
                lower_and_compile(failing)
            except DeliberateFailure:
                pass
            else:
                raise AssertionError("Compilation exception must propagate")
            assert at.config.jaxtyping_disable is initial
    at.config.update("jaxtyping_disable", False)
    assert_typechecking_on(state)
    print(json.dumps({
        "passed": True, "backend": jax.default_backend(),
        "state_class": f"{utils.TrainState.__module__}.{utils.TrainState.__name__}",
        "official_train_step_used": True, "original_arginfo_failure_reproduced": True,
        "abstract_lower_compile_with_donation_passed": True,
        "compiled_and_direct_jit_identical": True,
        "frozen_raw_and_ema_unchanged": True,
        "typechecking_restored_after_success_and_lower_compile_failures": True,
        "previously_disabled_flag_preserved": True,
        "gpu_used": False, "base_arrays_loaded": False,
    }, indent=2))


if __name__ == "__main__":
    main()
