"""Process-local SigLIP LoRA for openpi 15a9616; official sources stay intact.

Call ``install(rank=8, alpha=16)`` before constructing Pi0.  The adapter changes
only SigLIP transformer blocks: query/key/value/output and both MLP matrices.
All original parameter names and shapes are preserved, so the official base
checkpoint loader can supply them unchanged.  Every new ``lora_b`` is zero at
initialization; ``lora_a`` is random.  Freezing the ViT base is the training
configuration's responsibility (exclude only ``lora_a`` and ``lora_b`` from its
ViT freeze filter).  No Gemma or action-head module is replaced.
"""

from __future__ import annotations

import math
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp

from openpi.models import siglip
from openpi.training import sharding

_ORIGINAL_ENCODER_BLOCK = siglip.Encoder1DBlock
_INSTALLED: tuple[int, float] | None = None


class LoRADense(nn.Dense):
    rank: int = 8
    alpha: float = 16.0

    @nn.compact
    def __call__(self, inputs):
        base = super().__call__(inputs)
        a = self.param(
            "lora_a", nn.initializers.normal(stddev=0.01),
            (inputs.shape[-1], self.rank), self.param_dtype,
        )
        b = self.param(
            "lora_b", nn.initializers.zeros,
            (self.rank, self.features), self.param_dtype,
        )
        inputs, a, b = nn.dtypes.promote_dtype(inputs, a, b, dtype=self.dtype)
        delta = jnp.matmul(jnp.matmul(inputs, a, precision=self.precision), b, precision=self.precision)
        return base + delta * jnp.asarray(self.alpha / self.rank, base.dtype)


class LoRADenseGeneral(nn.DenseGeneral):
    """DenseGeneral plus a flattened low-rank branch, retaining its dense kernel."""

    rank: int = 8
    alpha: float = 16.0

    @nn.compact
    def __call__(self, inputs):
        # SigLIP uses only a final-axis q/k/v projection or final-two-axis output
        # projection, with no DenseGeneral batch dimensions.
        if self.batch_dims:
            raise ValueError("SigLIP LoRA does not support DenseGeneral batch_dims")
        axes = (self.axis,) if isinstance(self.axis, int) else tuple(self.axis)
        axes = tuple(axis % inputs.ndim for axis in axes)
        if axes != tuple(range(inputs.ndim - len(axes), inputs.ndim)):
            raise ValueError(f"SigLIP LoRA expects final contracting axes; got {axes}")
        features = (self.features,) if isinstance(self.features, int) else tuple(self.features)
        base = super().__call__(inputs)
        in_features = math.prod(inputs.shape[axis] for axis in axes)
        out_features = math.prod(features)
        a = self.param(
            "lora_a", nn.initializers.normal(stddev=0.01),
            (in_features, self.rank), self.param_dtype,
        )
        b = self.param(
            "lora_b", nn.initializers.zeros,
            (self.rank, out_features), self.param_dtype,
        )
        inputs, a, b = nn.dtypes.promote_dtype(inputs, a, b, dtype=self.dtype)
        flat = inputs.reshape(inputs.shape[:-len(axes)] + (in_features,))
        delta = jnp.matmul(jnp.matmul(flat, a, precision=self.precision), b, precision=self.precision)
        delta = delta.reshape(inputs.shape[:-len(axes)] + features)
        return base + delta * jnp.asarray(self.alpha / self.rank, base.dtype)


class SiglipLoRAAttention(nn.Module):
    """The exact no-dropout self-attention configuration used by SigLIP blocks.

    This deliberately implements only that configuration; it is never installed
    globally as a replacement for Flax attention (e.g. Gemma remains untouched).
    """

    num_heads: int
    dtype: Any = jnp.float32
    rank: int = 8
    alpha: float = 16.0

    @nn.compact
    def __call__(self, inputs_q, inputs_k):
        width = inputs_q.shape[-1]
        if width % self.num_heads:
            raise ValueError("SigLIP width must be divisible by number of heads")
        head_dim = width // self.num_heads
        kw = dict(
            features=(self.num_heads, head_dim), axis=-1, dtype=self.dtype,
            kernel_init=nn.initializers.xavier_uniform(), rank=self.rank, alpha=self.alpha,
        )
        query = LoRADenseGeneral(name="query", **kw)(inputs_q)
        key = LoRADenseGeneral(name="key", **kw)(inputs_k)
        value = LoRADenseGeneral(name="value", **kw)(inputs_k)
        attended = nn.dot_product_attention(query, key, value, deterministic=True, dtype=self.dtype)
        return LoRADenseGeneral(
            features=width, axis=(-2, -1), dtype=self.dtype,
            kernel_init=nn.initializers.xavier_uniform(), rank=self.rank,
            alpha=self.alpha, name="out",
        )(attended)


class SiglipLoRAMlp(nn.Module):
    mlp_dim: int | None = None
    dropout: float = 0.0
    dtype_mm: str = "float32"
    rank: int = 8
    alpha: float = 16.0

    @nn.compact
    def __call__(self, x, deterministic=True):
        width = x.shape[-1]
        kw = dict(
            dtype=self.dtype_mm, kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.normal(stddev=1e-6), rank=self.rank, alpha=self.alpha,
        )
        x = LoRADense(self.mlp_dim or 4 * width, name="Dense_0", **kw)(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.dropout)(x, deterministic)
        return LoRADense(width, name="Dense_1", **kw)(x)


def _make_encoder_block(rank: int, alpha: float):
    class SiglipLoRAEncoderBlock(nn.Module):
        mlp_dim: int | None = None
        num_heads: int = 12
        dropout: float = 0.0
        dtype_mm: str = "float32"

        @nn.compact
        def __call__(self, x, deterministic=True):
            out = {}
            x = sharding.activation_sharding_constraint(x)
            y = nn.LayerNorm(dtype=self.dtype_mm)(x)
            y = out["sa"] = SiglipLoRAAttention(
                num_heads=self.num_heads, dtype=self.dtype_mm, rank=rank,
                alpha=alpha, name="MultiHeadDotProductAttention_0",
            )(y, y)
            y = sharding.activation_sharding_constraint(y)
            y = nn.Dropout(rate=self.dropout)(y, deterministic)
            x = out["+sa"] = x + y
            y = nn.LayerNorm(dtype=self.dtype_mm)(x)
            y = out["mlp"] = SiglipLoRAMlp(
                mlp_dim=self.mlp_dim, dropout=self.dropout, dtype_mm=self.dtype_mm,
                rank=rank, alpha=alpha, name="MlpBlock_0",
            )(y, deterministic)
            y = sharding.activation_sharding_constraint(y)
            y = nn.Dropout(rate=self.dropout)(y, deterministic)
            x = out["+mlp"] = x + y
            x = sharding.activation_sharding_constraint(x)
            return x, out

    return SiglipLoRAEncoderBlock


def install(rank: int = 8, alpha: float = 16.0) -> dict[str, Any]:
    """Install an idempotent process-local extension before model construction."""
    global _INSTALLED
    if rank <= 0 or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("LoRA rank and alpha must be positive and finite")
    requested = (rank, float(alpha))
    if _INSTALLED is not None and _INSTALLED != requested:
        raise RuntimeError(f"SigLIP LoRA already installed with {_INSTALLED}, refusing {requested}")
    if _INSTALLED is None:
        if siglip.Encoder1DBlock is not _ORIGINAL_ENCODER_BLOCK:
            raise RuntimeError("SigLIP Encoder1DBlock was already replaced by another extension")
        siglip.Encoder1DBlock = _make_encoder_block(rank, float(alpha))
        _INSTALLED = requested
    return {
        "rank": rank, "alpha": float(alpha), "scale": alpha / rank,
        "targets": ["query", "key", "value", "out", "MlpBlock_0/Dense_0", "MlpBlock_0/Dense_1"],
        "lora_b_zero_init": True, "official_parameter_keys_preserved": True,
        "scope": "openpi.models.siglip.Encoder1DBlock only",
    }
