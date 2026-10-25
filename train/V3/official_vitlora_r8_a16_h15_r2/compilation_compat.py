"""Compile abstract official TrainState inputs without checking JAX ArgInfo nodes.

JAX's explicit ``lower`` path temporarily rebuilds PyTrees with ``ArgInfo``
leaves. The pinned official TrainState's jaxtyping decorator rejects those
internal placeholders, although normal runtime values are valid. This narrow
scope changes no model, optimizer, loss, EMA, sharding, or donation operation.
It must finish before actual training starts.
"""
from __future__ import annotations

from typing import Any


def lower_and_compile(jitted: Any, *args: Any, **kwargs: Any) -> Any:
    """Return ``jitted.lower(*args, **kwargs).compile()`` with a temporary guard.

    Import lazily so importing this helper cannot initialize JAX before the
    caller selects its backend/GPU. The official context manager does not use
    ``finally`` in this pinned version, so explicitly restore its initial flag
    even when lowering or compilation raises. Runtime checking stays enabled
    when it was enabled on entry.
    """
    from openpi.shared import array_typing as at

    initial = at.config.jaxtyping_disable
    try:
        with at.disable_typechecking():
            return jitted.lower(*args, **kwargs).compile()
    finally:
        at.config.update("jaxtyping_disable", initial)
