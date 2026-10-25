"""Strict JSON reporting and fail-fast logged-metric checks; stdlib only.

Non-finite numbers are represented as JSON null, with their original location
and NaN/+Infinity/-Infinity label retained in explicit metadata. Reporting must
remain possible after a numerical failure so the queue can observe that failure.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
import math
import numbers
import os
from pathlib import Path
import tempfile
from typing import Any


class NonFiniteMetricError(FloatingPointError):
    """The offending metric row has already been durably written as valid JSON."""

    def __init__(self, record: dict[str, Any], issues: list[dict[str, str]]):
        self.record = record
        self.nonfinite_metrics = issues
        description = ", ".join(f"{item['path']}={item['value']}" for item in issues)
        super().__init__(f"Non-finite logged training metric(s): {description}")


def _pointer(parent: str, key: Any) -> str:
    return parent + "/" + str(key).replace("~", "~0").replace("/", "~1")


def sanitize(value: Any) -> tuple[Any, list[dict[str, str]]]:
    """Return a JSON-compatible copy and every non-finite JSON-pointer location.

    Optional array/scalar libraries are supported through ``tolist``/``item``
    without importing or initializing JAX, NumPy, or an accelerator. Unsupported
    objects fail explicitly instead of silently changing their meaning.
    """
    issues: list[dict[str, str]] = []

    def visit(item: Any, path: str) -> Any:
        if item is None or isinstance(item, (str, bool)):
            return item
        if isinstance(item, numbers.Integral):
            return int(item)
        if isinstance(item, numbers.Real):
            number = float(item)
            if math.isfinite(number):
                # Preserve native finite JSON scalars, normalizing only foreign
                # scalar types that the stdlib encoder cannot serialize.
                return item if isinstance(item, float) else number
            label = "NaN" if math.isnan(number) else "+Infinity" if number > 0 else "-Infinity"
            issues.append({"path": path or "/", "value": label})
            return None
        if isinstance(item, Mapping):
            result = {}
            for key, child in item.items():
                if not isinstance(key, str):
                    raise TypeError(f"JSON object keys must be strings, got {type(key).__name__} at {path or '/'}")
                result[key] = visit(child, _pointer(path, key))
            return result
        if isinstance(item, (list, tuple)):
            return [visit(child, _pointer(path, index)) for index, child in enumerate(item)]
        if hasattr(item, "tolist") and callable(item.tolist):
            converted = item.tolist()
            if converted is item:
                raise TypeError(f"tolist() did not convert {type(item).__name__} at {path or '/'}")
            return visit(converted, path)
        if hasattr(item, "item") and callable(item.item):
            converted = item.item()
            if converted is item:
                raise TypeError(f"item() did not convert {type(item).__name__} at {path or '/'}")
            return visit(converted, path)
        raise TypeError(f"Unsupported JSON value {type(item).__name__} at {path or '/'}")

    return visit(value, ""), issues


def sanitize_record(record: Mapping[str, Any], *, issue_key: str = "nonfinite_values") -> dict[str, Any]:
    """Sanitize a record, retaining previous evidence and adding new evidence."""
    if not isinstance(record, Mapping):
        raise TypeError("A runtime record must be a JSON object")
    safe, issues = sanitize(record)
    if issues:
        previous = safe.get(issue_key, [])
        if not isinstance(previous, list):
            raise TypeError(f"{issue_key} must be a list when present")
        # Preserve prior evidence during failure/status republishing without
        # duplicating it on each status publication.
        combined = list(previous)
        for issue in issues:
            if issue not in combined:
                combined.append(issue)
        safe[issue_key] = combined
        safe["nonfinite_detected"] = True
    return safe


def dumps_record(record: Mapping[str, Any], **kwargs: Any) -> str:
    """Serialize a sanitized record, unconditionally rejecting JSON extensions."""
    kwargs.pop("allow_nan", None)
    kwargs.setdefault("ensure_ascii", False)
    return json.dumps(sanitize_record(record), allow_nan=False, **kwargs)


def atomic_json(path: str | Path, record: Mapping[str, Any]) -> dict[str, Any]:
    """Atomically publish strict JSON, including after NaN/Inf was detected."""
    path = Path(path)
    safe = sanitize_record(record)
    encoded = json.dumps(safe, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return safe


def append_metrics(path: str | Path, row: Mapping[str, Any]) -> dict[str, Any]:
    """Write one valid JSONL row, then fail the training on any non-finite metric.

    Call this before the upstream W&B log call or the next training status is
    published. The outer trainer exception handler can then publish ``failed``
    with ``error.nonfinite_metrics`` (or the exception's record) safely.
    """
    safe = sanitize_record(row, issue_key="nonfinite_metrics")
    encoded = json.dumps(safe, ensure_ascii=False, allow_nan=False) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    issues = safe.get("nonfinite_metrics", [])
    if issues:
        raise NonFiniteMetricError(safe, issues)
    return safe
