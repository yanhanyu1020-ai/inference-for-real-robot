"""Map remote wall-clock timestamps onto the desktop clock timeline.

The clock exchange offset is defined as ``remote - desktop``.  This module
keeps that convention and only derives new values from immutable samples.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .models import ClockEstimate, ClockSample


LOW_RTT_MARGIN_NS = 2_000_000
MAX_EXTRAPOLATION_NS = 10_000_000_000
DEFAULT_MAX_CLOCK_GAP_NS = 10_000_000_000
P450_HEADER_MAX_SKEW_NS = 1_000_000_000  # Domain sanity check, NOT a precision bound.


class InsufficientClockAnchors(ValueError):
    """Raised when mapping would require a drift estimate without two anchors."""


def _round_div(numerator: int, denominator: int) -> int:
    """Return an integer quotient rounded to nearest, ties to even."""
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    sign = -1 if numerator < 0 else 1
    quotient, remainder = divmod(abs(numerator), denominator)
    twice_remainder = remainder * 2
    if twice_remainder > denominator or (twice_remainder == denominator and quotient % 2):
        quotient += 1
    return sign * quotient


def _ceil_div(numerator: int, denominator: int) -> int:
    if numerator < 0 or denominator <= 0:
        raise ValueError("ceil division requires a non-negative numerator and positive denominator")
    return (numerator + denominator - 1) // denominator


def _median_int(values: Sequence[int]) -> int:
    if not values:
        raise ValueError("median requires at least one value")
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return _round_div(ordered[middle - 1] + ordered[middle], 2)


def _median_ceil(values: Sequence[int]) -> int:
    """Return a median rounded upward, suitable for non-negative uncertainty."""
    if not values:
        raise ValueError("median requires at least one value")
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle] + 1) // 2


@dataclass(frozen=True)
class _Anchor:
    remote_wall_ns: int
    offset_ns: int
    uncertainty_ns: int
    sample_count: int
    selected_sequence: int


@dataclass(frozen=True)
class ClockTimeline:
    """A filtered sequence of offset anchors for one remote host."""

    host: str
    anchors: tuple[_Anchor, ...]
    max_gap_ns: int = DEFAULT_MAX_CLOCK_GAP_NS
    unsupported_gaps: tuple[tuple[int, int, int, int], ...] = ()

    def remote_to_desktop(self, remote_wall_ns: int, *, allow_single_anchor: bool = False) -> ClockEstimate:
        """Estimate desktop time for a remote timestamp without mutating samples."""
        if not isinstance(remote_wall_ns, int):
            raise TypeError("remote_wall_ns must be an integer nanosecond timestamp")
        if len(self.anchors) == 1 and remote_wall_ns != self.anchors[0].remote_wall_ns:
            if not allow_single_anchor:
                raise InsufficientClockAnchors(
                    "at least two anchors are required to map an off-anchor timestamp"
                )
            anchor = self.anchors[0]
            # A short recording cannot establish drift.  Preserve an explicitly
            # degraded, conservative constant-offset mapping rather than discard
            # its data/report altogether.
            uncertainty_ns = anchor.uncertainty_ns + abs(remote_wall_ns - anchor.remote_wall_ns)
            return ClockEstimate(
                host=self.host, offset_ns=anchor.offset_ns, jitter_ns=uncertainty_ns,
                sample_count=anchor.sample_count, selected_sequence=anchor.selected_sequence,
                desktop_time_ns=remote_wall_ns - anchor.offset_ns,
                estimated_error_ns=uncertainty_ns, degraded=True,
            )
        left, right, extrapolation_ns = self._bracketing_anchors(remote_wall_ns)
        offset_ns = _interpolated_offset(left, right, remote_wall_ns)
        uncertainty_ns = max(left.uncertainty_ns, right.uncertainty_ns)
        degraded = extrapolation_ns > MAX_EXTRAPOLATION_NS
        for remote_start, remote_end, _, _ in self.unsupported_gaps:
            if remote_start < remote_wall_ns < remote_end:
                # A finite placeholder keeps mapping APIs usable, but this is
                # deliberately not a precision claim for an unobserved clock.
                uncertainty_ns = max(uncertainty_ns, remote_end - remote_start)
                degraded = True
        if extrapolation_ns:
            uncertainty_ns += _observed_drift_error(left, right, extrapolation_ns)
        # Interpolation can round an offset by at most half a nanosecond.
        if left is not right:
            uncertainty_ns += 1
        return ClockEstimate(
            host=self.host,
            offset_ns=offset_ns,
            jitter_ns=uncertainty_ns,
            sample_count=max(left.sample_count, right.sample_count),
            selected_sequence=left.selected_sequence if left is right else None,
            desktop_time_ns=remote_wall_ns - offset_ns,
            estimated_error_ns=uncertainty_ns,
            degraded=degraded,
        )

    def to_mapping_dict(self) -> dict[str, Any]:
        """Return a self-contained mapping usable by offline post-processing."""
        anchors = [{
            "remote_wall_ns": item.remote_wall_ns,
            "offset_ns": item.offset_ns,
            "uncertainty_ns": item.uncertainty_ns,
            "sample_count": item.sample_count,
            "selected_sequence": item.selected_sequence,
        } for item in self.anchors]
        segments = [{
            "remote_start_ns": left.remote_wall_ns,
            "remote_end_ns": right.remote_wall_ns,
            "offset_start_ns": left.offset_ns,
            "offset_end_ns": right.offset_ns,
            "max_uncertainty_ns": (None if any(
                start < right.remote_wall_ns and end > left.remote_wall_ns
                for start, end, _, _ in self.unsupported_gaps)
                else max(left.uncertainty_ns, right.uncertainty_ns) + 1),
        } for left, right in zip(self.anchors, self.anchors[1:])]
        return {
            "host": self.host,
            "mapping_type": "piecewise_linear_offset",
            "anchors": anchors,
            "segments": segments,
            "max_clock_gap_ns": self.max_gap_ns,
            "unsupported_gaps": [{"remote_start_ns": start, "remote_end_ns": end,
                                  "desktop_start_ns": desktop_start, "desktop_end_ns": desktop_end,
                                  "estimated_error_ns": None}
                                 for start, end, desktop_start, desktop_end in self.unsupported_gaps],
            "valid_domain": {
                "remote_start_ns": self.anchors[0].remote_wall_ns,
                "remote_end_ns": self.anchors[-1].remote_wall_ns,
                "max_extrapolation_ns": MAX_EXTRAPOLATION_NS,
            },
            "uncertainty": {
                "anchor_max_ns": max(anchor.uncertainty_ns for anchor in self.anchors),
                "single_anchor_requires_degraded_constant_offset": len(self.anchors) == 1,
            },
        }

    def _bracketing_anchors(self, remote_wall_ns: int) -> tuple[_Anchor, _Anchor, int]:
        first, last = self.anchors[0], self.anchors[-1]
        if remote_wall_ns <= first.remote_wall_ns:
            second = self.anchors[1] if len(self.anchors) > 1 else first
            return first, second, first.remote_wall_ns - remote_wall_ns
        if remote_wall_ns >= last.remote_wall_ns:
            penultimate = self.anchors[-2] if len(self.anchors) > 1 else last
            return penultimate, last, remote_wall_ns - last.remote_wall_ns
        for index in range(1, len(self.anchors)):
            right = self.anchors[index]
            if remote_wall_ns <= right.remote_wall_ns:
                return self.anchors[index - 1], right, 0
        raise AssertionError("timestamp should have been bracketed by the final anchor")


def _interpolated_offset(left: _Anchor, right: _Anchor, remote_wall_ns: int) -> int:
    if left.remote_wall_ns == right.remote_wall_ns:
        return left.offset_ns
    elapsed_ns = remote_wall_ns - left.remote_wall_ns
    span_ns = right.remote_wall_ns - left.remote_wall_ns
    return left.offset_ns + _round_div((right.offset_ns - left.offset_ns) * elapsed_ns, span_ns)


def _observed_drift_error(left: _Anchor, right: _Anchor, extrapolation_ns: int) -> int:
    span_ns = abs(right.remote_wall_ns - left.remote_wall_ns)
    if span_ns == 0:
        return 0
    return _ceil_div(abs(right.offset_ns - left.offset_ns) * extrapolation_ns, span_ns)


def build_clock_timeline(samples: Sequence[ClockSample], window_ns: int,
                         max_gap_ns: int = DEFAULT_MAX_CLOCK_GAP_NS) -> ClockTimeline:
    """Build offset anchors, rejecting models contradicted by observed probes.

    Raises ValueError when a probe's RTT interval cannot overlap the fitted
    model's uncertainty interval; filtering must not hide clock changes.
    """
    if window_ns <= 0:
        raise ValueError("window_ns must be positive")
    if max_gap_ns <= 0:
        raise ValueError("max_gap_ns must be positive")
    if not samples:
        raise ValueError("at least one clock sample is required")
    host = samples[0].host
    # Sorting by remote wall time can hide a backward step. Inspect the
    # original chronology using the desktop monotonic clock first.
    chronological = sorted(samples, key=lambda sample: sample.local_receive_mono_ns)
    for left, right in zip(chronological, chronological[1:]):
        remote_elapsed = right.remote_monotonic_ns - left.remote_monotonic_ns
        local_elapsed = right.local_receive_mono_ns - left.local_receive_mono_ns
        remote_wall_elapsed = right.remote_receive_wall_ns - left.remote_receive_wall_ns
        local_wall_elapsed = right.local_receive_wall_ns - left.local_receive_wall_ns
        # Allow 50 ms jitter plus 2000 ppm elapsed-time disagreement. This
        # detects large discontinuities, not every possible subthreshold step.
        tolerance = 50_000_000 + max(remote_elapsed, local_elapsed, 0) // 500
        if (remote_elapsed < 0 or remote_wall_elapsed < 0 or local_wall_elapsed < 0
                or abs(remote_wall_elapsed - remote_elapsed) > tolerance
                or abs(local_wall_elapsed - local_elapsed) > tolerance):
            raise ValueError(f"{host} clock discontinuity; start a new episode after clock adjustment or reboot")
    previous_remote_ns = samples[0].remote_send_wall_ns
    for sample in samples:
        if sample.host != host:
            raise ValueError("clock timeline samples must belong to one host")
        if sample.remote_send_wall_ns < previous_remote_ns:
            raise ValueError("clock timeline samples must be ordered by remote wall time")
        previous_remote_ns = sample.remote_send_wall_ns

    first_remote_ns = samples[0].remote_send_wall_ns
    windows: list[list[ClockSample]] = []
    current_window = -1
    for sample in samples:
        window = (sample.remote_send_wall_ns - first_remote_ns) // window_ns
        if window != current_window:
            windows.append([])
            current_window = window
        windows[-1].append(sample)

    anchors = tuple(_window_anchor(window) for window in windows)
    gaps = tuple((left.remote_send_wall_ns, right.remote_send_wall_ns,
                  left.remote_send_wall_ns - left.offset_ns,
                  right.remote_send_wall_ns - right.offset_ns)
                 for left, right in zip(samples, samples[1:])
                 if max(right.remote_send_wall_ns - left.remote_send_wall_ns,
                        right.local_receive_wall_ns - left.local_receive_wall_ns) > max_gap_ns)
    timeline = ClockTimeline(host=host, anchors=anchors, max_gap_ns=max_gap_ns, unsupported_gaps=gaps)
    for sample in samples:
        if len(anchors) == 1:
            # The permissive single-anchor mapping adds elapsed time to its
            # error. That fallback must not mask contradictory observations.
            model_offset_ns = anchors[0].offset_ns
            model_error_ns = anchors[0].uncertainty_ns
        else:
            estimate = timeline.remote_to_desktop(sample.remote_send_wall_ns)
            model_offset_ns = estimate.offset_ns
            model_error_ns = estimate.estimated_error_ns
        residual_ns = abs(sample.offset_ns - model_offset_ns)
        # Each exchange admits a true offset within half its network RTT of
        # the midpoint estimate. Wide/asymmetric exchanges therefore remain
        # usable evidence even when excluded from the low-RTT anchor fit.
        # One extra ns covers rounding of the exchange's midpoint offset.
        allowed_ns = model_error_ns + _ceil_div(sample.rtt_ns, 2) + 1
        if residual_ns > allowed_ns:
            raise ValueError(
                f"{host} clock model inconsistent with probe sequence {sample.sequence}: "
                f"offset residual {residual_ns} ns exceeds interval allowance {allowed_ns} ns; "
                "clock adjustment or nonlinear drift requires a new episode or clock model"
            )
    return timeline


def _window_anchor(samples: Sequence[ClockSample]) -> _Anchor:
    minimum_rtt_ns = min(sample.rtt_ns for sample in samples)
    retained = [sample for sample in samples if sample.rtt_ns <= minimum_rtt_ns + LOW_RTT_MARGIN_NS]
    offset_ns = _median_int([sample.offset_ns for sample in retained])
    residual_ns = _median_ceil([abs(sample.offset_ns - offset_ns) for sample in retained])
    selected = min(retained, key=lambda sample: (sample.rtt_ns, sample.sequence))
    return _Anchor(
        remote_wall_ns=_median_int([sample.remote_send_wall_ns for sample in retained]),
        offset_ns=offset_ns,
        uncertainty_ns=max(_ceil_div(minimum_rtt_ns, 2), residual_ns),
        sample_count=len(retained),
        selected_sequence=selected.sequence,
    )


def choose_p450_time(header_ns: int | None, bag_ns: int) -> tuple[int, str]:
    """Prefer a positive header near receipt time, regardless of calendar date."""
    if header_ns is not None and header_ns > 0 and abs(header_ns - bag_ns) <= P450_HEADER_MAX_SKEW_NS:
        return header_ns, "header"
    return bag_ns, "bag"


def common_interval(first_times: Sequence[int], t0_guard_ns: int,
                    stop_request_ns: int) -> tuple[int, int]:
    """Return the interval shared by all streams after a guard period."""
    if not first_times:
        raise ValueError("at least one first-frame time is required")
    if t0_guard_ns < 0:
        raise ValueError("t0_guard_ns must not be negative")
    start_ns = max(first_times) + t0_guard_ns
    if stop_request_ns < start_ns:
        raise ValueError("stop request precedes the common interval")
    return start_ns, stop_request_ns
