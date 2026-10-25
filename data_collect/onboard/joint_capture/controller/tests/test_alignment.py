from __future__ import annotations

import pytest

from jointctl.alignment import (
    build_clock_timeline,
    choose_p450_time,
    common_interval,
)
from jointctl.models import ClockEstimate, ClockSample


EPOCH_NS = 1_788_336_367_000_000_000


def synthetic_samples(*, offset_ns: int, drift_ppm: int, duration_s: int,
                      outlier_every: int) -> list[ClockSample]:
    """Deterministic four-timestamp exchanges with periodic high-RTT probes."""
    samples: list[ClockSample] = []
    for sequence in range(duration_s):
        local_midpoint_ns = EPOCH_NS + sequence * 1_000_000_000
        current_offset_ns = offset_ns + drift_ppm * sequence * 1_000
        rtt_ns = 500_000_000 if sequence % outlier_every == 0 else 2_000_000
        half_rtt_ns = rtt_ns // 2
        remote_wall_ns = local_midpoint_ns + current_offset_ns
        samples.append(ClockSample(
            host="p450",
            sequence=sequence,
            local_send_wall_ns=local_midpoint_ns - half_rtt_ns,
            local_send_mono_ns=local_midpoint_ns - half_rtt_ns,
            remote_receive_wall_ns=remote_wall_ns,
            remote_send_wall_ns=remote_wall_ns,
            remote_monotonic_ns=local_midpoint_ns,
            local_receive_wall_ns=local_midpoint_ns + half_rtt_ns,
            local_receive_mono_ns=local_midpoint_ns + half_rtt_ns,
            rtt_ns=rtt_ns,
            offset_ns=current_offset_ns,
        ))
    return samples


def test_timeline_recovers_offset_and_500ppm_drift_with_outliers():
    samples = synthetic_samples(
        offset_ns=-2_000_000_000,
        drift_ppm=-500,
        duration_s=600,
        outlier_every=10,
    )
    timeline = build_clock_timeline(samples, window_ns=5_000_000_000)
    estimate = timeline.remote_to_desktop(samples[-1].remote_send_wall_ns)
    assert abs(estimate.desktop_time_ns - samples[-1].local_receive_wall_ns) < 5_000_000
    assert estimate.estimated_error_ns < 10_000_000


def test_p450_header_is_preferred_when_epoch_is_sane():
    assert choose_p450_time(1_788_336_367_051_827_000, 1_788_336_367_120_694_000)[1] == "header"


def test_p450_bag_time_is_used_for_zero_header():
    assert choose_p450_time(0, 1_788_336_367_120_694_000)[1] == "bag"


@pytest.mark.parametrize("header,bag,source", [
    (1_000_000_000_000, 1_000_080_000_000, "header"),
    (1_788_336_367_000_000_000, 1_000_080_000_000, "bag"),
    (1_000_000_000_000, 1_005_000_000_000, "bag"),
    (-1, 1_000_000_000_000, "bag"),
])
def test_p450_header_domain_check_does_not_require_calendar_date(header, bag, source):
    assert choose_p450_time(header, bag) == (header if source == "header" else bag, source)


@pytest.mark.parametrize("step", [500_000_000, -2_000_000_000])
def test_timeline_rejects_clock_step_instead_of_interpolating_it(step):
    from dataclasses import replace
    samples = synthetic_samples(offset_ns=0, drift_ppm=0, duration_s=12, outlier_every=100)
    samples = [replace(s, remote_receive_wall_ns=s.remote_receive_wall_ns + (step if i >= 6 else 0),
                       remote_send_wall_ns=s.remote_send_wall_ns + (step if i >= 6 else 0),
                       offset_ns=s.offset_ns + (step if i >= 6 else 0))
               for i, s in enumerate(samples)]
    samples.sort(key=lambda sample: sample.remote_send_wall_ns)
    with pytest.raises(ValueError, match="clock discontinuity"):
        build_clock_timeline(samples, window_ns=5_000_000_000)


def test_timeline_interpolates_offsets_between_windows_with_integer_math():
    samples = synthetic_samples(offset_ns=1_000, drift_ppm=1_000,
                                duration_s=12, outlier_every=100)
    timeline = build_clock_timeline(samples, window_ns=5_000_000_000)
    remote_time_ns = samples[7].remote_send_wall_ns + 500_000_000
    estimate = timeline.remote_to_desktop(remote_time_ns)
    # Remote time runs 1,001 ns per desktop microsecond in this synthetic case.
    assert estimate.desktop_time_ns == EPOCH_NS + 7_499_500_500
    assert estimate.degraded is False


def test_timeline_marks_far_extrapolation_degraded_and_increases_error():
    samples = synthetic_samples(offset_ns=0, drift_ppm=1_000,
                                duration_s=12, outlier_every=100)
    timeline = build_clock_timeline(samples, window_ns=5_000_000_000)
    near = timeline.remote_to_desktop(samples[-1].remote_send_wall_ns + 5_000_000_000)
    far = timeline.remote_to_desktop(samples[-1].remote_send_wall_ns + 20_000_000_000)
    assert near.degraded is False
    assert far.degraded is True
    assert far.estimated_error_ns > near.estimated_error_ns


def test_common_interval_starts_after_all_first_frames_and_ends_at_stop_request():
    assert common_interval((100, 250, 180), 50, 1_000) == (300, 1_000)


def test_odd_rtt_anchor_uncertainty_rounds_up_conservatively():
    sample = ClockSample(
        host="p450", sequence=1,
        local_send_wall_ns=100, local_send_mono_ns=100,
        remote_receive_wall_ns=110, remote_send_wall_ns=110,
        remote_monotonic_ns=110,
        local_receive_wall_ns=105, local_receive_mono_ns=105,
        rtt_ns=5, offset_ns=8,
    )
    estimate = build_clock_timeline((sample,), window_ns=5_000_000_000).remote_to_desktop(110)
    assert estimate.estimated_error_ns == 3


def test_single_anchor_maps_its_exact_remote_timestamp():
    sample = synthetic_samples(offset_ns=123, drift_ppm=0, duration_s=1,
                               outlier_every=100)[0]
    estimate = build_clock_timeline((sample,), window_ns=5_000_000_000).remote_to_desktop(
        sample.remote_send_wall_ns
    )
    assert estimate.desktop_time_ns == EPOCH_NS


def test_single_anchor_rejects_off_anchor_timestamp_without_drift_evidence():
    sample = synthetic_samples(offset_ns=123, drift_ppm=0, duration_s=1,
                               outlier_every=100)[0]
    timeline = build_clock_timeline((sample,), window_ns=5_000_000_000)
    with pytest.raises(ValueError, match="at least two anchors"):
        timeline.remote_to_desktop(sample.remote_send_wall_ns + 1)


def test_two_offset_residuals_round_half_nanosecond_up_for_uncertainty():
    samples = (
        ClockSample("p450", 1, 0, 0, 1, 1, 1, 0, 0, 0, 0),
        ClockSample("p450", 2, 0, 0, 2, 2, 2, 0, 0, 0, 1),
    )
    estimate = build_clock_timeline(samples, window_ns=5_000_000_000).remote_to_desktop(2)
    assert estimate.estimated_error_ns >= 1


def test_alignment_estimate_round_trip_preserves_mapping_status_fields():
    estimate = ClockEstimate(
        host="p450", offset_ns=-2_000, jitter_ns=400, sample_count=3,
        selected_sequence=9, desktop_time_ns=1_788_336_367_000_000_000,
        estimated_error_ns=1_000, degraded=True,
    )
    restored = ClockEstimate.from_dict(estimate.to_dict())
    assert restored.desktop_time_ns == estimate.desktop_time_ns
    assert restored.estimated_error_ns == estimate.estimated_error_ns
    assert restored.degraded is True
