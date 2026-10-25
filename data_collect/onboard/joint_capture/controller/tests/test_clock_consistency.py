import pytest

from jointctl.alignment import build_clock_timeline
from jointctl.models import ClockSample


def probes(*, count=100, step_ns=0, pulse_ns=0, pulse_rtt_ns=1_000_000):
    samples = []
    for sequence in range(count):
        start = 10**12 + sequence * 200_000_000
        rtt = pulse_rtt_ns if sequence == 34 else 1_000_000
        offset = (step_ns if sequence >= 35 else 0) + (pulse_ns if sequence == 34 else 0)
        remote = start + rtt // 2 + offset
        samples.append(ClockSample.from_exchange(
            host="p450", sequence=sequence,
            local_send_wall_ns=start, local_send_mono_ns=start,
            local_receive_wall_ns=start + rtt, local_receive_mono_ns=start + rtt,
            remote_receive_wall_ns=remote, remote_send_wall_ns=remote,
            remote_monotonic_ns=start + rtt // 2,
        ))
    return samples


@pytest.mark.parametrize("step_ns", [20_000_000, -20_000_000, 3_000_000])
def test_rejects_subthreshold_clock_step_that_contradicts_probe_intervals(step_ns):
    # A narrow reported error must not hide the smoothing of a wall-clock step.
    with pytest.raises(ValueError, match="clock model inconsistent with probe"):
        build_clock_timeline(probes(step_ns=step_ns), 5_000_000_000)


@pytest.mark.parametrize("rtt_ns", [1_000_000, 6_000_000])
def test_checks_transient_probe_even_when_low_rtt_filter_discards_it(rtt_ns):
    with pytest.raises(ValueError, match="clock model inconsistent with probe"):
        build_clock_timeline(probes(pulse_ns=20_000_000, pulse_rtt_ns=rtt_ns), 5_000_000_000)


def test_single_anchor_does_not_hide_inconsistent_probe_with_degraded_mapping():
    with pytest.raises(ValueError, match="clock model inconsistent with probe"):
        build_clock_timeline(probes(count=40, step_ns=20_000_000), 50_000_000_000)


def test_asymmetric_high_rtt_probe_is_compatible_with_narrow_clock_model():
    # A 20 ms midpoint bias is plausible inside a 60 ms exchange's RTT interval.
    samples = probes(pulse_ns=20_000_000, pulse_rtt_ns=60_000_000)
    estimate = build_clock_timeline(samples, 5_000_000_000).remote_to_desktop(
        samples[34].remote_send_wall_ns
    )
    assert estimate.offset_ns == 0
    assert estimate.estimated_error_ns == 500_001
    assert estimate.degraded is False
