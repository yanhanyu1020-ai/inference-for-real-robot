from jointctl.models import ClockSample, EpisodeManifest


def test_clock_sample_computes_rtt_and_offset():
    sample = ClockSample.from_exchange(
        host="p450",
        sequence=1,
        local_send_wall_ns=1_000_000_000,
        local_send_mono_ns=10_000,
        remote_receive_wall_ns=1_100_001_000,
        remote_send_wall_ns=1_100_002_000,
        remote_monotonic_ns=55_000,
        local_receive_wall_ns=1_000_005_000,
        local_receive_mono_ns=15_000,
    )
    assert sample.rtt_ns == 4_000
    assert sample.offset_ns == 99_999_000


def test_manifest_json_round_trip_preserves_integer_nanoseconds():
    manifest = EpisodeManifest.new("joint_20260902_170000_ab12", "demo", "joint")
    restored = EpisodeManifest.from_dict(manifest.to_dict())
    assert restored.episode_id == manifest.episode_id
    assert isinstance(restored.created_desktop_ns, int)


def test_clock_sample_uses_exact_integer_offset_rounding_at_epoch_scale():
    sample = ClockSample.from_exchange(
        host="p450",
        sequence=2,
        local_send_wall_ns=1_000_000_000_000_000_000,
        local_send_mono_ns=10_000,
        remote_receive_wall_ns=2_000_000_000_000_000_000,
        remote_send_wall_ns=2_000_000_000_000_005_002,
        remote_monotonic_ns=55_000,
        local_receive_wall_ns=1_000_000_000_000_005_000,
        local_receive_mono_ns=16_000,
    )
    assert sample.offset_ns == 1_000_000_000_000_000_001
