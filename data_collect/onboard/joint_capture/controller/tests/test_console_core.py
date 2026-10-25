import pytest


def test_memory_uses_available_not_free():
    from jointctl.telemetry import parse_meminfo
    assert parse_meminfo('MemTotal: 1000 kB\nMemFree: 20 kB\nMemAvailable: 250 kB\n') == {
        'total_bytes':1024000,'available_bytes':256000,'used_bytes':768000,'used_percent':75.0}


@pytest.mark.parametrize('text',['MemTotal: 1000 kB','MemTotal: 0 kB\nMemAvailable: 0 kB',
                                'MemTotal: 10 kB\nMemAvailable: 20 kB'])
def test_bad_memory_is_unknown_not_zero(text):
    from jointctl.telemetry import parse_meminfo
    with pytest.raises(ValueError): parse_meminfo(text)


def test_unsafe_integer_cannot_lose_browser_precision():
    from jointctl.telemetry import safe_json
    assert safe_json({'t0':1789364402353640400,'offset':-1789359871201231562,'count':5}) == {
        't0':'1789364402353640400','offset':'-1789359871201231562','count':5}


def test_operation_lock_excludes_competitor_and_releases(tmp_path):
    from jointctl.operation_lock import operation_lock, OperationBusy
    with operation_lock(tmp_path):
        with pytest.raises(OperationBusy):
            with operation_lock(tmp_path): pass
    with operation_lock(tmp_path): pass
