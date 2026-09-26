import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fit import decide


def test_healthy_server_without_residency_is_inconclusive():
    assert decide(exit_code=None, shared_bytes=0, health_ok=True) == "inconclusive"
    assert decide(exit_code=None, shared_bytes=1, health_ok=True) == "inconclusive"


def test_residency_evidence_controls_fit_or_spill():
    assert decide(
        exit_code=None,
        shared_bytes=0,
        health_ok=True,
        residency="device-resident",
    ) == "fit"
    assert decide(
        exit_code=None,
        shared_bytes=0,
        health_ok=True,
        residency="spill",
    ) == "spill"


def test_nonzero_exit_or_failed_health_is_exit():
    assert decide(exit_code=1, shared_bytes=0, health_ok=True) == "exit"
    assert decide(exit_code=None, shared_bytes=0, health_ok=False) == "exit"


def test_nonfinite_or_negative_shared_reading_is_rejected():
    for value in (float("nan"), float("inf"), -1):
        try:
            decide(exit_code=None, shared_bytes=value, health_ok=True)
        except ValueError as exc:
            assert "shared_bytes" in str(exc)
        else:
            raise AssertionError(f"invalid shared reading accepted: {value!r}")


def test_unknown_residency_label_is_rejected():
    try:
        decide(exit_code=None, shared_bytes=0, health_ok=True, residency="maybe")
    except ValueError as exc:
        assert "residency" in str(exc)
    else:
        raise AssertionError("unknown residency labels must fail")


if __name__ == "__main__":
    test_healthy_server_without_residency_is_inconclusive()
    test_residency_evidence_controls_fit_or_spill()
    test_nonzero_exit_or_failed_health_is_exit()
    test_nonfinite_or_negative_shared_reading_is_rejected()
    test_unknown_residency_label_is_rejected()
    print("ok")
