import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fill_curve import CHECKPOINTS, REPS, WINDOW, headline, row_from_timings

PP = 512
TG = 128


def _timings(**overrides):
    value = {
        "cache_n": 32768,
        "prompt_n": PP,
        "prompt_per_second": 300.0,
        "predicted_n": TG,
        "predicted_per_second": 40.0,
    }
    value.update(overrides)
    return value


def test_constants_match_the_plan():
    assert CHECKPOINTS == (0, 32768, 65536, 131072, 190464)
    assert WINDOW == 262144
    assert REPS == 3


def test_valid_row_requires_exact_cache_and_counts():
    row = row_from_timings(
        32768,
        _timings(),
        window=WINDOW,
        tokens_cached=32768 + PP,
        tokens_evaluated=32768 + PP,
    )
    assert row["valid"] is True
    assert row["pp512"] == 300.0
    assert row["tg128"] == 40.0


def test_wrong_counts_and_cached_prefill_off_by_one_are_invalid():
    assert row_from_timings(32768, _timings(prompt_n=513))["valid"] is False
    assert row_from_timings(32768, _timings(cache_n=32767, prompt_n=1))["valid"] is False
    assert row_from_timings(32768, _timings(predicted_n=127))["valid"] is False


def test_fitted_window_is_enforced_not_module_default():
    row = row_from_timings(190464, _timings(cache_n=190464), window=131072)
    assert row["valid"] is False
    assert "window" in " ".join(row["errors"])


def test_missing_nonfinite_or_nonpositive_rates_are_invalid():
    missing = _timings()
    del missing["prompt_per_second"]
    assert row_from_timings(32768, missing)["valid"] is False
    for value in (float("nan"), float("inf"), 0.0, -1.0):
        row = row_from_timings(32768, _timings(predicted_per_second=value))
        assert row["valid"] is False
        assert row["tg128"] is None


def test_truncation_and_cross_check_mismatch_are_invalid():
    assert row_from_timings(32768, _timings(), truncated=True)["valid"] is False
    assert row_from_timings(32768, _timings(), tokens_cached=1)["valid"] is False
    assert row_from_timings(32768, _timings(), tokens_evaluated=1)["valid"] is False


def test_final_tokens_cached_includes_generated_tail_but_is_bounded():
    prompt_total = 32768 + PP
    assert row_from_timings(
        32768,
        _timings(),
        tokens_cached=prompt_total + TG - 1,
        tokens_evaluated=prompt_total,
    )["valid"] is True
    assert row_from_timings(
        32768,
        _timings(),
        tokens_cached=prompt_total + TG + 1,
        tokens_evaluated=prompt_total,
    )["valid"] is False


def test_headline_requires_exactly_three_valid_reps():
    rows = [row_from_timings(32768, _timings(prompt_per_second=rate)) for rate in (300, 100, 200)]
    result = headline(rows)
    assert result == {
        "valid": True,
        "valid_reps": 3,
        "pp512_median": 200.0,
        "tg128_median": 40.0,
    }
    invalid = headline(rows[:2])
    assert invalid["valid"] is False
    assert invalid["valid_reps"] == 2
    assert invalid["pp512_median"] is None
    assert invalid["tg128_median"] is None


def test_depth_plus_sample_fits_plan_window():
    assert 190464 + PP + TG < WINDOW
    row = row_from_timings(190464, _timings(cache_n=190464), window=WINDOW)
    assert row["valid"] is True
    assert math.isfinite(row["pp512"])


if __name__ == "__main__":
    test_constants_match_the_plan()
    test_valid_row_requires_exact_cache_and_counts()
    test_wrong_counts_and_cached_prefill_off_by_one_are_invalid()
    test_fitted_window_is_enforced_not_module_default()
    test_missing_nonfinite_or_nonpositive_rates_are_invalid()
    test_truncation_and_cross_check_mismatch_are_invalid()
    test_headline_requires_exactly_three_valid_reps()
    test_depth_plus_sample_fits_plan_window()
    print("ok")
