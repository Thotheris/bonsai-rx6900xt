import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fill_curve import row_from_timings, CHECKPOINTS, WINDOW

PP = 512
TG = 128


def test_constants_match_the_plan():
    assert CHECKPOINTS == (0, 32768, 65536, 131072, 190464)
    assert WINDOW == 262144


def test_valid_row():
    timings = {"cache_n": 32768, "prompt_n": 512, "prompt_per_second": 300.0,
               "predicted_n": 128, "predicted_per_second": 40.0}
    row = row_from_timings(32768, timings)
    assert row["valid"] is True
    assert row["pp512"] == 300.0
    assert row["tg128"] == 40.0


def test_wrong_prompt_n_is_invalid():
    timings = {"cache_n": 32768, "prompt_n": 513, "prompt_per_second": 300.0,
               "predicted_n": 128, "predicted_per_second": 40.0}
    assert row_from_timings(32768, timings)["valid"] is False


def test_cache_shorter_than_checkpoint_is_invalid():
    timings = {"cache_n": 100, "prompt_n": 512, "prompt_per_second": 300.0,
               "predicted_n": 128, "predicted_per_second": 40.0}
    assert row_from_timings(32768, timings)["valid"] is False


def test_depth_plus_sample_must_fit_window():
    assert 190464 + PP + TG < WINDOW
    assert row_from_timings(190464, {
        "cache_n": 190464, "prompt_n": PP, "prompt_per_second": 1.0,
        "predicted_n": TG, "predicted_per_second": 1.0,
    })["valid"] is True


if __name__ == "__main__":
    test_constants_match_the_plan()
    test_valid_row()
    test_wrong_prompt_n_is_invalid()
    test_cache_shorter_than_checkpoint_is_invalid()
    test_depth_plus_sample_must_fit_window()
    print("ok")
