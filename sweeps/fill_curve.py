"""Strict pp512/tg128 row validation and three-repetition aggregation."""

import math
import statistics

CHECKPOINTS = (0, 32768, 65536, 131072, 190464)
WINDOW = 262144
PP = 512
TG = 128
REPS = 3


def _count(value, name, errors):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        errors.append(f"{name} is missing or not numeric")
        return None
    number = float(value)
    if not math.isfinite(number) or not number.is_integer() or number < 0:
        errors.append(f"{name} must be a finite non-negative integer")
        return None
    return int(number)


def _rate(value, name, errors):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        errors.append(f"{name} is missing or not numeric")
        return None
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        errors.append(f"{name} must be finite and positive")
        return None
    return number


def row_from_timings(
    depth,
    timings,
    window=WINDOW,
    *,
    truncated=False,
    tokens_cached=None,
    tokens_evaluated=None,
):
    errors = []
    if isinstance(depth, bool) or not isinstance(depth, int) or depth < 0:
        raise ValueError("depth must be a non-negative integer")
    if isinstance(window, bool) or not isinstance(window, int) or window <= 0:
        raise ValueError("window must be a positive integer")
    if not isinstance(timings, dict):
        timings = {}
        errors.append("timings must be an object")

    cache_n = _count(timings.get("cache_n"), "cache_n", errors)
    prompt_n = _count(timings.get("prompt_n"), "prompt_n", errors)
    predicted_n = _count(timings.get("predicted_n"), "predicted_n", errors)
    pp512 = _rate(timings.get("prompt_per_second"), "prompt_per_second", errors)
    tg128 = _rate(timings.get("predicted_per_second"), "predicted_per_second", errors)

    if cache_n is not None and cache_n != depth:
        errors.append(f"cache_n must equal depth ({depth})")
    if prompt_n is not None and prompt_n != PP:
        errors.append(f"prompt_n must equal {PP}")
    if predicted_n is not None and predicted_n != TG:
        errors.append(f"predicted_n must equal {TG}")
    if depth + PP + TG > window:
        errors.append("depth plus pp512/tg128 sample exceeds fitted window")
    if truncated is not False:
        errors.append("completion was truncated")

    expected_prompt = depth + PP
    if tokens_cached is not None:
        observed = _count(tokens_cached, "tokens_cached", errors)
        # The final response is emitted after sampling TG tokens. server-context
        # retains generated tokens in the slot cache, so this field is not the
        # same quantity as timings.cache_n. It must contain the scored prompt
        # and may contain up to the generated tail.
        if observed is not None and not expected_prompt <= observed <= expected_prompt + TG:
            errors.append(
                f"tokens_cached must be between {expected_prompt} and {expected_prompt + TG}"
            )
    if tokens_evaluated is not None:
        observed = _count(tokens_evaluated, "tokens_evaluated", errors)
        if observed is not None and observed != expected_prompt:
            errors.append(f"tokens_evaluated must equal {expected_prompt}")

    valid = not errors
    return {
        "depth": depth,
        "valid": valid,
        "errors": errors,
        "pp512": pp512 if valid else None,
        "tg128": tg128 if valid else None,
        "cache_n": cache_n,
        "prompt_n": prompt_n,
        "predicted_n": predicted_n,
        "tokens_cached": tokens_cached,
        "tokens_evaluated": tokens_evaluated,
        "truncated": truncated,
    }


def median(values):
    values = list(values)
    if len(values) != REPS:
        raise ValueError(f"median requires exactly {REPS} values")
    checked = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("median values must be numeric")
        value = float(value)
        if not math.isfinite(value) or value <= 0:
            raise ValueError("median values must be finite and positive")
        checked.append(value)
    return float(statistics.median(checked))


def headline(rows):
    rows = list(rows)
    valid_rows = [row for row in rows if isinstance(row, dict) and row.get("valid") is True]
    if len(rows) != REPS or len(valid_rows) != REPS:
        return {
            "valid": False,
            "valid_reps": len(valid_rows),
            "pp512_median": None,
            "tg128_median": None,
        }
    return {
        "valid": True,
        "valid_reps": REPS,
        "pp512_median": median(row["pp512"] for row in valid_rows),
        "tg128_median": median(row["tg128"] for row in valid_rows),
    }
