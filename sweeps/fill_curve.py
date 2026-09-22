import statistics

CHECKPOINTS = (0, 32768, 65536, 131072, 190464)
WINDOW = 262144
PP = 512
TG = 128
REPS = 3


def row_from_timings(depth, timings):
    cache_n = int(timings["cache_n"])
    prompt_n = int(timings["prompt_n"])
    predicted_n = int(timings["predicted_n"])
    valid = cache_n >= depth and prompt_n == PP and predicted_n == TG and depth + PP + TG <= WINDOW
    return {
        "depth": depth,
        "valid": valid,
        "pp512": float(timings["prompt_per_second"]) if valid else None,
        "tg128": float(timings["predicted_per_second"]) if valid else None,
        "cache_n": cache_n,
        "prompt_n": prompt_n,
        "predicted_n": predicted_n,
    }


def median(values):
    return statistics.median(values)
