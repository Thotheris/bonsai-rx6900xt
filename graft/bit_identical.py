"""Strict comparison helpers for llama-server completion receipts."""

import hashlib
import math
from collections.abc import Mapping, Sequence

_TIMING_COUNTS = ("cache_n", "prompt_n", "predicted_n")
_TIMING_RATES = ("prompt_per_second", "predicted_per_second")


def _token_ids(value, name):
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise ValueError(f"{name} tokens must be a non-empty list of integer ids")
    result = list(value)
    if any(isinstance(token, bool) or not isinstance(token, int) for token in result):
        raise ValueError(f"{name} tokens must contain only integer ids")
    return result


def _text(value, name):
    if not isinstance(value, str):
        raise ValueError(f"{name} content must be a UTF-8 string")
    return value


def _finite_number(value, name, *, minimum=0.0):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    if number < minimum:
        raise ValueError(f"{name} must be >= {minimum:g}")
    return number


def validate_completion(response):
    """Validate and normalize a non-streaming ``/completion`` response.

    Token receipts must be present and non-empty. Timing values are retained as
    evidence even though identity is decided solely by exact token ids and the
    SHA-256 hashes of the UTF-8 content strings.
    """
    if not isinstance(response, Mapping):
        raise ValueError("completion response must be an object")
    if "tokens" not in response:
        raise ValueError("completion response is missing tokens")
    if "content" not in response:
        raise ValueError("completion response is missing content")
    if "timings" not in response or not isinstance(response["timings"], Mapping):
        raise ValueError("completion response is missing timings")

    tokens = _token_ids(response["tokens"], "completion")
    content = _text(response["content"], "completion")
    timings = dict(response["timings"])

    for key in _TIMING_COUNTS:
        value = _finite_number(timings.get(key), f"timings.{key}")
        if not value.is_integer():
            raise ValueError(f"timings.{key} must be an integer count")
        timings[key] = int(value)
    for key in _TIMING_RATES:
        timings[key] = _finite_number(timings.get(key), f"timings.{key}")

    if timings["predicted_n"] != len(tokens):
        raise ValueError("timings.predicted_n must equal the token receipt length")

    normalized = dict(response)
    normalized["tokens"] = tokens
    normalized["content"] = content
    normalized["timings"] = timings
    return normalized


def compare(left_ids, right_ids, left_text, right_text):
    """Compare exact token ids and SHA-256 of exact UTF-8 content."""
    left_ids = _token_ids(left_ids, "left")
    right_ids = _token_ids(right_ids, "right")
    left_text = _text(left_text, "left")
    right_text = _text(right_text, "right")

    mismatch = None
    for index, (left, right) in enumerate(zip(left_ids, right_ids)):
        if left != right:
            mismatch = index
            break
    if mismatch is None and len(left_ids) != len(right_ids):
        mismatch = min(len(left_ids), len(right_ids))

    left_sha = hashlib.sha256(left_text.encode("utf-8")).hexdigest()
    right_sha = hashlib.sha256(right_text.encode("utf-8")).hexdigest()
    return {
        "identical": mismatch is None and left_sha == right_sha,
        "first_token_mismatch": mismatch,
        "left_sha256": left_sha,
        "right_sha256": right_sha,
        "left_n": len(left_ids),
        "right_n": len(right_ids),
    }


def _provenance_key(receipt, side):
    try:
        model = receipt["provenance"]["model"]
        path = str(model["path"])
        sha256 = str(model["sha256"]).lower()
    except (KeyError, TypeError) as exc:
        raise ValueError(f"{side} receipt is missing model provenance") from exc
    if (
        not path
        or len(sha256) != 64
        or any(char not in "0123456789abcdef" for char in sha256)
    ):
        raise ValueError(f"{side} receipt has invalid model provenance")
    return path.casefold(), sha256


def compare_receipts(left, right):
    """Compare two complete receipts, refusing cross-model comparisons."""
    if not isinstance(left, Mapping) or not isinstance(right, Mapping):
        raise ValueError("receipts must be objects")
    if _provenance_key(left, "left") != _provenance_key(right, "right"):
        raise ValueError("identity arms must use the same model file and SHA-256")
    try:
        left_request = left["request"]
        right_request = right["request"]
        left_prompt = left_request["prompt"]
        right_prompt = right_request["prompt"]
    except (KeyError, TypeError) as exc:
        raise ValueError("receipts must include the exact request prompt") from exc
    if left_prompt != right_prompt:
        raise ValueError("identity arms must use the same prompt")
    if left_request != right_request:
        raise ValueError("identity arms must use the same exact request protocol")

    left_response = validate_completion(left.get("response"))
    right_response = validate_completion(right.get("response"))
    result = compare(
        left_response["tokens"],
        right_response["tokens"],
        left_response["content"],
        right_response["content"],
    )
    result["model_sha256"] = left["provenance"]["model"]["sha256"].lower()
    return result
