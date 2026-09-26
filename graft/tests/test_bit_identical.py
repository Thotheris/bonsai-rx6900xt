import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bit_identical import compare, compare_receipts, validate_completion


def test_equal_ids_and_text_are_identical():
    result = compare([1, 2, 3], [1, 2, 3], "abc", "abc")
    assert result["identical"] is True
    assert result["first_token_mismatch"] is None
    assert result["left_sha256"] == hashlib.sha256(b"abc").hexdigest()


def test_token_mismatch_reports_index():
    result = compare([1, 2, 3], [1, 9, 3], "abc", "abc")
    assert result["identical"] is False
    assert result["first_token_mismatch"] == 1


def test_text_mismatch_fails_even_if_ids_match():
    result = compare([1], [1], "abc", "abd")
    assert result["identical"] is False
    assert result["first_token_mismatch"] is None
    assert result["left_sha256"] != result["right_sha256"]


def test_length_mismatch_reports_shorter_length():
    result = compare([1, 2], [1, 2, 3], "ab", "abc")
    assert result["identical"] is False
    assert result["first_token_mismatch"] == 2


def test_empty_tokens_are_rejected():
    try:
        compare([], [], "", "")
    except ValueError as exc:
        assert "tokens" in str(exc)
    else:
        raise AssertionError("empty token receipts must fail")


def test_non_integer_tokens_are_rejected():
    try:
        compare(["1"], ["1"], "a", "a")
    except ValueError as exc:
        assert "integer" in str(exc)
    else:
        raise AssertionError("non-integer token ids must fail")


def _receipt(model_sha="a" * 64, tokens=None, content="abc"):
    tokens = [1, 2] if tokens is None else tokens
    return {
        "provenance": {
            "model": {"path": "model.gguf", "sha256": model_sha},
            "binary": {"path": "llama-server.exe", "sha256": "b" * 64},
        },
        "request": {"prompt": "hello", "return_tokens": True},
        "response": {
            "content": content,
            "tokens": tokens,
            "timings": {
                "cache_n": 0,
                "prompt_n": 1,
                "prompt_per_second": 10.0,
                "predicted_n": len(tokens),
                "predicted_per_second": 5.0,
            },
        },
    }


def test_completion_rejects_missing_and_nonfinite_timings():
    receipt = _receipt()
    del receipt["response"]["tokens"]
    try:
        validate_completion(receipt["response"])
    except ValueError as exc:
        assert "tokens" in str(exc)
    else:
        raise AssertionError("missing tokens must fail")

    receipt = _receipt()
    receipt["response"]["timings"]["predicted_per_second"] = float("nan")
    try:
        validate_completion(receipt["response"])
    except ValueError as exc:
        assert "finite" in str(exc)
    else:
        raise AssertionError("nonfinite timings must fail")


def test_receipts_must_use_same_model_file():
    try:
        compare_receipts(_receipt(), _receipt(model_sha="c" * 64))
    except ValueError as exc:
        assert "same model" in str(exc)
    else:
        raise AssertionError("different model files must not be compared")

    try:
        compare_receipts(_receipt(model_sha="z" * 64), _receipt(model_sha="z" * 64))
    except ValueError as exc:
        assert "provenance" in str(exc)
    else:
        raise AssertionError("non-hexadecimal model hashes must not be accepted")


def test_receipts_must_use_same_prompt_and_request_protocol():
    right = _receipt()
    right["request"]["n_predict"] = 64
    try:
        compare_receipts(_receipt(), right)
    except ValueError as exc:
        assert "request" in str(exc)
    else:
        raise AssertionError("different identity request protocols must not be compared")


def test_receipt_comparison_includes_utf8_hash_and_exact_ids():
    result = compare_receipts(_receipt(), _receipt(tokens=[1, 9]))
    assert result["identical"] is False
    assert result["first_token_mismatch"] == 1
    assert result["left_sha256"] == hashlib.sha256(b"abc").hexdigest()


if __name__ == "__main__":
    test_equal_ids_and_text_are_identical()
    test_token_mismatch_reports_index()
    test_text_mismatch_fails_even_if_ids_match()
    test_length_mismatch_reports_shorter_length()
    test_empty_tokens_are_rejected()
    test_non_integer_tokens_are_rejected()
    test_completion_rejects_missing_and_nonfinite_timings()
    test_receipts_must_use_same_model_file()
    test_receipt_comparison_includes_utf8_hash_and_exact_ids()
    print("ok")
