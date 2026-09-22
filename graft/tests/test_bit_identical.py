import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bit_identical import compare


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


if __name__ == "__main__":
    test_equal_ids_and_text_are_identical()
    test_token_mismatch_reports_index()
    test_text_mismatch_fails_even_if_ids_match()
    test_length_mismatch_reports_shorter_length()
    print("ok")
