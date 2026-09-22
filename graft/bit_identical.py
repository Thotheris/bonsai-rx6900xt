import hashlib


def compare(left_ids, right_ids, left_text, right_text):
    left_ids = list(left_ids)
    right_ids = list(right_ids)
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
