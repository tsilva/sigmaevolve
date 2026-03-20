from sigmaevolve.hashing import compute_script_hash, normalize_source


def test_source_normalization_is_stable():
    left = "print('x')\r\n"
    right = "print('x')\n\n"
    assert normalize_source(left) == "print('x')\n"
    assert normalize_source(left) == normalize_source(right)
    assert compute_script_hash(left) == compute_script_hash(right)
