from hubert_separator.utils.model import fix_len_compatibility


def test_fix_len_compatibility() -> None:
    len = fix_len_compatibility(1499)
    assert len == 1500
