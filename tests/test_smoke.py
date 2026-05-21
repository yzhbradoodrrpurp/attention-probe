import attention


def test_attention_entry_imports() -> None:
    assert callable(attention.main)
