from app.ingestion.chunker import chunk_document

SAMPLE = """כותרת המסמך

פסקת פתיחה קצרה.

ראשון

פסקה ראשונה של העלייה הראשונה.

פסקה שנייה של העלייה הראשונה.

שני

פסקה של העלייה השנייה.

לסיכום

שורת סיכום.
"""


def test_sections_are_labelled_from_headings():
    chunks = chunk_document(SAMPLE, max_chars=200, overlap_chars=20)
    sections = {c.section for c in chunks}
    assert {"intro", "aliyah-1", "aliyah-2", "summary"} <= sections


def test_chunk_indices_are_sequential_and_text_nonempty():
    chunks = chunk_document(SAMPLE, max_chars=200, overlap_chars=20)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert all(c.text.strip() for c in chunks)


def test_long_section_is_split_and_respects_max_chars():
    long_text = "ראשון\n\n" + ("מילה " * 500)
    chunks = chunk_document(long_text, max_chars=300, overlap_chars=40)
    assert len(chunks) > 1
    assert all(len(c.text) <= 300 for c in chunks)
    assert all(c.section == "aliyah-1" for c in chunks)


def test_overlap_must_be_less_than_max():
    import pytest

    with pytest.raises(ValueError, match="overlap_chars must be less than max_chars"):
        chunk_document("ראשון\n\ntext", max_chars=100, overlap_chars=100)
    with pytest.raises(ValueError, match="overlap_chars must be less than max_chars"):
        chunk_document("ראשון\n\ntext", max_chars=50, overlap_chars=50)


def test_split_keeps_hebrew_words_intact():
    text = "ראשון\n\n" + " ".join(["שלום"] * 40)
    chunks = chunk_document(text, max_chars=30, overlap_chars=8)
    assert len(chunks) > 1
    for chunk in chunks:
        words = chunk.text.split()
        assert words
        assert all(word == "שלום" for word in words)


def test_combining_mark_stays_with_base_letter():
    cluster = "e\u0301"
    text = "ראשון\n\n" + ("x" * 8) + cluster + ("y" * 12)
    chunks = chunk_document(text, max_chars=9, overlap_chars=2)
    joined = "".join(c.text for c in chunks)
    assert cluster in joined
    for chunk in chunks:
        assert not chunk.text.startswith("\u0301")
        if chunk.text.endswith("e"):
            assert chunk.text.endswith(cluster)


def test_unbreakable_token_still_advances():
    text = "ראשון\n\n" + ("א" * 80)
    chunks = chunk_document(text, max_chars=25, overlap_chars=5)
    assert len(chunks) > 1
    assert all(c.text for c in chunks)
    assert sum(len(c.text) for c in chunks) >= 80
