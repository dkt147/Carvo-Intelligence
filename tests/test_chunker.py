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
