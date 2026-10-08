import zipfile
from pathlib import Path

import pytest

from app.ingestion.docx_loader import extract_text, iter_documents

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def _docx(path: Path, body_xml: str) -> Path:
    document = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="{W}">
  <w:body>
    {body_xml}
  </w:body>
</w:document>'''
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", document)
        archive.writestr(
            "[Content_Types].xml",
            "<Types xmlns='http://schemas.openxmlformats.org/package/2006/content-types'></Types>",
        )
    return path


def test_extracts_paragraphs_tabs_and_skips_deletes_and_field_codes(tmp_path):
    path = _docx(
        tmp_path / "sample.docx",
        """
    <w:p>
      <w:r><w:t>Hello</w:t></w:r>
      <w:r><w:tab/></w:r>
      <w:r><w:t>World</w:t></w:r>
    </w:p>
    <w:p>
      <w:del>
        <w:r><w:delText>deleted</w:delText></w:r>
      </w:del>
      <w:r><w:t>kept</w:t></w:r>
    </w:p>
    <w:p>
      <w:r><w:instrText>PAGE</w:instrText></w:r>
      <w:r><w:t>12</w:t></w:r>
    </w:p>
    """,
    )
    text = extract_text(path)
    assert "Hello\tWorld" in text
    assert "kept" in text
    assert "deleted" not in text
    assert "PAGE" not in text
    assert "12" in text


def test_entities_are_not_double_decoded(tmp_path):
    path = _docx(
        tmp_path / "ents.docx",
        """<w:p><w:r><w:t>A &amp;lt; B &amp; C</w:t></w:r></w:p>""",
    )
    text = extract_text(path)
    assert "A &lt; B & C" in text
    assert "< B" not in text


def test_zip_entry_size_guard(tmp_path, monkeypatch):
    path = _docx(tmp_path / "big.docx", "<w:p><w:r><w:t>ok</w:t></w:r></w:p>")
    monkeypatch.setattr("app.ingestion.docx_loader.MAX_DOCX_XML_BYTES", 10)
    with pytest.raises(ValueError, match="exceeds"):
        extract_text(path)


def test_iter_documents_skips_temp_and_empty(tmp_path):
    _docx(tmp_path / "real.docx", "<w:p><w:r><w:t>visible</w:t></w:r></w:p>")
    _docx(tmp_path / "~$lock.docx", "<w:p><w:r><w:t>ignored</w:t></w:r></w:p>")
    docs = list(iter_documents(tmp_path))
    assert [d.parashah for d in docs] == ["real"]
    assert docs[0].text == "visible"
