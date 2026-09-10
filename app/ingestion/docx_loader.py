"""Minimal .docx text extraction using the standard library only.

A .docx is a zip; the body text lives in word/document.xml. We turn paragraph
and line-break tags into newlines and strip the rest of the markup. Good enough
to feed a chunker; not a full-fidelity converter.
"""

import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

_PARAGRAPH_BREAK = re.compile(r"</w:p>")
_LINE_BREAK = re.compile(r"<w:br\b[^>]*/>")
_TAG = re.compile(r"<[^>]+>")
_WS_RUN = re.compile(r"[ \t]{2,}")


@dataclass(frozen=True)
class LoadedDocument:
    path: Path
    book: str          # parent folder name (e.g. "שמות")
    parashah: str      # file stem
    text: str


def extract_text(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        raw = archive.read("word/document.xml").decode("utf-8", "ignore")

    raw = _LINE_BREAK.sub("\n", raw)
    raw = _PARAGRAPH_BREAK.sub("\n", raw)
    raw = _TAG.sub("", raw)
    raw = raw.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")

    lines = [_WS_RUN.sub(" ", line).strip() for line in raw.splitlines()]
    # collapse 3+ blank lines to a single blank line
    out: list[str] = []
    blank = 0
    for line in lines:
        if line:
            blank = 0
            out.append(line)
        else:
            blank += 1
            if blank == 1:
                out.append("")
    return "\n".join(out).strip()


def iter_documents(root: str | Path) -> Iterator[LoadedDocument]:
    root_path = Path(root)
    for path in sorted(root_path.rglob("*.docx")):
        if path.name.startswith("~$"):
            continue
        try:
            text = extract_text(path)
        except (KeyError, zipfile.BadZipFile):
            continue
        if not text:
            continue
        rel_parent = path.parent.name if path.parent != root_path else ""
        yield LoadedDocument(
            path=path,
            book=rel_parent,
            parashah=path.stem,
            text=text,
        )
