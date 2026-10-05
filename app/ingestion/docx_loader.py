"""Minimal .docx text extraction using the standard library only.

A .docx is a zip; the body text lives in word/document.xml. ElementTree walks
paragraphs and runs so entities decode once, tabs stay tabs, and tracked
deletions / field instructions are not treated as body text.
"""

import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
from xml.etree import ElementTree as ET

MAX_DOCX_XML_BYTES = 20 * 1024 * 1024
_WS_RUN = re.compile(r" {2,}")


@dataclass(frozen=True)
class LoadedDocument:
    path: Path
    book: str          # parent folder name (e.g. "שמות")
    parashah: str      # file stem
    text: str


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


_SKIP_SUBTREES = frozenset({"del", "delText", "instrText", "moveFrom"})


def _read_entry(archive: zipfile.ZipFile, name: str) -> str:
    info = archive.getinfo(name)
    if info.file_size > MAX_DOCX_XML_BYTES:
        raise ValueError(
            f"{info.filename} uncompressed size {info.file_size} exceeds "
            f"{MAX_DOCX_XML_BYTES} bytes"
        )
    total = 0
    chunks: list[bytes] = []
    with archive.open(info) as src:
        while True:
            block = src.read(65536)
            if not block:
                break
            total += len(block)
            if total > MAX_DOCX_XML_BYTES:
                raise ValueError(
                    f"{name} uncompressed size {total} exceeds "
                    f"{MAX_DOCX_XML_BYTES} bytes"
                )
            chunks.append(block)
    return b"".join(chunks).decode("utf-8", "ignore")


def _walk(elem: ET.Element, parts: list[str]) -> None:
    name = _local(elem.tag)
    if name in _SKIP_SUBTREES:
        return
    if name == "tab":
        parts.append("\t")
    elif name in {"br", "cr"}:
        parts.append("\n")
    elif name == "t" and elem.text:
        parts.append(elem.text)

    for child in list(elem):
        _walk(child, parts)

    if name == "p":
        parts.append("\n")


def extract_text(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        raw = _read_entry(archive, "word/document.xml")

    root = ET.fromstring(raw)
    parts: list[str] = []
    _walk(root, parts)
    text = "".join(parts)

    lines = [_WS_RUN.sub(" ", line).strip() for line in text.splitlines()]
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
        except (KeyError, zipfile.BadZipFile, ET.ParseError, ValueError, OSError):
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
