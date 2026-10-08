"""Split a protocol document into retrieval chunks.

Two levels:
  1. Section split on the aliyah headings (ראשון .. שביעי) and לסיכום, so a chunk
     keeps its structural label.
  2. Within a section, pack whole paragraphs into ~`max_chars` windows with a
     small trailing overlap so a match near a boundary keeps its context.

Overflow splits on whitespace so Hebrew words stay intact. Combining marks stay
with their base character. overlap must be strictly less than max_chars so the
split loop always advances.
"""

import unicodedata
from dataclasses import dataclass

_SECTION_HEADINGS = {
    "ראשון": "aliyah-1",
    "שני": "aliyah-2",
    "שלישי": "aliyah-3",
    "רביעי": "aliyah-4",
    "חמישי": "aliyah-5",
    "שישי": "aliyah-6",
    "שביעי": "aliyah-7",
    "מפטיר": "maftir",
    "לסיכום": "summary",
}


@dataclass(frozen=True)
class Chunk:
    text: str
    section: str
    chunk_index: int


def _iter_sections(text: str) -> list[tuple[str, list[str]]]:
    sections: list[tuple[str, list[str]]] = [("intro", [])]
    for raw_line in text.splitlines():
        line = raw_line.strip()
        key = _SECTION_HEADINGS.get(line)
        if key is not None:
            sections.append((key, []))
            continue
        sections[-1][1].append(raw_line)
    return [(name, lines) for name, lines in sections if any(l.strip() for l in lines)]


def _paragraphs(lines: list[str]) -> list[str]:
    paragraphs: list[str] = []
    buf: list[str] = []
    for line in lines:
        if line.strip():
            buf.append(line.strip())
        elif buf:
            paragraphs.append(" ".join(buf))
            buf = []
    if buf:
        paragraphs.append(" ".join(buf))
    return paragraphs


def _is_combining(char: str) -> bool:
    return unicodedata.combining(char) != 0


def _cut_index(text: str, max_chars: int) -> int:
    if len(text) <= max_chars:
        return len(text)

    cut = max_chars
    # Do not split between a base character and its combining marks.
    if cut < len(text) and _is_combining(text[cut]):
        while cut > 0 and _is_combining(text[cut]):
            cut -= 1
        if cut == 0:
            cut = max_chars
            while cut < len(text) and _is_combining(text[cut]):
                cut += 1
            return cut

    window = text[:cut]
    space = window.rfind(" ")
    if space > 0:
        return space
    return max(cut, 1)


def _overlap_tail(text: str, overlap_chars: int) -> str:
    if overlap_chars <= 0 or not text:
        return ""
    if len(text) <= overlap_chars:
        return text
    start = len(text) - overlap_chars
    while start > 0 and _is_combining(text[start]):
        start -= 1
    tail = text[start:]
    space = tail.find(" ")
    if space >= 0:
        tail = tail[space + 1 :]
    while tail and _is_combining(tail[0]) and len(tail) < len(text):
        tail = text[-(len(tail) + 1) :]
    return tail.strip()


def chunk_document(
    text: str, *, max_chars: int = 1100, overlap_chars: int = 150
) -> list[Chunk]:
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be less than max_chars")
    if overlap_chars < 0:
        raise ValueError("overlap_chars must be >= 0")

    chunks: list[Chunk] = []
    index = 0

    for section, lines in _iter_sections(text):
        current = ""
        for para in _paragraphs(lines):
            if current and len(current) + len(para) + 1 > max_chars:
                chunks.append(Chunk(current.strip(), section, index))
                index += 1
                tail = _overlap_tail(current, overlap_chars)
                current = f"{tail} {para}".strip()
            else:
                current = f"{current} {para}".strip()

            while len(current) > max_chars:
                cut = _cut_index(current, max_chars)
                piece = current[:cut].rstrip()
                if not piece:
                    piece = current[: max(cut, 1)]
                    cut = len(piece)
                chunks.append(Chunk(piece, section, index))
                index += 1
                remainder = current[cut:].lstrip()
                if not remainder:
                    current = ""
                    break
                tail = _overlap_tail(piece, overlap_chars)
                nxt = f"{tail} {remainder}".strip()
                # Drop overlap if it would not shrink the window (prevents a loop).
                if len(nxt) >= len(current):
                    current = remainder
                else:
                    current = nxt

        if current.strip():
            chunks.append(Chunk(current.strip(), section, index))
            index += 1

    return chunks
