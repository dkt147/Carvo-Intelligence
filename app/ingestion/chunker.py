"""Split a protocol document into retrieval chunks.

Two levels:
  1. Section split on the aliyah headings (ראשון .. שביעי) and לסיכום, so a chunk
     keeps its structural label.
  2. Within a section, pack whole paragraphs into ~`max_chars` windows with a
     small trailing overlap so a match near a boundary keeps its context.
"""

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


def chunk_document(
    text: str, *, max_chars: int = 1100, overlap_chars: int = 150
) -> list[Chunk]:
    chunks: list[Chunk] = []
    index = 0

    for section, lines in _iter_sections(text):
        current = ""
        for para in _paragraphs(lines):
            if current and len(current) + len(para) + 1 > max_chars:
                chunks.append(Chunk(current.strip(), section, index))
                index += 1
                tail = current[-overlap_chars:] if overlap_chars else ""
                current = f"{tail} {para}".strip()
            else:
                current = f"{current} {para}".strip()

            while len(current) > max_chars:
                chunks.append(Chunk(current[:max_chars].strip(), section, index))
                index += 1
                current = current[max_chars - overlap_chars :].strip()

        if current.strip():
            chunks.append(Chunk(current.strip(), section, index))
            index += 1

    return chunks
