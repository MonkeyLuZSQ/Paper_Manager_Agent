from __future__ import annotations


def normalize_text(text: str) -> str:
    lines = [line.strip() for line in text.replace("\r\n", "\n").split("\n")]
    cleaned_lines: list[str] = []

    previous_blank = False
    for line in lines:
        is_blank = not line
        if is_blank and previous_blank:
            continue
        cleaned_lines.append(line)
        previous_blank = is_blank

    return "\n".join(cleaned_lines).strip()


def chunk_text(text: str, max_chars: int = 12000, overlap: int = 800) -> list[str]:
    """Split long papers into overlapping chunks for LLM processing."""
    if max_chars <= overlap:
        raise ValueError("max_chars must be larger than overlap")

    text = normalize_text(text)
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunk = text[start:end]

        if end < len(text):
            split_at = max(chunk.rfind("\n\n"), chunk.rfind(". "), chunk.rfind("。"))
            if split_at > max_chars * 0.6:
                end = start + split_at + 1
                chunk = text[start:end]

        chunks.append(chunk.strip())
        if end >= len(text):
            break
        start = max(0, end - overlap)

    return chunks


# ---------------------------------------------------------------------------
# Section-aware chunking
# ---------------------------------------------------------------------------

_SECTION_HEADERS = [
    "abstract",
    "1. introduction",
    "introduction",
    "2. method",
    "2. methods",
    "2. methodology",
    "2. formulation",
    "2. model",
    "2. approach",
    "2. background",
    "2. related work",
    "method",
    "methods",
    "methodology",
    "formulation",
    "model description",
    "mathematical formulation",
    "governing equations",
    "3. algorithm",
    "3. numerical method",
    "3. numerical scheme",
    "3. implementation",
    "algorithm",
    "numerical method",
    "numerical scheme",
    "numerical algorithm",
    "4. numerical examples",
    "4. experiments",
    "4. results",
    "4. numerical results",
    "5. numerical examples",
    "5. experiments",
    "5. results",
    "5. numerical results",
    "numerical examples",
    "numerical experiments",
    "numerical results",
    "experiments",
    "results and discussion",
    "results",
    "6. conclusion",
    "7. conclusion",
    "conclusion",
    "conclusions",
    "concluding remarks",
    "discussion",
    "references",
]


def detect_section_headers(text: str) -> list[tuple[int, str]]:
    """Return (char_offset, header_title) for each section header found."""
    boundaries: list[tuple[int, str]] = []
    lines = text.split("\n")
    offset = 0
    for line in lines:
        stripped = line.strip().lower()
        if not stripped:
            offset += len(line) + 1
            continue
        for header in _SECTION_HEADERS:
            if stripped == header or stripped.startswith(header + " ") or stripped.startswith(header + "."):
                if not boundaries or offset != boundaries[-1][0]:
                    boundaries.append((offset, line.strip()))
                break
        offset += len(line) + 1
    return boundaries


def chunk_text_section_aware(
    text: str,
    max_chars: int = 1800,
    overlap: int = 180,
) -> list[str]:
    """Split text into chunks that respect section boundaries.

    Sections are detected by common academic paper headers.  Each section is
    chunked independently so that no chunk spans two different sections.
    Sections shorter than *max_chars* are kept as a single chunk.
    """
    text = normalize_text(text)
    if len(text) <= max_chars:
        return [text]

    boundaries = detect_section_headers(text)
    if not boundaries:
        return chunk_text(text, max_chars=max_chars, overlap=overlap)

    # Build section texts
    sections: list[str] = []
    for i, (start, _header) in enumerate(boundaries):
        end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(text)
        section_text = text[start:end].strip()
        if section_text:
            sections.append(section_text)

    # Pre-text before first detected header
    if boundaries and boundaries[0][0] > 200:
        sections.insert(0, text[: boundaries[0][0]].strip())

    if not sections:
        return chunk_text(text, max_chars=max_chars, overlap=overlap)

    # Chunk each section independently
    chunks: list[str] = []
    for section in sections:
        if len(section) <= max_chars:
            chunks.append(section)
        else:
            section_chunks = chunk_text(section, max_chars=max_chars, overlap=overlap)
            chunks.extend(section_chunks)

    return chunks if chunks else [text]
