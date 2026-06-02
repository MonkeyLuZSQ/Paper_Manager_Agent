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
