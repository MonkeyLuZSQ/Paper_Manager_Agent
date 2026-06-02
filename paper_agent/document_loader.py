from __future__ import annotations

from pathlib import Path


SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md"}


def load_document(path: Path) -> str:
    """Load a paper from PDF, TXT, or Markdown."""
    if not path.exists():
        raise FileNotFoundError(f"Paper not found: {path}")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_SUFFIXES))
        raise ValueError(f"Unsupported file type '{suffix}'. Supported: {supported}")

    if suffix == ".pdf":
        return _load_pdf(path)

    return path.read_text(encoding="utf-8", errors="ignore")


def _load_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "pypdf is required to read PDF files. Install dependencies with: "
            "pip install -r requirements.txt"
        ) from exc

    reader = PdfReader(str(path))
    pages: list[str] = []

    for index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(f"[Page {index}]\n{text.strip()}")

    if not pages:
        raise ValueError(
            "No extractable text was found in this PDF. "
            "If it is scanned, run OCR first and save it as text or searchable PDF."
        )

    return "\n\n".join(pages)
