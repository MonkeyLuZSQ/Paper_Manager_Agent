from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from paper_agent.document_loader import SUPPORTED_SUFFIXES, load_document
from paper_agent.text_utils import chunk_text, normalize_text


DEFAULT_INDEX_PATH = Path("data/chunks/index.json")


@dataclass(frozen=True)
class PaperChunk:
    chunk_id: str
    paper_id: str
    paper_name: str
    section: str
    page: int | None
    text: str
    title: str = ""
    token_count: int = 0


def list_supported_papers(paper_dir: Path) -> list[Path]:
    if not paper_dir.exists():
        return []
    return sorted(
        path
        for path in paper_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )


def build_index(
    paper_dir: Path,
    index_path: Path = DEFAULT_INDEX_PATH,
    chunk_chars: int = 1800,
    overlap: int = 180,
) -> list[PaperChunk]:
    chunks: list[PaperChunk] = []
    for paper_path in list_supported_papers(paper_dir):
        raw_text = load_document(paper_path)
        text = normalize_text(raw_text)
        paper_id = _safe_id(paper_path.stem)
        paper_chunks = chunk_text(text, max_chars=chunk_chars, overlap=overlap)
        current_page: int | None = None
        for index, chunk in enumerate(paper_chunks, start=1):
            inferred_page = _infer_page(chunk)
            if inferred_page is not None:
                current_page = inferred_page
            chunks.append(
                PaperChunk(
                    chunk_id=f"{paper_id}_{index:04d}",
                    paper_id=paper_id,
                    paper_name=paper_path.name,
                    title=paper_path.stem,
                    section=_infer_section(chunk),
                    page=inferred_page or current_page,
                    text=chunk,
                    token_count=_count_tokens(chunk),
                )
            )

    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(
        json.dumps([asdict(chunk) for chunk in chunks], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _try_build_embedding_index(chunks)
    return chunks


def load_index(index_path: Path = DEFAULT_INDEX_PATH) -> list[PaperChunk]:
    if not index_path.exists():
        return []
    data = json.loads(index_path.read_text(encoding="utf-8"))
    return [_chunk_from_dict(item) for item in data]


def load_or_build_index(
    paper_dir: Path,
    index_path: Path = DEFAULT_INDEX_PATH,
    chunk_chars: int = 1800,
    overlap: int = 180,
) -> list[PaperChunk]:
    chunks = load_index(index_path)
    if chunks:
        return chunks
    return build_index(paper_dir, index_path, chunk_chars=chunk_chars, overlap=overlap)


def _safe_id(value: str) -> str:
    normalized = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "_", value).strip("_")
    return normalized[:80] or "paper"


def _infer_page(text: str) -> int | None:
    match = re.search(r"\[Page\s+(\d+)\]", text)
    return int(match.group(1)) if match else None


def _infer_section(text: str) -> str:
    lowered = text[:1200].lower()
    section_keywords = [
        ("abstract", "Abstract"),
        ("introduction", "Introduction"),
        ("method", "Method"),
        ("algorithm", "Algorithm"),
        ("formulation", "Formulation"),
        ("numerical", "Numerical examples"),
        ("experiment", "Experiments"),
        ("result", "Results"),
        ("conclusion", "Conclusion"),
        ("reference", "References"),
    ]
    for keyword, section in section_keywords:
        if keyword in lowered:
            return section
    return "Unknown"


def _chunk_from_dict(item: dict) -> PaperChunk:
    text = str(item.get("text") or "")
    return PaperChunk(
        chunk_id=str(item.get("chunk_id") or ""),
        paper_id=str(item.get("paper_id") or ""),
        paper_name=str(item.get("paper_name") or item.get("title") or ""),
        title=str(item.get("title") or Path(str(item.get("paper_name") or "")).stem),
        section=str(item.get("section") or "Unknown"),
        page=item.get("page"),
        text=text,
        token_count=int(item.get("token_count") or _count_tokens(text)),
    )


def _count_tokens(text: str) -> int:
    return len(re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+|[^\w\s]", text, flags=re.UNICODE))


def _try_build_embedding_index(chunks: list[PaperChunk]) -> None:
    try:
        from paper_agent.vector_retriever import ensure_chunk_embeddings

        ensure_chunk_embeddings(chunks)
    except Exception as exc:
        print(f"Embedding index skipped: {exc}")
