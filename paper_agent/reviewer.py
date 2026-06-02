from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from paper_agent.document_loader import load_document
from paper_agent.llm_client import VLLMClient
from paper_agent.prompts import (
    REVIEWER_SYSTEM_PROMPT,
    chunk_review_prompt,
    final_review_prompt,
)
from paper_agent.text_utils import chunk_text, normalize_text


@dataclass(frozen=True)
class ReviewResult:
    paper_path: Path
    chunks: int
    report: str


def resolve_paper_path(paper: str, paper_dir: Path) -> Path:
    """Resolve a user supplied paper name inside paper_dir."""
    direct = Path(paper)
    if direct.is_absolute():
        return direct

    candidate = paper_dir / paper
    if candidate.exists():
        return candidate

    matches = [path for path in paper_dir.iterdir() if path.is_file() and path.name == paper]
    if matches:
        return matches[0]

    fuzzy_matches = [
        path
        for path in paper_dir.iterdir()
        if path.is_file() and paper.lower() in path.name.lower()
    ]
    if len(fuzzy_matches) == 1:
        return fuzzy_matches[0]
    if len(fuzzy_matches) > 1:
        names = "\n".join(f"- {path.name}" for path in fuzzy_matches)
        raise ValueError(f"Multiple papers match '{paper}':\n{names}")

    raise FileNotFoundError(f"Paper '{paper}' was not found in {paper_dir}")


def review_paper(
    paper_path: Path,
    llm: VLLMClient,
    max_chars: int = 12000,
    overlap: int = 800,
) -> ReviewResult:
    raw_text = load_document(paper_path)
    text = normalize_text(raw_text)
    chunks = chunk_text(text, max_chars=max_chars, overlap=overlap)

    notes: list[str] = []
    total = len(chunks)
    for index, chunk in enumerate(chunks, start=1):
        notes.append(
            llm.chat(
                REVIEWER_SYSTEM_PROMPT,
                chunk_review_prompt(index, total, chunk),
            )
        )

    report = llm.chat(
        REVIEWER_SYSTEM_PROMPT,
        final_review_prompt(paper_path.stem, notes),
    )
    return ReviewResult(paper_path=paper_path, chunks=total, report=report)


def write_report(result: ReviewResult, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{result.paper_path.stem}_review.md"
    output_path.write_text(result.report, encoding="utf-8")
    return output_path
