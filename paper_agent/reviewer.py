from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from paper_agent.document_loader import load_document
from paper_agent.llm_client import VLLMClient
from paper_agent.prompts import (
    REVIEWER_SYSTEM_PROMPT,
    chunk_review_prompt,
    compact_notes_prompt,
    final_review_prompt,
)
from paper_agent.text_utils import chunk_text, normalize_text


REVIEW_CACHE_DIR = Path("data/review_cache")


@dataclass(frozen=True)
class ReviewResult:
    paper_path: Path
    chunks: int
    report: str
    mode: str = "quick"
    llm_calls: int = 0


@dataclass(frozen=True)
class SummaryModeConfig:
    max_review_chunks: int
    batch_size: int
    note_tokens: int
    compact_tokens: int
    final_tokens: int
    compression_rounds: int
    max_chunk_chars: int


SUMMARY_MODE_CONFIGS = {
    "quick": SummaryModeConfig(
        max_review_chunks=8,
        batch_size=3,
        note_tokens=220,
        compact_tokens=0,
        final_tokens=500,
        compression_rounds=0,
        max_chunk_chars=1100,
    ),
    "standard": SummaryModeConfig(
        max_review_chunks=12,
        batch_size=2,
        note_tokens=220,
        compact_tokens=220,
        final_tokens=500,
        compression_rounds=1,
        max_chunk_chars=1100,
    ),
    "deep": SummaryModeConfig(
        max_review_chunks=30,
        batch_size=1,
        note_tokens=240,
        compact_tokens=220,
        final_tokens=500,
        compression_rounds=99,
        max_chunk_chars=2500,
    ),
}


def compact_notes(
    llm: VLLMClient,
    notes: list[str],
    batch_size: int = 4,
    max_rounds: int = 99,
    max_tokens: int = 220,
) -> tuple[list[str], int]:
    llm_calls = 0
    round_index = 1
    while len(notes) > 2 and round_index <= max_rounds:
        compacted: list[str] = []
        batches = [notes[index : index + batch_size] for index in range(0, len(notes), batch_size)]
        total = len(batches)
        for index, batch in enumerate(batches, start=1):
            print(f"Compressing notes round {round_index}, batch {index}/{total}...")
            compacted.append(
                llm.chat(
                    REVIEWER_SYSTEM_PROMPT,
                    compact_notes_prompt(index, total, batch),
                    max_tokens=max_tokens,
                )
            )
            llm_calls += 1
        notes = compacted
        round_index += 1
    return notes, llm_calls


def resolve_paper_path(paper: str, paper_dir: Path) -> Path:
    """Resolve a user supplied paper name inside paper_dir."""
    direct = Path(paper)
    if direct.exists():
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
    summary_mode: str = "quick",
) -> ReviewResult:
    if summary_mode not in SUMMARY_MODE_CONFIGS:
        raise ValueError(f"Unknown summary_mode '{summary_mode}'. Use quick, standard, or deep.")

    config = SUMMARY_MODE_CONFIGS[summary_mode]
    raw_text = load_document(paper_path)
    text = normalize_text(raw_text)
    chunks = chunk_text(text, max_chars=max_chars, overlap=overlap)
    selected_chunks = select_review_chunks(chunks, config.max_review_chunks, config.max_chunk_chars)

    notes: list[str] = []
    llm_calls = 0
    cached_notes = load_cached_notes(paper_path, summary_mode)
    if cached_notes:
        print(f"Using cached review notes: {len(cached_notes)} note(s).")
        notes = cached_notes
    else:
        total = len(selected_chunks)
        batches = [
            selected_chunks[index : index + config.batch_size]
            for index in range(0, len(selected_chunks), config.batch_size)
        ]
        for index, batch in enumerate(batches, start=1):
            print(f"Reviewing chunk batch {index}/{len(batches)} ({summary_mode})...")
            joined_chunk = "\n\n".join(
                f"===== Selected chunk {chunk_index} =====\n{chunk}"
                for chunk_index, chunk in enumerate(batch, start=1)
            )
            notes.append(
                llm.chat(
                    REVIEWER_SYSTEM_PROMPT,
                    chunk_review_prompt(index, total, joined_chunk),
                    max_tokens=config.note_tokens,
                )
            )
            llm_calls += 1
        write_cached_notes(paper_path, summary_mode, notes)

    if config.compression_rounds > 0:
        notes, compact_calls = compact_notes(
            llm,
            notes,
            max_rounds=config.compression_rounds,
            max_tokens=config.compact_tokens,
        )
        llm_calls += compact_calls
    print("Writing final review report...")
    report = llm.chat(
        REVIEWER_SYSTEM_PROMPT,
        final_review_prompt(paper_path.stem, notes),
        max_tokens=min(llm.config.max_tokens, config.final_tokens),
    )
    llm_calls += 1
    return ReviewResult(
        paper_path=paper_path,
        chunks=len(selected_chunks),
        report=report,
        mode=summary_mode,
        llm_calls=llm_calls,
    )


def write_report(result: ReviewResult, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{result.paper_path.stem}总结.md"
    output_path.write_text(result.report, encoding="utf-8")
    return output_path


def select_review_chunks(chunks: list[str], max_review_chunks: int, max_chunk_chars: int) -> list[str]:
    scored = [(section_priority(chunk), index, chunk[:max_chunk_chars].strip()) for index, chunk in enumerate(chunks)]
    scored.sort(key=lambda item: (item[0], item[1]))

    selected: list[tuple[int, str]] = []
    seen_indexes: set[int] = set()
    for _priority, index, chunk in scored:
        if index in seen_indexes or not chunk:
            continue
        selected.append((index, chunk))
        seen_indexes.add(index)
        if len(selected) >= max_review_chunks:
            break

    selected.sort(key=lambda item: item[0])
    return [chunk for _index, chunk in selected]


def section_priority(chunk: str) -> int:
    lowered = chunk[:1500].lower()
    priorities = [
        (0, ["abstract"]),
        (1, ["introduction"]),
        (2, ["method", "algorithm", "formulation", "scheme", "approach"]),
        (3, ["result", "experiment", "numerical", "example", "simulation"]),
        (4, ["conclusion"]),
    ]
    for priority, keywords in priorities:
        if any(keyword in lowered for keyword in keywords):
            return priority
    return 9


def cache_path_for(paper_path: Path, summary_mode: str) -> Path:
    safe_name = "".join(char if char.isalnum() or char in "-_." else "_" for char in paper_path.stem)
    return REVIEW_CACHE_DIR / safe_name / f"{summary_mode}_notes.json"


def load_cached_notes(paper_path: Path, summary_mode: str) -> list[str]:
    cache_path = cache_path_for(paper_path, summary_mode)
    if not cache_path.exists():
        return []
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    notes = data.get("notes", [])
    return notes if isinstance(notes, list) else []


def write_cached_notes(paper_path: Path, summary_mode: str, notes: list[str]) -> None:
    cache_path = cache_path_for(paper_path, summary_mode)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps({"paper": paper_path.name, "mode": summary_mode, "notes": notes}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
