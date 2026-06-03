from __future__ import annotations

import math
import re
from dataclasses import dataclass

from paper_agent.paper_store import PaperChunk


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: PaperChunk
    score: float


def count_tokens(text: str) -> int:
    return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))


def search_chunks(
    chunks: list[PaperChunk],
    query: str,
    top_k: int = 5,
    max_chunk_chars: int = 700,
) -> list[RetrievedChunk]:
    terms = _query_terms(query)
    if not terms:
        return []

    results: list[RetrievedChunk] = []
    total_docs = max(1, len(chunks))
    doc_freq = {
        term: sum(1 for chunk in chunks if term in _searchable_text(chunk))
        for term in terms
    }

    for chunk in chunks:
        text = _searchable_text(chunk)
        score = 0.0
        for term in terms:
            occurrences = text.count(term)
            if occurrences:
                idf = math.log((total_docs + 1) / (doc_freq[term] + 1)) + 1
                score += occurrences * idf
        if score:
            trimmed = chunk.text[:max_chunk_chars].strip()
            results.append(
                RetrievedChunk(
                    chunk=PaperChunk(
                        chunk_id=chunk.chunk_id,
                        paper_id=chunk.paper_id,
                        paper_name=chunk.paper_name,
                        section=chunk.section,
                        page=chunk.page,
                        text=trimmed,
                    ),
                    score=score,
                )
            )

    return sorted(results, key=lambda item: item.score, reverse=True)[:top_k]


def format_retrieved_chunks(results: list[RetrievedChunk]) -> str:
    if not results:
        return "当前没有检索到相关文献片段。"

    formatted: list[str] = []
    for index, result in enumerate(results, start=1):
        chunk = result.chunk
        page = f", page={chunk.page}" if chunk.page else ""
        formatted.append(
            f"[{index}] chunk_id={chunk.chunk_id}, paper={chunk.paper_name}, "
            f"section={chunk.section}{page}, score={result.score:.2f}\n"
            f"{chunk.text}"
        )
    return "\n\n".join(formatted)


def _query_terms(query: str) -> list[str]:
    raw_terms = re.findall(r"[A-Za-z][A-Za-z0-9_\-]+|[\u4e00-\u9fff]{2,}", query.lower())
    stop_words = {"the", "and", "for", "with", "this", "that", "paper", "what", "how"}
    return [term for term in raw_terms if term not in stop_words]


def _searchable_text(chunk: PaperChunk) -> str:
    return f"{chunk.paper_name} {chunk.section} {chunk.text}".lower()
