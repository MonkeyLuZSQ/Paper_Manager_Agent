from __future__ import annotations

import math
import re
from dataclasses import dataclass

from paper_agent.paper_store import PaperChunk
from paper_agent.query_rewriter import RewrittenQuery, rewrite_query


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: PaperChunk
    score: float


def count_tokens(text: str) -> int:
    return len(re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+|[^\w\s]", text, flags=re.UNICODE))


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


def hybrid_search_chunks(
    chunks: list[PaperChunk],
    query: str,
    top_k: int = 5,
    max_chunk_chars: int = 650,
) -> list[RetrievedChunk]:
    rewritten = rewrite_query(query)
    weighted_queries: list[tuple[str, float]] = [
        (rewritten.original_query, 0.8),
        (rewritten.english_query, 1.2),
        (rewritten.academic_query, 1.1),
    ]
    weighted_queries.extend((keyword_query, 1.35) for keyword_query in rewritten.keyword_queries)

    merged: dict[str, RetrievedChunk] = {}
    for query_text, weight in weighted_queries:
        for result in search_chunks(chunks, query_text, top_k=max(top_k, 8), max_chunk_chars=max_chunk_chars):
            chunk = result.chunk
            section_bonus = section_hint_bonus(chunk.section, rewritten)
            score = result.score * weight + section_bonus
            existing = merged.get(chunk.chunk_id)
            if not existing or score > existing.score:
                merged[chunk.chunk_id] = RetrievedChunk(chunk=chunk, score=score)

    return sorted(merged.values(), key=lambda item: item.score, reverse=True)[:top_k]


def format_retrieved_chunks(results: list[RetrievedChunk]) -> str:
    if not results:
        return "当前没有检索到相关文献片段。"

    formatted: list[str] = []
    for index, result in enumerate(results, start=1):
        chunk = result.chunk
        page = f", page={chunk.page}" if chunk.page else ", page=unknown"
        formatted.append(
            f"[{index}] paper_id={chunk.paper_id}, title={chunk.title or chunk.paper_name}, "
            f"page={page.removeprefix(', page=')}, "
            f"section={chunk.section}, chunk_id={chunk.chunk_id}, score={result.score:.2f}\n"
            f"{chunk.text}"
        )
    return "\n\n".join(formatted)


def section_hint_bonus(section: str, rewritten: RewrittenQuery) -> float:
    section_lower = section.lower()
    for hint in rewritten.section_hints:
        if hint.lower() in section_lower or section_lower in hint.lower():
            return 4.0
    return 0.0


def _query_terms(query: str) -> list[str]:
    raw_terms = re.findall(r"[A-Za-z][A-Za-z0-9_\-]+|[\u4e00-\u9fff]{2,}", query.lower())
    stop_words = {"the", "and", "for", "with", "this", "that", "paper", "what", "how"}
    terms = [term for term in raw_terms if term not in stop_words]
    return terms + _expanded_terms(query)


def _expanded_terms(query: str) -> list[str]:
    lowered = query.lower()
    expansions = {
        ("核心算法", "算法", "方法", "怎么实现", "如何实现"): [
            "algorithm",
            "method",
            "scheme",
            "formulation",
            "ridc",
            "revisionist",
            "deferred",
            "correction",
        ],
        ("时空并行", "空间时间并行", "时间并行", "并行"): [
            "space-time",
            "space",
            "time",
            "parallel",
            "communicator",
            "mpi",
            "temporal",
            "spatial",
        ],
        ("主要内容", "讲什么", "研究什么", "摘要", "总结"): [
            "abstract",
            "introduction",
            "conclusion",
            "method",
            "result",
        ],
        ("算例", "实验", "结果", "验证"): [
            "numerical",
            "experiment",
            "result",
            "simulation",
            "example",
            "accuracy",
        ],
        ("公式", "推导", "方程", "模型"): [
            "equation",
            "model",
            "formulation",
            "navier",
            "stokes",
        ],
    }
    expanded: list[str] = []
    for triggers, terms in expansions.items():
        if any(trigger in lowered for trigger in triggers):
            expanded.extend(terms)
    return expanded


def _searchable_text(chunk: PaperChunk) -> str:
    return f"{chunk.paper_name} {chunk.section} {chunk.text}".lower()
