from __future__ import annotations

import hashlib
import importlib.util
import json
import time
from dataclasses import asdict
from pathlib import Path

from paper_agent.embedding_client import EmbeddingClient, EmbeddingConfig
from paper_agent.paper_store import DEFAULT_INDEX_PATH, PaperChunk, load_index
from paper_agent.query_rewriter import detect_query_language, rewrite_query
from paper_agent.retriever import RetrievedChunk, hybrid_search_chunks


DEFAULT_EMBEDDING_DIR = Path("data/embeddings")
DEFAULT_EMBEDDING_PATH = DEFAULT_EMBEDDING_DIR / "chunk_embeddings.npy"
DEFAULT_META_PATH = DEFAULT_EMBEDDING_DIR / "chunk_meta.json"

_embedding_cache: dict[str, object] = {}


def _get_cached_embeddings(
    embeddings_path: Path,
    meta_path: Path,
) -> tuple[object, list[dict]] | None:
    """Return (normalized_embeddings, meta_chunks) from cache if file unchanged."""
    global _embedding_cache
    try:
        emb_mtime = embeddings_path.stat().st_mtime
        meta_mtime = meta_path.stat().st_mtime
    except OSError:
        return None
    cache_key = f"{embeddings_path}:{emb_mtime}:{meta_path}:{meta_mtime}"
    if _embedding_cache.get("key") == cache_key:
        return _embedding_cache["data"]
    try:
        import numpy as np
        embeddings = _l2_normalize(np.load(embeddings_path))
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta_chunks = meta.get("chunks", [])
        _embedding_cache = {"key": cache_key, "data": (embeddings, meta_chunks)}
        return _embedding_cache["data"]
    except Exception:
        return None


def ensure_chunk_embeddings(
    chunks: list[PaperChunk],
    embeddings_path: Path = DEFAULT_EMBEDDING_PATH,
    meta_path: Path = DEFAULT_META_PATH,
    config: EmbeddingConfig | None = None,
) -> bool:
    base_config = config or EmbeddingConfig.from_file()
    config = base_config.for_indexing()
    if not config.embedding_enabled:
        print("embedding_enabled=false; embedding index skipped.")
        return False

    cached = _cache_matches(chunks, meta_path, config)
    print(f"embedding_model={config.embedding_model}")
    print(f"embedding_device={config.embedding_device}")
    print(f"indexed_chunk_count={len(chunks)}")
    print(f"embedding_cache_hit={str(cached).lower()}")
    if cached and embeddings_path.exists():
        return True

    if not chunks:
        return False

    client = EmbeddingClient(config)
    vectors = client.embed_texts([chunk.text for chunk in chunks])

    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Embedding cache requires numpy.") from exc

    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_path, np.asarray(vectors, dtype="float32"))
    _embedding_cache.clear()
    meta_path.write_text(
        json.dumps(
            {
                "embedding_model": client.model_name,
                "embedding_backend": client.backend,
                "embedding_device": config.embedding_device,
                "configured_embedding_model": config.embedding_model,
                "configured_embedding_backend": config.embedding_backend,
                "configured_embedding_device": base_config.embedding_device,
                "configured_embedding_index_device": base_config.embedding_index_device,
                "configured_embedding_query_device": base_config.embedding_query_device,
                "embedding_multilingual": config.embedding_multilingual,
                "chunks": [_chunk_meta(chunk) for chunk in chunks],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return True


def retrieve_by_embedding(
    query: str,
    top_k: int = 8,
    chunks: list[PaperChunk] | None = None,
    embeddings_path: Path = DEFAULT_EMBEDDING_PATH,
    meta_path: Path = DEFAULT_META_PATH,
    config: EmbeddingConfig | None = None,
) -> list[RetrievedChunk]:
    started = time.perf_counter()
    base_config = config or EmbeddingConfig.from_file()
    config = base_config.for_query()
    language = detect_query_language(query)
    print(f"query_language={language}")

    if not config.embedding_enabled or not embeddings_path.exists() or not meta_path.exists():
        print("embedding_search_time=0.000s")
        print("retrieved_chunks=0")
        return []

    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Vector retrieval requires numpy.") from exc

    embeddings_data = _get_cached_embeddings(embeddings_path, meta_path)
    if embeddings_data is None:
        print("embedding_search_time=0.000s")
        print("retrieved_chunks=0")
        return []
    embeddings, meta_chunks = embeddings_data

    rewritten = rewrite_query(query)
    query_texts = _dedupe([rewritten.original_query, rewritten.english_query])
    client = EmbeddingClient(config)
    print(f"query_embedding_device={config.embedding_device}")
    query_vectors = np.asarray(client.embed_texts(query_texts), dtype="float32")
    print(f"query_embedding_model={client.model_name}")
    query_vectors = _l2_normalize(query_vectors)
    if embeddings.shape[1] != query_vectors.shape[1]:
        print(
            "embedding_dimension_mismatch="
            f"chunks:{embeddings.shape[1]},query:{query_vectors.shape[1]}"
        )
        print("embedding_search_time=0.000s")
        print("retrieved_chunks=0")
        return []

    allowed = {chunk.chunk_id: chunk for chunk in (chunks or load_index(DEFAULT_INDEX_PATH))}
    scores = embeddings @ query_vectors.T
    best_scores = scores.max(axis=1)

    results: list[RetrievedChunk] = []
    for row_index, score in enumerate(best_scores):
        if row_index >= len(meta_chunks):
            continue
        chunk_id = str(meta_chunks[row_index].get("chunk_id") or "")
        chunk = allowed.get(chunk_id)
        if not chunk:
            continue
        results.append(RetrievedChunk(chunk=chunk, score=float(score)))

    results.sort(key=lambda item: item.score, reverse=True)
    elapsed = time.perf_counter() - started
    print(f"embedding_search_time={elapsed:.3f}s")
    print(f"retrieved_chunks={min(top_k, len(results))}")
    return _trim_results(results[:top_k])


_SECTION_PRIORITY: dict[str, int] = {
    "abstract": 3,
    "introduction": 2,
    "method": 5,
    "methods": 5,
    "methodology": 5,
    "formulation": 5,
    "model": 5,
    "algorithm": 5,
    "approach": 4,
    "framework": 5,
    "experiment": 4,
    "experiments": 4,
    "experimental": 4,
    "numerical": 4,
    "result": 5,
    "results": 5,
    "simulation": 4,
    "evaluation": 4,
    "analysis": 3,
    "discussion": 3,
    "conclusion": 2,
    "conclusions": 2,
    "related work": 2,
    "background": 2,
    "preliminary": 1,
    "preliminaries": 1,
    "implementation": 4,
    "performance": 4,
}

_STOP_WORDS: set[str] = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "above", "below", "between", "under", "again",
    "further", "then", "once", "here", "there", "when", "where", "why",
    "how", "all", "each", "every", "both", "few", "more", "most", "other",
    "some", "such", "no", "not", "only", "own", "same", "so", "than",
    "too", "very", "just", "because", "but", "and", "or", "if", "while",
    "about", "what", "which", "this", "that", "these", "those", "it", "its",
}


def _extract_query_terms(query: str) -> list[str]:
    """Extract meaningful terms from a query for keyword-level scoring."""
    import re
    tokens = re.findall(r"[a-zA-Z\u4e00-\u9fff]+(?:[-_][a-zA-Z]+)*", query.lower())
    return [t for t in tokens if t not in _STOP_WORDS and len(t) > 1]


def _section_score(section: str) -> float:
    """Return a priority score for a paper section (0.0–1.0)."""
    if not section:
        return 0.3
    lower = section.lower().strip()
    for key, priority in _SECTION_PRIORITY.items():
        if key in lower:
            return min(priority / 5.0, 1.0)
    return 0.3


def _keyword_overlap_score(query_terms: list[str], text: str) -> float:
    """Fraction of query terms that appear in the chunk text."""
    if not query_terms:
        return 0.0
    text_lower = text.lower()
    hits = sum(1 for term in query_terms if term in text_lower)
    return hits / len(query_terms)


def _exact_match_bonus(query_terms: list[str], text: str) -> float:
    """Extra bonus for exact multi-word phrase matches."""
    text_lower = text.lower()
    bonus = 0.0
    for i in range(len(query_terms)):
        for j in range(i + 2, min(i + 5, len(query_terms) + 1)):
            phrase = " ".join(query_terms[i:j])
            if phrase in text_lower:
                bonus += 0.15 * (j - i)
    return min(bonus, 0.5)


def _rerank_candidates(
    results: list[RetrievedChunk],
    query: str,
    top_k: int,
) -> list[RetrievedChunk]:
    """Rescore candidates with multiple relevance signals and return top_k."""
    if not results:
        return []

    query_terms = _extract_query_terms(query)

    emb_scores = _normalized_scores(results)
    emb_values = [item.score for item in results]
    emb_min, emb_max = min(emb_values), max(emb_values)
    emb_range = emb_max - emb_min if emb_max > emb_min else 1.0

    reranked: list[RetrievedChunk] = []
    for item in results:
        chunk = item.chunk

        emb_norm = (item.score - emb_min) / emb_range if emb_range > 0 else 0.5
        kw_score = _keyword_overlap_score(query_terms, chunk.text)
        sec_score = _section_score(chunk.section)
        exact_bonus = _exact_match_bonus(query_terms, chunk.text)

        final_score = (
            0.40 * emb_norm
            + 0.25 * kw_score
            + 0.15 * sec_score
            + 0.20 * min(exact_bonus / 0.5, 1.0)
        )

        reranked.append(RetrievedChunk(chunk=chunk, score=final_score))

    reranked.sort(key=lambda item: item.score, reverse=True)
    return _trim_results(reranked[:top_k])


def retrieve(
    query: str,
    chunks: list[PaperChunk],
    top_k: int = 8,
    mode: str = "hybrid",
) -> list[RetrievedChunk]:
    if mode == "keyword":
        return hybrid_search_chunks(chunks, query=query, top_k=top_k)

    recall_k = max(top_k * 3, 24)

    embedding_results: list[RetrievedChunk] = []
    try:
        embedding_results = retrieve_by_embedding(query, top_k=recall_k, chunks=chunks)
    except Exception as exc:
        print(f"Embedding retrieval skipped: {exc}")

    rewritten = rewrite_query(query)
    keyword_results = hybrid_search_chunks(chunks, query=query, top_k=recall_k)
    for keyword_query in rewritten.keyword_queries:
        keyword_results.extend(hybrid_search_chunks(chunks, query=keyword_query, top_k=max(recall_k // 2, 8)))

    if mode == "embedding":
        return _rerank_candidates(embedding_results, query, top_k=top_k)
    if not embedding_results:
        return _rerank_candidates(keyword_results, query, top_k=top_k)

    merged = _merge_hybrid_raw(embedding_results, keyword_results)
    return _rerank_candidates(merged, query, top_k=top_k)


def _merge_hybrid_raw(
    embedding_results: list[RetrievedChunk],
    keyword_results: list[RetrievedChunk],
) -> list[RetrievedChunk]:
    """Merge embedding and keyword results without trimming, for reranking."""
    embedding_scores = _normalized_scores(embedding_results)
    keyword_scores = _normalized_scores(keyword_results)
    chunks_by_id = {item.chunk.chunk_id: item.chunk for item in [*embedding_results, *keyword_results]}
    merged: list[RetrievedChunk] = []
    for chunk_id, chunk in chunks_by_id.items():
        score = 0.65 * embedding_scores.get(chunk_id, 0.0) + 0.35 * keyword_scores.get(chunk_id, 0.0)
        merged.append(RetrievedChunk(chunk=chunk, score=score))
    merged.sort(key=lambda item: item.score, reverse=True)
    return merged




def _normalized_scores(results: list[RetrievedChunk]) -> dict[str, float]:
    if not results:
        return {}
    values = [item.score for item in results]
    low = min(values)
    high = max(values)
    if high == low:
        return {item.chunk.chunk_id: 1.0 for item in results}
    return {item.chunk.chunk_id: (item.score - low) / (high - low) for item in results}


def _trim_results(results: list[RetrievedChunk], max_chars: int = 650) -> list[RetrievedChunk]:
    trimmed: list[RetrievedChunk] = []
    for item in results:
        chunk = item.chunk
        trimmed.append(
            RetrievedChunk(
                chunk=PaperChunk(
                    chunk_id=chunk.chunk_id,
                    paper_id=chunk.paper_id,
                    paper_name=chunk.paper_name,
                    section=chunk.section,
                    page=chunk.page,
                    text=chunk.text[:max_chars].strip(),
                    title=chunk.title,
                    token_count=chunk.token_count,
                ),
                score=item.score,
            )
        )
    return trimmed


def _cache_matches(chunks: list[PaperChunk], meta_path: Path, config: EmbeddingConfig) -> bool:
    if not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if meta.get("configured_embedding_model") != config.embedding_model:
        return False
    if meta.get("embedding_model") == config.embedding_hashing_model:
        if config.embedding_backend == "sentence_transformers":
            return False
        if config.embedding_backend == "auto" and _sentence_transformers_available():
            return False
    cached_chunks = meta.get("chunks", [])
    if len(cached_chunks) != len(chunks):
        return False
    return all(cached == _chunk_meta(chunk) for cached, chunk in zip(cached_chunks, chunks))


def _sentence_transformers_available() -> bool:
    return importlib.util.find_spec("sentence_transformers") is not None


def _chunk_meta(chunk: PaperChunk) -> dict[str, object]:
    data = asdict(chunk)
    data["text_hash"] = hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()
    data.pop("text", None)
    return data


def _l2_normalize(matrix):
    import numpy as np

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        key = item.strip().lower()
        if item.strip() and key not in seen:
            seen.add(key)
            output.append(item.strip())
    return output
