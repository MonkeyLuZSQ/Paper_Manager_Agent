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


def ensure_chunk_embeddings(
    chunks: list[PaperChunk],
    embeddings_path: Path = DEFAULT_EMBEDDING_PATH,
    meta_path: Path = DEFAULT_META_PATH,
    config: EmbeddingConfig | None = None,
) -> bool:
    config = config or EmbeddingConfig.from_file()
    if not config.embedding_enabled:
        print("embedding_enabled=false; embedding index skipped.")
        return False

    cached = _cache_matches(chunks, meta_path, config)
    print(f"embedding_model={config.embedding_model}")
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
    meta_path.write_text(
        json.dumps(
            {
                "embedding_model": client.model_name,
                "embedding_backend": client.backend,
                "configured_embedding_model": config.embedding_model,
                "configured_embedding_backend": config.embedding_backend,
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
    config = config or EmbeddingConfig.from_file()
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

    rewritten = rewrite_query(query)
    query_texts = _dedupe([rewritten.original_query, rewritten.english_query])
    client = EmbeddingClient(config)
    query_vectors = np.asarray(client.embed_texts(query_texts), dtype="float32")
    query_vectors = _l2_normalize(query_vectors)

    embeddings = _l2_normalize(np.load(embeddings_path))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta_chunks = meta.get("chunks", [])
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


def retrieve(
    query: str,
    chunks: list[PaperChunk],
    top_k: int = 8,
    mode: str = "hybrid",
) -> list[RetrievedChunk]:
    if mode == "keyword":
        return hybrid_search_chunks(chunks, query=query, top_k=top_k)

    embedding_results: list[RetrievedChunk] = []
    try:
        embedding_results = retrieve_by_embedding(query, top_k=max(top_k, 8), chunks=chunks)
    except Exception as exc:
        print(f"Embedding retrieval skipped: {exc}")

    rewritten = rewrite_query(query)
    keyword_results = hybrid_search_chunks(chunks, query=query, top_k=max(top_k, 8))
    for keyword_query in rewritten.keyword_queries:
        keyword_results.extend(hybrid_search_chunks(chunks, query=keyword_query, top_k=max(top_k, 8)))

    if mode == "embedding":
        return embedding_results[:top_k]
    if not embedding_results:
        return keyword_results[:top_k]

    return _merge_hybrid(embedding_results, keyword_results, top_k=top_k)


def _merge_hybrid(
    embedding_results: list[RetrievedChunk],
    keyword_results: list[RetrievedChunk],
    top_k: int,
) -> list[RetrievedChunk]:
    embedding_scores = _normalized_scores(embedding_results)
    keyword_scores = _normalized_scores(keyword_results)
    chunks_by_id = {item.chunk.chunk_id: item.chunk for item in [*embedding_results, *keyword_results]}
    merged: list[RetrievedChunk] = []
    for chunk_id, chunk in chunks_by_id.items():
        score = 0.65 * embedding_scores.get(chunk_id, 0.0) + 0.35 * keyword_scores.get(chunk_id, 0.0)
        merged.append(RetrievedChunk(chunk=chunk, score=score))
    merged.sort(key=lambda item: item.score, reverse=True)
    return _trim_results(merged[:top_k])


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
