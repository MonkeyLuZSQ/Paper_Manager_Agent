from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_CONFIG_PATH = Path("config.yaml")


@dataclass(frozen=True)
class EmbeddingConfig:
    embedding_enabled: bool = True
    embedding_backend: str = "auto"
    embedding_model: str = "BAAI/bge-m3"
    embedding_hashing_model: str = "local-hashing-multilingual-v1"
    embedding_fallback_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    embedding_device: str = "cpu"
    embedding_batch_size: int = 8
    embedding_normalize: bool = True
    embedding_multilingual: bool = True
    embedding_hash_dim: int = 768

    @classmethod
    def from_file(cls, path: Path = DEFAULT_CONFIG_PATH) -> "EmbeddingConfig":
        values = _read_simple_yaml(path)
        return cls(
            embedding_enabled=_as_bool(values.get("embedding_enabled"), cls.embedding_enabled),
            embedding_backend=str(values.get("embedding_backend") or cls.embedding_backend),
            embedding_model=str(values.get("embedding_model") or cls.embedding_model),
            embedding_hashing_model=str(values.get("embedding_hashing_model") or cls.embedding_hashing_model),
            embedding_fallback_model=str(values.get("embedding_fallback_model") or cls.embedding_fallback_model),
            embedding_device=str(values.get("embedding_device") or cls.embedding_device),
            embedding_batch_size=int(values.get("embedding_batch_size") or cls.embedding_batch_size),
            embedding_normalize=_as_bool(values.get("embedding_normalize"), cls.embedding_normalize),
            embedding_multilingual=_as_bool(values.get("embedding_multilingual"), cls.embedding_multilingual),
            embedding_hash_dim=int(values.get("embedding_hash_dim") or cls.embedding_hash_dim),
        )


class EmbeddingClient:
    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        self.config = config or EmbeddingConfig.from_file()
        self.model_name = self.config.embedding_model
        self.backend = "sentence_transformers"
        self._model: Any | None = None

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not self.config.embedding_enabled:
            raise RuntimeError("Embedding is disabled in config.yaml.")
        if self._use_hashing_backend():
            return [_hashing_embedding(text, self.config.embedding_hash_dim) for text in texts]

        model = self._load_model()
        vectors = model.encode(
            texts,
            batch_size=self.config.embedding_batch_size,
            normalize_embeddings=self.config.embedding_normalize,
            show_progress_bar=False,
        )
        return [list(map(float, vector)) for vector in vectors]

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model

        try:
            from sentence_transformers import SentenceTransformer
        except ModuleNotFoundError:
            if self.config.embedding_backend == "sentence_transformers":
                raise
            self._switch_to_hashing("sentence-transformers is not installed")
            return None

        try:
            self._model = SentenceTransformer(self.config.embedding_model, device=self.config.embedding_device)
        except Exception as primary_exc:
            self.model_name = self.config.embedding_fallback_model
            try:
                self._model = SentenceTransformer(self.config.embedding_fallback_model, device=self.config.embedding_device)
            except Exception:
                if self.config.embedding_backend == "sentence_transformers":
                    raise primary_exc
                self._switch_to_hashing(f"sentence-transformers model load failed: {primary_exc}")
                return None
        return self._model

    def _use_hashing_backend(self) -> bool:
        if self.config.embedding_backend == "hashing":
            self._switch_to_hashing("configured hashing backend")
            return True
        if self.backend == "hashing":
            return True
        self._load_model()
        return self.backend == "hashing"

    def _switch_to_hashing(self, reason: str) -> None:
        self.backend = "hashing"
        self.model_name = self.config.embedding_hashing_model
        print(f"Embedding backend fallback: {reason}; using {self.model_name}.")


def embed_text(text: str) -> list[float]:
    return EmbeddingClient().embed_text(text)


def embed_texts(texts: list[str]) -> list[list[float]]:
    return EmbeddingClient().embed_texts(texts)


def _read_simple_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    values: dict[str, Any] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = _parse_scalar(value.strip())
    return values


def _parse_scalar(value: str) -> Any:
    cleaned = value.strip().strip('"').strip("'")
    lowered = cleaned.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(cleaned)
    except ValueError:
        return cleaned


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "on"}


def _hashing_embedding(text: str, dim: int) -> list[float]:
    vector = [0.0] * dim
    lowered = text.lower()
    features = _hashing_features(lowered)
    for feature in features:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "little", signed=False)
        index = value % dim
        sign = 1.0 if (value >> 63) == 0 else -1.0
        vector[index] += sign
    norm = math.sqrt(sum(item * item for item in vector)) or 1.0
    return [item / norm for item in vector]


def _hashing_features(text: str) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9_\-]+|[\u4e00-\u9fff]", text)
    features = list(words)
    joined = " ".join(words)
    for ngram in (2, 3):
        for index in range(max(0, len(words) - ngram + 1)):
            features.append("_".join(words[index : index + ngram]))
    for index in range(max(0, len(joined) - 3)):
        features.append(joined[index : index + 4])
    return features
