from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from paper_agent.paper_store import PaperChunk, build_index, list_supported_papers, load_or_build_index
from paper_agent.retriever import RetrievedChunk, format_retrieved_chunks
from paper_agent.reviewer import resolve_paper_path
from paper_agent.vector_retriever import retrieve


@dataclass
class ToolBox:
    paper_dir: Path
    index_path: Path
    chunks: list[PaperChunk]

    @classmethod
    def create(cls, paper_dir: Path, index_path: Path) -> "ToolBox":
        chunks = load_or_build_index(paper_dir, index_path)
        return cls(paper_dir=paper_dir, index_path=index_path, chunks=chunks)

    def list_papers(self) -> str:
        papers = list_supported_papers(self.paper_dir)
        if not papers:
            return f"No supported papers found in {self.paper_dir}."
        return "\n".join(f"- {path.name}" for path in papers)

    def rebuild_index(self) -> str:
        self.chunks = build_index(self.paper_dir, self.index_path)
        return f"Indexed {len(self.chunks)} chunks from {self.paper_dir}."

    def resolve_paper(self, paper: str) -> Path:
        return resolve_paper_path(paper, self.paper_dir)

    def chunks_for_paper(self, paper_name: str) -> list[PaperChunk]:
        return [chunk for chunk in self.chunks if chunk.paper_name == paper_name]

    def search_papers(self, query: str, top_k: int = 5, paper_name: str | None = None) -> str:
        chunks = self.chunks_for_paper(paper_name) if paper_name else self.chunks
        results = retrieve(query, chunks=chunks, top_k=top_k, mode="hybrid")
        if not results and _needs_fallback(query):
            priority = _fallback_priority(query)
            selected = sorted(
                chunks,
                key=lambda chunk: (priority.get(chunk.section, 9), chunk.chunk_id),
            )[:top_k]
            results = [
                RetrievedChunk(
                    chunk=PaperChunk(
                        chunk_id=chunk.chunk_id,
                        paper_id=chunk.paper_id,
                        paper_name=chunk.paper_name,
                        section=chunk.section,
                        page=chunk.page,
                        text=chunk.text[:700].strip(),
                    ),
                    score=0.1,
                )
                for chunk in selected
            ]
        return format_retrieved_chunks(results)

    def read_chunks(self, chunk_ids: list[str]) -> str:
        selected = [chunk for chunk in self.chunks if chunk.chunk_id in set(chunk_ids)]
        if not selected:
            return "No matching chunks were found."
        return format_retrieved_chunks([RetrievedChunk(chunk=chunk, score=1.0) for chunk in selected])


def _is_overview_query(query: str) -> bool:
    lowered = query.lower()
    keywords = [
        "summary",
        "summarize",
        "overview",
        "abstract",
        "introduction",
        "conclusion",
        "总结",
        "概括",
        "主要",
        "研究什么",
        "讲什么",
        "摘要",
    ]
    return any(keyword in lowered for keyword in keywords)


def _needs_fallback(query: str) -> bool:
    lowered = query.lower()
    keywords = [
        "summary",
        "summarize",
        "overview",
        "method",
        "algorithm",
        "parallel",
        "result",
        "总结",
        "概括",
        "主要",
        "研究什么",
        "讲什么",
        "摘要",
        "核心算法",
        "算法",
        "方法",
        "时空并行",
        "并行",
        "算例",
        "实验",
        "结果",
    ]
    return any(keyword in lowered for keyword in keywords)


def _fallback_priority(query: str) -> dict[str, int]:
    lowered = query.lower()
    if any(keyword in lowered for keyword in ["核心算法", "算法", "方法", "实现", "推导"]):
        return {"Method": 0, "Algorithm": 0, "Formulation": 0, "Introduction": 1, "Results": 2}
    if any(keyword in lowered for keyword in ["时空并行", "并行", "space-time", "parallel"]):
        return {"Method": 0, "Algorithm": 0, "Formulation": 0, "Introduction": 1, "Results": 2}
    if any(keyword in lowered for keyword in ["算例", "实验", "结果", "验证"]):
        return {"Numerical examples": 0, "Experiments": 0, "Results": 0, "Method": 1, "Conclusion": 2}
    return {"Abstract": 0, "Introduction": 1, "Conclusion": 2, "Method": 3, "Results": 4}
