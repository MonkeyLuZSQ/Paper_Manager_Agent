from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from paper_agent.paper_store import PaperChunk, build_index, list_supported_papers, load_or_build_index
from paper_agent.retriever import RetrievedChunk, format_retrieved_chunks, search_chunks
from paper_agent.reviewer import resolve_paper_path


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
        results = search_chunks(chunks, query=query, top_k=top_k)
        if not results and _is_overview_query(query):
            priority = {"Abstract": 0, "Introduction": 1, "Conclusion": 2}
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
