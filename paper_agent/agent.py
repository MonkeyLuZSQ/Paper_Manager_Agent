from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from paper_agent.llm_client import LLMConfig, VLLMClient
from paper_agent.retriever import count_tokens
from paper_agent.reviewer import review_paper, write_report
from paper_agent.tools import ToolBox


def load_agent_instructions(path: Path = Path("agent.md")) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore").strip()


AGENT_SYSTEM_PROMPT = """你是一个本地文献管理和论文阅读 agent。
你只能基于工具返回的文献片段和对话上下文回答，不要编造文献中没有的信息。
如果检索片段不足，请明确说“当前检索片段不足”。
回答时尽量引用片段编号，例如 [1]、[2]。

你可以选择的动作只有：
1. search_papers: 根据 query 检索文献 chunk。
2. read_chunks: 根据 chunk_ids 读取指定 chunk。
3. final_answer: 给出最终回答。

每次只输出一个 JSON 对象，不要输出 Markdown 代码块。
"""


@dataclass
class ConversationTurn:
    role: str
    content: str


@dataclass
class PaperAgent:
    llm: VLLMClient
    tools: ToolBox
    output_dir: Path = Path("outputs")
    max_input_tokens: int = 1500
    recent_turns: int = 4
    max_tool_calls: int = 3
    active_paper_name: str | None = None
    conversation_summary: str = ""
    messages: list[ConversationTurn] = field(default_factory=list)
    review_instructions: str = field(default_factory=load_agent_instructions)

    @classmethod
    def create(
        cls,
        config: LLMConfig,
        paper_dir: Path,
        index_path: Path,
        output_dir: Path = Path("outputs"),
        max_input_tokens: int = 1500,
    ) -> "PaperAgent":
        return cls(
            llm=VLLMClient(config),
            tools=ToolBox.create(paper_dir, index_path),
            output_dir=output_dir,
            max_input_tokens=max_input_tokens,
        )

    def run(self, user_input: str) -> str:
        summary_answer = self._handle_summary_request(user_input)
        if summary_answer:
            self._remember(user_input, summary_answer)
            return summary_answer

        observations: list[str] = []
        forced_context = self.tools.search_papers(user_input, top_k=2, paper_name=self.active_paper_name)
        observations.append(f"Initial retrieval:\n{forced_context}")

        for step in range(1, self.max_tool_calls + 1):
            prompt = self._build_react_prompt(user_input, observations, step)
            raw = self.llm.chat(AGENT_SYSTEM_PROMPT, prompt, max_tokens=160)
            action = self._parse_action(raw)

            if action["action"] == "search_papers":
                args = action.get("args", {})
                query = str(args.get("query") or user_input)
                top_k = min(3, int(args.get("top_k") or 3))
                observations.append(
                    f"search_papers({query!r}):\n"
                    f"{self.tools.search_papers(query, top_k=top_k, paper_name=self.active_paper_name)}"
                )
                continue

            if action["action"] == "read_chunks":
                args = action.get("args", {})
                chunk_ids = args.get("chunk_ids") or []
                if not isinstance(chunk_ids, list):
                    chunk_ids = []
                observations.append(f"read_chunks({chunk_ids!r}):\n{self.tools.read_chunks(chunk_ids)}")
                continue

            if action["action"] == "final_answer":
                answer = str(action.get("answer") or "").strip()
                if answer:
                    self._remember(user_input, answer)
                    return answer

            observations.append(f"Invalid action at step {step}. Raw model output:\n{raw}")

        answer = self._force_final_answer(user_input, observations)
        self._remember(user_input, answer)
        return answer

    def _build_react_prompt(self, user_input: str, observations: list[str], step: int) -> str:
        return f"""请根据当前问题和已有 observation 选择下一步动作。

【项目级 agent.md 规范】
{self.review_instructions or "暂无"}

输出格式只能是 JSON：
{{"action": "search_papers", "args": {{"query": "...", "top_k": 5}}}}
或
{{"action": "read_chunks", "args": {{"chunk_ids": ["..."]}}}}
或
{{"action": "final_answer", "answer": "..."}}

【对话摘要】
{self.conversation_summary or "暂无"}

【当前选中文献】
{self.active_paper_name or "暂无。用户尚未指定文献时，不要假定只讨论某一篇。"}

【最近对话】
{self._recent_messages_text()}

【当前问题】
{user_input}

【Observation】
{self._trim_text(chr(10).join(observations), self.max_input_tokens)}

这是第 {step}/{self.max_tool_calls} 次工具决策。若信息足够，请使用 final_answer。
"""

    def _force_final_answer(self, user_input: str, observations: list[str]) -> str:
        prompt = f"""请基于已有文献片段直接回答用户问题。

【项目级 agent.md 规范】
{self.review_instructions or "暂无"}

【对话摘要】
{self.conversation_summary or "暂无"}

【当前选中文献】
{self.active_paper_name or "暂无"}

【最近对话】
{self._recent_messages_text()}

【当前问题】
{user_input}

【检索与工具结果】
{self._trim_text(chr(10).join(observations), self.max_input_tokens)}

要求：
- 使用中文。
- 尽量引用片段编号。
- 如果证据不足，请明确说明“当前检索片段不足”。
"""
        return self.llm.chat(AGENT_SYSTEM_PROMPT, prompt, max_tokens=self.llm.config.max_tokens)

    def _handle_summary_request(self, user_input: str) -> str | None:
        if not _is_summary_request(user_input):
            return None

        paper_path = self._extract_requested_paper(user_input)
        if paper_path is None:
            return (
                "请告诉我要总结 `paper_rep/` 中的哪一篇文献。当前可用文献：\n"
                f"{self.tools.list_papers()}"
            )

        summary_mode = detect_summary_mode(user_input)
        if summary_mode == "deep":
            print("Deep review will be slower on local GPU.")

        print(f"Summarizing paper: {paper_path} ({summary_mode})")
        result = review_paper(
            paper_path,
            self.llm,
            max_chars=2500,
            overlap=250,
            summary_mode=summary_mode,
        )
        output_path = write_report(result, self.output_dir)
        self.active_paper_name = paper_path.name

        if not self.tools.chunks_for_paper(paper_path.name):
            self.tools.rebuild_index()

        return (
            f"已按专业审稿人视角完成总结，并写入：{output_path}\n"
            f"总结模式：{result.mode}；本次总结 LLM 调用次数：{result.llm_calls}\n"
            f"当前选中文献已切换为：{paper_path.name}\n"
            "你可以继续围绕这篇文献提问。"
        )

    def _extract_requested_paper(self, user_input: str) -> Path | None:
        papers = list(self.tools.paper_dir.iterdir()) if self.tools.paper_dir.exists() else []
        candidates = [path for path in papers if path.is_file()]
        normalized_input = user_input.lower()

        for path in candidates:
            if path.name.lower() in normalized_input or path.stem.lower() in normalized_input:
                return path

        cleaned = user_input
        for marker in ["总结", "概括", "阅读", "审稿", "summarize", "review", "paper_rep", "/", "\\"]:
            cleaned = cleaned.replace(marker, " ")
        cleaned = " ".join(cleaned.split()).strip()
        if not cleaned:
            return None

        try:
            return self.tools.resolve_paper(cleaned)
        except (FileNotFoundError, ValueError):
            matches = [
                path
                for path in candidates
                if cleaned.lower() in path.name.lower() or cleaned.lower() in path.stem.lower()
            ]
            if len(matches) == 1:
                return matches[0]

            terms = _paper_query_terms(cleaned)
            scored: list[tuple[int, Path]] = []
            for path in candidates:
                name = path.name.lower()
                score = sum(1 for term in terms if term in name)
                if score:
                    scored.append((score, path))
            scored.sort(key=lambda item: item[0], reverse=True)
            if scored and (len(scored) == 1 or scored[0][0] > scored[1][0]):
                return scored[0][1]
            return None

    def _remember(self, user_input: str, answer: str) -> None:
        self.messages.append(ConversationTurn("User", user_input))
        self.messages.append(ConversationTurn("Agent", answer))
        if len(self.messages) > self.recent_turns * 2:
            old_messages = self.messages[: -self.recent_turns * 2]
            self.messages = self.messages[-self.recent_turns * 2 :]
            old_text = "\n".join(f"{item.role}: {item.content}" for item in old_messages)
            self.conversation_summary = self._summarize_memory(old_text)

    def _summarize_memory(self, old_text: str) -> str:
        prompt = f"""请把旧对话压缩为短期记忆摘要，保留用户研究兴趣、已确认的论文信息、待办事项和约束。

【已有摘要】
{self.conversation_summary or "暂无"}

【旧对话】
{self._trim_text(old_text, 900)}
"""
        return self.llm.chat(AGENT_SYSTEM_PROMPT, prompt, max_tokens=220)

    def _recent_messages_text(self) -> str:
        if not self.messages:
            return "暂无"
        return "\n".join(f"{item.role}: {item.content}" for item in self.messages[-self.recent_turns * 2 :])

    def _trim_text(self, text: str, max_tokens: int) -> str:
        if count_tokens(text) <= max_tokens:
            return text
        chars = max(800, max_tokens * 3)
        return text[:chars] + "\n...[已截断]"

    def _parse_action(self, raw: str) -> dict[str, Any]:
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            raw = raw.removeprefix("json").strip()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return {"action": "final_answer", "answer": raw}

        if not isinstance(data, dict):
            return {"action": "final_answer", "answer": raw}
        if data.get("action") not in {"search_papers", "read_chunks", "final_answer"}:
            return {"action": "final_answer", "answer": raw}
        return data


def _is_summary_request(text: str) -> bool:
    lowered = text.lower()
    keywords = ["总结", "审稿", "阅读这篇", "summarize", "review"]
    return any(keyword in lowered for keyword in keywords)


def detect_summary_mode(text: str) -> str:
    lowered = text.lower()
    deep_keywords = [
        "深度阅读",
        "详细综述",
        "完整综述",
        "完整 review",
        "完整review",
        "详细 review",
        "详细review",
        "deep review",
        "full review",
        "detailed summary",
        "review",
        "详细总结",
    ]
    standard_keywords = ["standard", "标准", "中等", "较详细", "常规综述"]
    if any(keyword in lowered for keyword in deep_keywords):
        return "deep"
    if any(keyword in lowered for keyword in standard_keywords):
        return "standard"
    return "quick"


def _paper_query_terms(text: str) -> list[str]:
    raw_terms = re.findall(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]{2,}", text.lower())
    stop_words = {
        "请",
        "帮我",
        "这篇",
        "文献",
        "论文",
        "文章",
        "总结",
        "阅读",
        "审稿",
        "paper",
        "pdf",
        "summarize",
        "review",
    }
    return [term for term in raw_terms if term not in stop_words and len(term) >= 2]
