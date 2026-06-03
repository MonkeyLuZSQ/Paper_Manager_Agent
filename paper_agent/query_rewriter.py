from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


TERMS_PATH = Path("academic_terms_zh_en.json")


@dataclass(frozen=True)
class RewrittenQuery:
    original_query: str
    english_query: str
    academic_query: str
    keyword_queries: list[str]
    section_hints: list[str]
    concept_aliases: list[str]


def detect_query_language(user_query: str) -> str:
    chinese_chars = re.findall(r"[\u4e00-\u9fff]", user_query)
    ascii_words = re.findall(r"[A-Za-z]{2,}", user_query)
    if chinese_chars and len(chinese_chars) >= len(ascii_words):
        return "zh"
    if chinese_chars:
        return "mixed"
    return "en"


def rewrite_query(user_query: str, terms_path: Path = TERMS_PATH) -> RewrittenQuery:
    language = detect_query_language(user_query)
    term_map = load_academic_terms(terms_path)
    aliases = concept_aliases(user_query, term_map)
    section_hints = infer_section_hints(user_query)
    keyword_queries = dedupe([*aliases, *section_hints])

    if language in {"zh", "mixed"}:
        english_query = build_english_query(user_query, aliases, section_hints)
    else:
        english_query = user_query

    academic_query = " ".join(dedupe([english_query, *aliases, *section_hints]))
    return RewrittenQuery(
        original_query=user_query,
        english_query=english_query,
        academic_query=academic_query,
        keyword_queries=keyword_queries,
        section_hints=section_hints,
        concept_aliases=aliases,
    )


def load_academic_terms(path: Path = TERMS_PATH) -> dict[str, list[str]]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(key): [str(item) for item in value] for key, value in data.items()}


def concept_aliases(query: str, term_map: dict[str, list[str]]) -> list[str]:
    lowered = query.lower()
    aliases: list[str] = []
    for zh_term, en_terms in term_map.items():
        if zh_term in query:
            aliases.extend(en_terms)

    direct_rules = {
        ("求的是什么方程", "什么方程", "方程", "模型"): [
            "governing equations",
            "mathematical model",
            "Navier-Stokes equations",
            "formulation",
            "discretization",
        ],
        ("核心算法", "算法", "如何实现", "怎么实现"): [
            "core algorithm",
            "algorithm",
            "method",
            "scheme",
            "formulation",
        ],
        ("时空并行", "时间并行", "空间并行"): [
            "space-time parallel",
            "time communicator",
            "space communicator",
            "parallel computing",
            "MPI",
        ],
        ("主要内容", "讲什么", "研究什么"): [
            "abstract",
            "introduction",
            "method",
            "conclusion",
        ],
        ("算例", "实验", "验证"): [
            "numerical experiment",
            "test problem",
            "accuracy",
            "speedup",
            "scalability",
        ],
    }
    for triggers, en_terms in direct_rules.items():
        if any(trigger in lowered for trigger in triggers):
            aliases.extend(en_terms)
    return dedupe(aliases)


def infer_section_hints(query: str) -> list[str]:
    lowered = query.lower()
    hints: list[str] = []
    if any(term in lowered for term in ["方程", "模型", "推导", "公式"]):
        hints.extend(["Abstract", "Introduction", "Method", "Formulation", "Governing Equations"])
    if any(term in lowered for term in ["核心算法", "算法", "方法", "实现", "时空并行", "并行"]):
        hints.extend(["Method", "Algorithm", "Formulation"])
    if any(term in lowered for term in ["算例", "实验", "结果", "验证", "图"]):
        hints.extend(["Numerical examples", "Experiments", "Results"])
    if any(term in lowered for term in ["主要内容", "讲什么", "研究什么", "总结", "摘要"]):
        hints.extend(["Abstract", "Introduction", "Conclusion"])
    if any(term in lowered for term in ["局限", "不足", "未来"]):
        hints.extend(["Conclusion", "Discussion", "Future Work"])
    return dedupe(hints or ["Abstract", "Introduction", "Method", "Conclusion"])


def build_english_query(query: str, aliases: list[str], section_hints: list[str]) -> str:
    lowered = query.lower()
    if any(term in lowered for term in ["方程", "模型", "求的是什么"]):
        return "What governing equations or mathematical model are used in this paper?"
    if any(term in lowered for term in ["核心算法", "算法"]):
        return "What is the core algorithm or numerical method proposed in this paper?"
    if any(term in lowered for term in ["时空并行", "并行"]):
        return "How does the paper implement space-time parallel computing?"
    if any(term in lowered for term in ["算例", "实验", "结果"]):
        return "What numerical experiments, results, accuracy, speedup or scalability are reported?"
    if any(term in lowered for term in ["主要内容", "讲什么", "研究什么", "总结"]):
        return "What are the main research problem, method, results and conclusions of this paper?"
    if aliases:
        return " ".join(aliases[:8])
    return " ".join(section_hints)


def dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        normalized = item.strip()
        key = normalized.lower()
        if normalized and key not in seen:
            seen.add(key)
            output.append(normalized)
    return output
