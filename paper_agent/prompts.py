from __future__ import annotations

from pathlib import Path


REVIEWER_SYSTEM_PROMPT = """你是一名严谨的学术论文审稿人和研究助理。
你的任务是阅读论文内容，提炼贡献，也要批判性地识别假设、实验、算法描述和结论中的不足。
请使用中文回答，措辞清晰、具体、可追溯，不要编造论文中没有出现的信息。
如果证据不足，请明确写出“论文片段中未说明”或“需要阅读全文确认”。
"""


def agent_instructions(path: Path = Path("agent.md")) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore").strip()


def chunk_review_prompt(chunk_index: int, total_chunks: int, paper_text: str) -> str:
    return f"""下面是一篇论文的第 {chunk_index}/{total_chunks} 个片段。请先做局部审稿笔记。

请按以下结构输出：
1. 本片段涉及的论文主题和问题背景
2. 本片段出现的主要方法、模型、算法或理论
3. 本片段出现的实验、算例、数据集或评价指标
4. 可识别的创新点或贡献
5. 可识别的不足、疑问或潜在问题
6. 对最终总结有用的关键词和术语

论文片段：
```text
{paper_text}
```
"""


def final_review_prompt(paper_name: str, notes: list[str]) -> str:
    joined_notes = "\n\n".join(
        f"===== 片段 {index} 审稿笔记 =====\n{note}"
        for index, note in enumerate(notes, start=1)
    )
    return f"""请基于以下分片审稿笔记，对论文《{paper_name}》形成一份完整的审稿式阅读总结。

必须严格按以下四个一级标题组织：

# 摘要
用 1-2 段概括论文研究问题、方法、结论和整体价值。

# 主要内容
梳理论文的研究背景、问题定义、整体思路、主要贡献和章节逻辑。

# 核心算法
总结论文的核心算法、模型结构、关键公式或流程。尽量写清输入、输出、关键步骤和技术动机。
如果论文不是算法型论文，请总结其核心方法论或理论框架。

# 算例分析
总结论文中的实验、算例、数据集、评价指标、对比方法和主要结果，并说明这些结果如何支撑论文观点。
如果实验细节不足，请明确指出缺失内容。

要求：
- 使用中文。
- 以专业审稿人的角度写作。
- 优先依据笔记中的证据，不要臆测。
- 不要输出与上述四个一级标题无关的额外一级标题。
- Markdown 文件将由程序保存为“论文文件名总结.md”，你只需要输出 Markdown 正文。
- 内容要具体，避免空泛评价。

分片审稿笔记如下：
```text
{joined_notes}
```
"""


def paper_card_prompt(paper_name: str, notes: list[str]) -> str:
    joined_notes = "\n\n".join(
        f"===== 审稿笔记 {index} =====\n{note}"
        for index, note in enumerate(notes, start=1)
    )
    return f"""请基于以下审稿笔记，为论文《{paper_name}》生成结构化 paper_card。

要求：
- 使用中文。
- 只依据笔记，不要编造。
- 尽量结构化、信息密度高。
- 如果信息不足，写“论文片段中未说明”。

请按以下字段输出：
title_en:
title_zh:
keywords_en:
keywords_zh:
summary_zh:
terminology_map:
研究问题：
研究对象：
核心方法：
核心算法/公式：
实验或算例：
主要结论：
创新点：
局限或疑问：

审稿笔记：
```text
{joined_notes}
```
"""


def section_review_prompt(paper_name: str, section_title: str, paper_card: str, notes: list[str]) -> str:
    joined_notes = "\n\n".join(
        f"===== 审稿笔记 {index} =====\n{note}"
        for index, note in enumerate(notes, start=1)
    )
    section_guidance = {
        "摘要": "概括论文研究问题、方法、结论和价值。用 1-3 段，信息要完整。",
        "主要内容": "说明研究背景、问题定义、整体思路、主要贡献和章节逻辑。",
        "核心算法": "说明核心模型、算法流程、关键公式、输入输出和技术动机；如果不是算法型论文，总结核心方法论。",
        "算例分析": "说明实验/算例设置、评价指标、对比方法、主要结果，以及结果如何支撑结论。",
    }
    return f"""请为论文《{paper_name}》撰写 Markdown 总结中的“{section_title}”一节。

写作要求：
- 只输出本节正文，不要输出一级标题。
- 使用中文。
- 以专业审稿人的视角写作。
- 依据 paper_card 和审稿笔记，不要编造。
- 如果证据不足，明确写“论文片段中未说明”。
- {section_guidance[section_title]}

paper_card：
```text
{paper_card}
```

审稿笔记：
```text
{joined_notes}
```
"""


def compact_notes_prompt(batch_index: int, total_batches: int, notes: list[str]) -> str:
    joined_notes = "\n\n".join(
        f"===== 笔记 {index} =====\n{note}"
        for index, note in enumerate(notes, start=1)
    )
    return f"""下面是论文分片审稿笔记的第 {batch_index}/{total_batches} 批。请压缩为一份中间审稿摘要。

要求：
- 使用中文。
- 保留论文主题、核心方法、算法/公式、算例证据、创新点和不足。
- 尽量精炼，不要逐条复述原文。
- 不要编造笔记中没有的信息。

审稿笔记：
```text
{joined_notes}
```
"""
