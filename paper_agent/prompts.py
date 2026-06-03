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
