# Paper Manager

这是一个运行在 WSL 上的本地论文阅读 agent。它使用 vLLM 提供的 OpenAI-compatible Chat Completions API 调用本地模型，支持两种工作方式：

- `review`：读取 `paper_rep/` 中指定文献，生成审稿式总结报告。
- `chat`：进入交互式 CLI agent。你可以指定 `paper_rep/` 中任意一篇文献让它总结，生成 Markdown 后继续围绕该文献问答。

当前项目针对 GTX 1660 Super 这类显存较小、不能使用 FlashAttention 2 的环境做了保守配置：不把整篇论文直接塞进 prompt，而是使用分块、检索、短期记忆和 token 预算控制。

## Agent 框架

目录结构：

```text
paper_agent/
├── cli.py              # CLI 入口：list / review / index / chat
├── agent.py            # 交互式单 agent，包含受控 ReAct loop 和对话记忆
├── tools.py            # 工具层：list_papers / search_papers / read_chunks / rebuild_index
├── paper_store.py      # 文献索引：读取 PDF/TXT/MD，生成带 metadata 的 chunk
├── retriever.py        # 本地关键词检索和 token 估算
├── reviewer.py         # 一次性审稿总结流程
├── llm_client.py       # vLLM OpenAI-compatible API 客户端
├── prompts.py          # 审稿总结和压缩 prompt
├── document_loader.py  # PDF/TXT/MD 文本读取
├── text_utils.py       # 文本清洗和分块
└── ../agent.md         # 专业审稿人式总结规范
```

数据目录：

```text
paper_rep/              # 放论文
data/chunks/index.json  # 本地 chunk 索引
outputs/                # 一次性 review 输出
logs/                   # vLLM 启动日志
```

交互式 agent 的核心流程：

```text
用户指定要总结的文献，或提出论文相关问题
  ↓
如果是总结请求：解析 paper_rep 中的目标文献，阅读全文分块，输出 outputs/<论文名>总结.md
  ↓
如果是问答请求：优先检索当前选中文献的相关 chunk
  ↓
模型选择工具动作 search_papers / read_chunks / final_answer
  ↓
Python 外层最多允许 3 次工具调用
  ↓
基于 observation 生成回答
  ↓
保存最近对话，旧对话压缩为 conversation_summary
```

当前版本先使用轻量关键词检索，后续可以把 `retriever.py` 替换为 embedding + 向量数据库。

## 环境准备

在项目根目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

如果系统没有 `venv` 或 `pip`，但安装了 `uv`：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv venv .venv
source .venv/bin/activate
UV_CACHE_DIR=/tmp/uv-cache uv pip install -r requirements.txt
```

`.env.example`：

```text
VLLM_MODEL=qwen3-4b
VLLM_BASE_URL=http://127.0.0.1:8000/v1
VLLM_API_KEY=EMPTY
```

## vLLM 启动配置

当前机器上的 vLLM 命令位于：

```text
/mnt/e/LLM_Project/vllm_demo/.venv/bin/vllm
```

并已通过下面的软链接加入 PATH：

```text
/home/zhengsq/.local/bin/vllm -> /mnt/e/LLM_Project/vllm_demo/.venv/bin/vllm
```

模型优先使用本地 ModelScope 缓存：

```text
/home/zhengsq/.cache/modelscope/hub/models/Qwen/Qwen3-4B-AWQ
```

GTX 1660 Super 的 compute capability 是 7.5，不支持 FlashAttention 2，因此启动参数必须包含：

```bash
--attention-backend TRITON_ATTN
```

项目中的 `start_vllm_qwen3_4b.sh` 会使用本地模型路径，并绑定：

```bash
--host 0.0.0.0
--port 8000
```

绑定 `0.0.0.0` 是为了解决 WSL 中 `ss` 能看到监听，但 Codex/代理环境访问 `127.0.0.1:8000` 不稳定的问题。

## 一键启动交互式 Agent

推荐直接运行：

```bash
./run_agent_wsl.sh
```

脚本会自动完成：

1. 查找 vLLM 命令。
2. 优先使用本地 ModelScope 模型缓存。
3. 检查 vLLM API 是否可用。
4. 如果端口假监听或旧进程卡住，自动停止并重启 vLLM。
5. 自动探测可访问的 WSL IP，例如 `http://169.254.x.x:8000/v1`。
6. 绕过本机代理变量，避免 `localhost` 请求被 `http_proxy=127.0.0.1:7890` 截走。
7. 构建本地 chunk 索引。
8. 进入交互式 CLI：

```text
User>
```

退出：

```text
exit
quit
q
```

示例问题：

```text
User> 请总结 Li 和 Han 这篇文献
User> 总结 Zhen 等 - 2024 - High-order space–time parallel computing of the Navier–Stokes equations.pdf
User> 它的核心算法步骤是什么？
User> 论文中的算例如何验证方法有效性？
```

总结完成后会生成：

```text
outputs/<PDF文章文件名去掉扩展名>总结.md
```

并将该文献设为当前交互上下文。后续问题会优先围绕这篇文献检索回答。

## 一次性审稿总结

如果只想生成固定格式的审稿报告，可以用旧模式：

```bash
AGENT_MODE=review ./run_agent_wsl.sh
```

输出目录：

```text
outputs/
```

报告结构：

```text
# 摘要
# 主要内容
# 核心算法
# 算例分析
```

也可以手动运行：

```bash
source .venv/bin/activate
python -m paper_agent.cli "Zhen 等 - 2024 - High-order space–time parallel computing of the Navier–Stokes equations.pdf" \
  --model qwen3-4b \
  --base-url http://169.254.162.26:8000/v1 \
  --chunk-chars 3000 \
  --overlap 300 \
  --max-tokens 500
```

其中 `base-url` 以一键脚本探测到的地址为准。

总结 Markdown 文件名统一采用：

```text
论文文件名总结.md
```

## 手动使用 CLI

查看论文：

```bash
python -m paper_agent.cli --list
```

构建索引：

```bash
python -m paper_agent.cli index
```

启动交互式 agent：

```bash
python -m paper_agent.cli chat \
  --model qwen3-4b \
  --base-url http://169.254.162.26:8000/v1 \
  --max-tokens 500 \
  --max-input-tokens 1000
```

索引参数：

```bash
python -m paper_agent.cli index \
  --chunk-chars 1800 \
  --overlap 180 \
  --index-path data/chunks/index.json
```

## 上下文控制

当前 vLLM 启动使用：

```bash
--max-model-len 2048
```

这意味着 prompt token + output token 总和不能超过 2048。项目中做了几层控制：

- 交互式 agent 默认 `--max-input-tokens 1000`。
- 一键脚本默认 `AGENT_MAX_TOKENS=500`。
- 检索只取少量相关 chunk。
- 旧对话不会无限塞进 prompt，会压缩为 `conversation_summary`。
- 一次性 review 会先生成短分块笔记，再多轮压缩，最后汇总。

可以按显存和速度调整：

```bash
AGENT_MAX_TOKENS=500 ./run_agent_wsl.sh
AGENT_CHUNK_CHARS=2500 AGENT_OVERLAP=250 AGENT_MODE=review ./run_agent_wsl.sh
VLLM_WAIT_SECONDS=1200 ./run_agent_wsl.sh
```

## 常见问题

### vLLM: command not found

确认：

```bash
which vllm
vllm --version
```

如果没有结果，设置：

```bash
export VLLM_BIN=/mnt/e/LLM_Project/vllm_demo/.venv/bin/vllm
```

### 8000 没有监听

先看日志：

```bash
tail -n 120 /tmp/paper_agent_logs/vllm_qwen3-4b_8000.log
```

确认端口：

```bash
ss -ltnp | grep 8000
```

一键脚本会自动处理多数情况，包括旧 vLLM 进程卡住、模型路径错误、端口假监听。

### curl 访问 localhost 返回 502

这是代理变量导致的常见问题。当前脚本会使用：

```bash
curl --noproxy '*'
```

并在运行 agent 时清理：

```text
http_proxy
https_proxy
HTTP_PROXY
HTTPS_PROXY
ALL_PROXY
```

`llm_client.py` 也会让 OpenAI/httpx 客户端忽略系统代理环境，避免手动 CLI 调用 vLLM 时被代理截走。

### token 超限

如果看到：

```text
maximum context length is 2048 tokens
```

这通常不是 vLLM 服务不可用，而是某一次 prompt + output token 总和超过了 `--max-model-len 2048`。交互式问答阶段也可能触发，因为 prompt 中会包含当前问题、最近对话、当前论文总结片段和检索到的英文 evidence。

当前代码已经做了这些控制：

- ReAct 工具决策输出限制为 `96` tokens。
- 当前论文 summary 片段限制为约 `260` tokens。
- 最近对话和 conversation summary 单独截断。
- 检索 observation 限制为约 `650` tokens。
- 强制最终回答阶段的检索结果限制为约 `900` tokens。
- 中文 token 估算按单字计数，避免中文被低估后挤爆上下文。

如果仍然看到超限，可以继续降低输出和输入预算：

```bash
AGENT_MAX_TOKENS=400 ./run_agent_wsl.sh
python -m paper_agent.cli chat --max-input-tokens 1200 --max-tokens 400
```

### 生成很慢

GTX 1660 Super 上 4B 模型生成速度有限。优先减少每轮输出：

```bash
AGENT_MAX_TOKENS=400 ./run_agent_wsl.sh
```

也可以先用小模型验证链路：

```bash
VLLM_MODEL_PATH=Qwen/Qwen3-0.6B VLLM_MODEL=qwen3-0.6b ./run_agent_wsl.sh
```

## 当前测试结论

已确认：

- PDF 文献可以读取。
- 本地 chunk 索引可以构建。
- vLLM 可以使用本地 ModelScope `Qwen3-4B-AWQ` 启动。
- `--attention-backend TRITON_ATTN` 可以避开 FlashAttention 2 限制。
- 绑定 `0.0.0.0` 后，WSL IP 可以访问 vLLM API。
- 交互式 agent 已支持检索工具、简化 ReAct loop、短期记忆和上下文预算。
- 最小真实问答已通过：agent 可以检索当前论文片段，并回答“这篇论文主要研究什么？”。

## 总结流程优化记录

本次为了适配本地 `qwen3-4b + GTX 1660 + vLLM + WSL`，对总结流程做了最小侵入式优化。

修改文件：

- `paper_agent/reviewer.py`：新增 `summary_mode`，实现 quick/standard/deep 三种总结模式；新增精选 chunk、review notes 缓存和 LLM 调用计数。
- `paper_agent/agent.py`：根据用户意图自动选择总结模式；deep 模式开始前提示 `Deep review will be slower on local GPU.`。
- `paper_agent/cli.py`：一次性 review 模式新增 `--summary-mode` 参数。
- `paper_agent/prompts.py`：压缩最终总结 prompt，减少 2048 上下文超限风险。
- `run_agent_wsl.sh`：默认 `AGENT_MAX_TOKENS=500`，review 模式支持 `SUMMARY_MODE=quick|standard|deep`。
- `agent.md`：补充三种总结模式说明，保留专业审稿人式输出规范。
- `README.md`：更新使用说明和本优化记录。

后续进一步优化：

- 最终 Markdown 不再由 LLM 一次性生成。
- 程序会先基于审稿笔记生成结构化 `paper_card`。
- 然后分别调用 LLM 生成 `摘要`、`主要内容`、`核心算法`、`算例分析` 四节。
- 最终 Markdown 由 Python 拼接：

```text
final_report = "# 摘要" + section_1 + "# 主要内容" + section_2 + "# 核心算法" + section_3 + "# 算例分析" + section_4
```

这样每次 LLM 调用的上下文更短，同时每一节可以有更充足的输出空间。

三种模式区别：

```text
quick    默认模式，最多精选 8 个 chunk，不做多轮压缩，适合“总结一下 / 简单总结 / 这篇论文讲什么”。
standard 中等模式，最多精选 12 个 chunk，允许一轮 notes compression。
deep     深度模式，最多 30 个 chunk，允许多轮 compression，仅在“深度阅读 / 详细综述 / 完整 review”等明确请求时启用。
```

chunk review 缓存位置：

```text
data/review_cache/<论文名>/<summary_mode>_notes.json
```

缓存的是分块审稿笔记。再次总结同一篇论文、同一模式时，会复用 notes，只重新做必要压缩和最终汇总。

避免 2048 上下文超限的方法：

- quick 默认只精选关键章节 chunk，而不是全文 30 个 chunk。
- 每个精选 chunk 会截断到较短文本。
- quick 不做多轮压缩，standard 只做一轮压缩，deep 才做多轮压缩。
- 分块笔记输出预算降到约 `240` tokens。
- 压缩笔记输出预算降到约 `220` tokens。
- 最终报告改为分章节生成，每节单独调用 LLM，避免一次性 summary prompt 超过 2048。
- 最终 prompt 不再整段塞入 `agent.md`，只保留必要总结规则。

查看每次总结的 LLM 调用次数：

交互式总结完成后，agent 会输出：

```text
总结模式：quick；本次总结 LLM 调用次数：N
```

一次性 review 模式会输出：

```text
Reviewed N selected chunk(s) in quick mode.
LLM calls: N
```

手动指定模式：

```bash
SUMMARY_MODE=standard AGENT_MODE=review ./run_agent_wsl.sh
SUMMARY_MODE=deep AGENT_MODE=review ./run_agent_wsl.sh
```

交互式模式中：

```text
User> 帮我总结下 Zhen 这篇论文
```

默认 quick。

```text
User> 请对 Zhen 这篇论文做深度阅读和完整 review
```

触发 deep。

## 中英文检索优化

当前项目的论文 chunk 保留英文原文，不做全文翻译。用户可以用中文提问，agent 会在内部自动生成英文检索 query，并执行 hybrid retrieval。

新增文件：

- `paper_agent/query_rewriter.py`：检测 query 语言，并生成 bilingual rewritten query。
- `academic_terms_zh_en.json`：常见学术术语中英映射表。
- `config.json`：检索和 embedding 配置。

`query_rewriter.py` 会把中文问题改写为：

```json
{
  "original_query": "这篇论文求的是什么方程？",
  "english_query": "What governing equations or mathematical model are used in this paper?",
  "academic_query": "...",
  "keyword_queries": ["governing equations", "mathematical model", "Navier-Stokes equations"],
  "section_hints": ["Abstract", "Introduction", "Method", "Formulation"],
  "concept_aliases": ["equation", "governing equation", "mathematical model"]
}
```

检索时会合并多路 query：

```text
original_query
english_query
academic_query
keyword_queries
section_hints 加权
```

然后按 `chunk_id` 去重、加权重排，返回英文原文 evidence chunk。

当前 embedding 状态：

```json
{
  "embedding_model": null,
  "embedding_multilingual": false,
  "enable_query_translation": true,
  "enable_keyword_expansion": true,
  "retrieval_backend": "keyword_hybrid"
}
```

也就是说，当前项目还没有真正接入 embedding 模型，检索后端是 keyword hybrid。中文问题检索英文文献时，必须启用 query translation + English keyword expansion。后续如果接入 embedding，建议使用 multilingual embedding 模型，并把 `embedding_multilingual` 设置为 `true`。

回答策略：

- 最终回答使用中文。
- 关键事实必须引用英文原文来源：

```text
[paper_id=..., page=..., section=..., chunk_id=...]
```

- 可以附少量英文原文 evidence，但不要把英文原文全文翻译后当作证据。
