# Paper Manager

Paper Manager 是一个基于本地 vLLM 的文献阅读 agent，面向"英文论文 + 中文提问"场景。PDF 原文按 chunk 保留英文证据，用户用中文指定文献、生成总结、继续追问，agent 通过 query rewrite、embedding 检索和关键词检索找到相关片段，再用中文回答。

## 核心功能

- 读取 `paper_rep/` 中的 PDF/TXT/MD 文献，支持交互式问答和连续追问。
- Agent 身份"小智"：幽默但专业的文献审稿人。
- Web 界面：`MonkeyLu's Paper Manager Agent` 学术检索风格工作台。
- 审稿式总结：输出 `# 摘要`、`# 主要内容`、`# 核心算法`、`# 算例分析` 四段式 Markdown。
- 三种总结模式：`quick`（默认）、`standard`、`deep`。
- 中英文检索：中文 query 自动改写为英文 query + 关键词 query + section hints。
- Hybrid retrieval：embedding 向量检索 + 关键词检索，multi-signal rerank 合并重排。
- 回答带英文证据引用：`[paper_id=..., page=..., section=..., chunk_id=...]`。

## 目录结构

```text
AGENTS.md                          # 仓库级编码指令
agent.md                           # 小智运行时审稿 prompt
paper_rep/                         # 论文存放目录
outputs/                           # 总结 Markdown 输出
data/chunks/index.json             # chunk 文本索引
data/embeddings/                   # embedding 缓存和元数据
data/review_cache/                 # 总结 notes 缓存

paper_agent/
├── cli.py                         # CLI 入口
├── agent.py                       # 交互式 agent (ReAct loop)
├── paper_store.py                 # 文献解析和 section-aware 分块
├── query_rewriter.py              # 中文问题 → 英文检索 query
├── retriever.py                   # 关键词检索
├── vector_retriever.py            # embedding 检索 + hybrid rerank
├── embedding_client.py            # embedding 客户端 (GPU/CPU fallback)
├── reviewer.py                    # quick/standard/deep 总结流程
├── prompts.py                     # prompt 模板
├── web_app.py                     # Web 界面服务
├── web_static/                    # Web 前端资源
└── llm_client.py                  # vLLM OpenAI-compatible 客户端
```

## 环境配置

在 WSL/Linux 项目根目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

如果使用 `uv`：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv venv .venv
source .venv/bin/activate
UV_CACHE_DIR=/tmp/uv-cache uv pip install -r requirements.txt
```

基础依赖已包含 `sentence-transformers`（CUDA torch），embedding 在 GPU 上完成。Embedding fallback 链路：

```text
BGE-M3 (1024d, GPU) → MiniLM-L12-v2 (384d, GPU) → CPU fallback → local-hashing-multilingual-v1 (768d)
```

embedding 配置在 `config.yaml`：

```yaml
embedding_enabled: true
embedding_backend: "auto"          # auto = 优先 GPU neural，失败自动降级
embedding_model: "BAAI/bge-m3"
embedding_device: "cuda"
retrieval_backend: "hybrid"
```

## 启动与使用

### 一键启动（推荐）

```bash
./run_agent_wsl.sh                # 构建索引 → 启动 vLLM → 进入 agent
AGENT_MODE=web ./run_agent_wsl.sh  # 启动 Web UI，默认 http://127.0.0.1:7860
```

脚本会自动：构建 chunk 索引并计算 embedding（GPU，在 vLLM 之前启动，避免显存竞争）→ 检查或启动 vLLM → 探测可访问的 API 地址 → 进入交互模式。

如果索引已是最新（无新论文加入），自动跳过重建。

### 仅启动 vLLM

```bash
./start_vllm_qwen3_4b.sh          # 只启动 vLLM API 服务
```

### 手动命令

```bash
python -m paper_agent.cli --list
python -m paper_agent.cli index
python -m paper_agent.cli chat --model qwen3-4b --base-url http://127.0.0.1:8000/v1
python -m paper_agent.cli "论文文件名" --model qwen3-4b --base-url http://127.0.0.1:8000/v1 --summary-mode quick
```

### 交互示例

```text
User> 总结 Zhen 这篇论文
User> 本文的核心算法是什么
User> 它是如何实现时空并行的
User> 这篇论文的算例如何验证方法有效性
```

退出：`q` / `quit` / `exit`

## 总结模式

```text
quick    默认，最多 8 个 chunk，不做多轮压缩。
standard 中等，最多 12 个 chunk，允许一轮 notes compression。
deep     深度，最多 30 个 chunk，允许多轮 compression。
```

普通"总结一下"默认走 `quick`。明确说"深度阅读 / 详细综述 / 完整 review"才触发 `deep`。

## 检索流程

```text
中文问题 → query rewrite (英文 query + 关键词 + section hints)
  → embedding 检索 (top_k×3) + 关键词检索 (top_k×3)
  → hybrid merge (加权合并)
  → multi-signal rerank (向量相似度 + 关键词覆盖 + 章节优先级 + 精确短语匹配)
  → top chunks 入 prompt
```

分块策略使用 section-aware chunking：先检测论文章节边界（abstract、method、experiment 等），按章节拆分后再对超长章节做二次分块，保证每个 chunk 语义内聚。

## 核心优化

本轮优化涉及五个层面：

### 1. Agent 决策层

- **总结意图精准识别**：用正则模式匹配替代简单关键词包含，避免"我来 review 一下这段代码"被误判为论文总结请求。
- **消除冗余预检索**：移除 ReAct 循环前的强制 retrieval，每轮节省约 700 字符 prompt 开销，让 agent 自主决定是否检索。
- **清理死代码**：移除 `load_agent_instructions`（加载但从未使用）、`_is_overview_query`（定义但从未调用）等无效逻辑。

### 2. 检索基础设施层

- **Embedding 矩阵内存缓存**：基于文件 mtime 的缓存失效机制，避免每次查询重复 `np.load` + JSON 解析，节省 50-200ms 磁盘 I/O。
- **学术术语词典缓存**：`load_academic_terms` 增加 mtime 缓存，避免重复读取和解析 JSON。

### 3. 报告生成层

- **分模式报告生成**：quick/standard 模式将 4 次独立 LLM 调用合并为 1 次，输入 token 减少约 3 倍；deep 模式保留 4 次分节调用，每节独立获得完整 token 预算（750 tokens），确保 30 chunk 丰富笔记的输出深度。
- **自动段落切分**：`_split_report_sections` 解析单次输出中的 `# ` 标题，拆分为四节 Markdown（供 quick/standard 使用）。

### 4. 启动脚本层

- **HuggingFace 离线模式**：`HF_HUB_OFFLINE=1` + `TRANSFORMERS_OFFLINE=1`，模型已缓存时不再尝试网络请求，消除 HuggingFace 超时阻塞。
- **索引增量检测**：比较 `index.json` mtime 与 `paper_rep/` 最新论文 mtime，无新论文时跳过重建，节省 5-30 秒启动时间。
- **GPU 显存分时复用**：embedding 计算在 vLLM 启动前完成，避免两者竞争显存。

### 5. LLM 通信层

- **重试 + 指数退避**：LLM 调用失败后自动重试 3 次（1s → 2s → 4s），提高不稳定环境下的鲁棒性。
- **thinking 支持缓存**：首次调用后缓存 `enable_thinking` 是否被支持，避免后续每次调用都触发 TypeError 回退。

## vLLM 参数

默认参数（适配 GTX 1660 SUPER 6GB）：

```text
VLLM_MAX_MODEL_LEN=4096
VLLM_GPU_MEMORY_UTILIZATION=0.80
VLLM_MAX_NUM_SEQS=2
VLLM_ENABLE_PREFIX_CACHING=1
VLLM_ENABLE_CHUNKED_PREFILL=1
VLLM_ENFORCE_EAGER=0
```

常用调参：

```bash
VLLM_FORCE_RESTART=1 VLLM_GPU_MEMORY_UTILIZATION=0.85 ./run_agent_wsl.sh
VLLM_FORCE_RESTART=1 VLLM_MAX_MODEL_LEN=6144 VLLM_GPU_MEMORY_UTILIZATION=0.85 ./run_agent_wsl.sh
```

## 常见问题

**vLLM 参数修改没生效**：脚本复用了旧服务，使用 `VLLM_FORCE_RESTART=1 ./run_agent_wsl.sh` 强制重启。

**无法连接 vLLM**：`curl --noproxy '*' http://127.0.0.1:8000/v1/models`，WSL 中 `run_agent_wsl.sh` 会自动探测可访问地址。

**上下文超限**：降低 `AGENT_MAX_TOKENS=400` 或扩大 `VLLM_MAX_MODEL_LEN`（需更多显存）。

**Embedding 模型加载失败**：`embedding_backend: "auto"` 自动走 fallback 链路，最终降级到本地 hashing embedding。

**HuggingFace 超时**：脚本已设置 `HF_HUB_OFFLINE=1`，确保模型已缓存在 `~/.cache/huggingface/hub/` 即可。

## 设计约束

- 不全文翻译英文论文，保留英文原文证据。
- 不把整篇论文塞进 prompt，通过 RAG 精选 chunk。
- 不在 chat 阶段重算所有 chunk embedding。
- 不大规模重构现有 agent 框架。
