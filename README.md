# Paper Manager

Paper Manager 是一个基于本地 vLLM 的文献阅读 agent。它面向“英文论文 + 中文提问”的使用场景：PDF 原文按 chunk 保留英文证据，用户用中文指定文献、生成总结、继续追问，agent 通过 query rewrite、embedding 检索和关键词检索找到相关片段，再用中文回答。

## 核心功能

- 读取 `paper_rep/` 中的 PDF/TXT/MD 文献。
- 交互式问答：指定当前文献后可连续追问。
- 审稿式总结：输出 `# 摘要`、`# 主要内容`、`# 核心算法`、`# 算例分析`。
- 三种总结模式：`quick` 默认、`standard` 中等、`deep` 深度。
- 中英文检索：中文 query 自动改写为英文 query、关键词 query 和 section hints。
- Hybrid retrieval：embedding 向量检索 + 关键词检索合并重排。
- 回答带英文证据引用：`[paper_id=..., page=..., section=..., chunk_id=...]`。

## 目录结构

```text
paper_rep/                         # 放论文
outputs/                           # 总结 Markdown
data/chunks/index.json             # chunk 文本索引
data/embeddings/chunk_embeddings.npy
data/embeddings/chunk_meta.json    # embedding 缓存和元数据
data/review_cache/                 # 总结 notes 缓存

paper_agent/
├── cli.py                         # CLI 入口
├── agent.py                       # 交互式 agent
├── paper_store.py                 # 文献解析和索引
├── query_rewriter.py              # 中文问题改写为英文检索 query
├── retriever.py                   # 关键词检索和格式化
├── vector_retriever.py            # embedding 检索和 hybrid rerank
├── embedding_client.py            # embedding 客户端
├── reviewer.py                    # quick/standard/deep 总结流程
├── prompts.py                     # prompt 模板
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

基础依赖已支持本地 `local-hashing-multilingual-v1` embedding fallback。若要使用 BGE-M3 / sentence-transformers neural embedding，可选安装：

```bash
pip install -r requirements-embedding.txt \
  -i https://mirrors.aliyun.com/pypi/simple/ \
  --extra-index-url https://mirrors.aliyun.com/pytorch-wheels/
```

embedding 配置在 [config.yaml](/mnt/e/LLM_Project/paper_mananger/config.yaml:1)：

```yaml
embedding_enabled: true
embedding_backend: "auto"
embedding_model: "BAAI/bge-m3"
embedding_hashing_model: "local-hashing-multilingual-v1"
embedding_device: "cpu"
retrieval_backend: "hybrid"
```

`embedding_backend: "auto"` 会优先尝试 neural embedding；不可用时自动使用本地 hashing embedding。

## 两个启动脚本

### `run_agent_wsl.sh`

[run_agent_wsl.sh](/mnt/e/LLM_Project/paper_mananger/run_agent_wsl.sh:1) 是日常推荐的一键脚本。

它会自动：

- 检查或启动 vLLM。
- 探测 WSL 中可访问的 vLLM API 地址。
- 构建 `data/chunks/index.json`。
- 构建或复用 embedding 缓存。
- 进入交互式 paper agent。

使用：

```bash
./run_agent_wsl.sh
```

如果修改了 vLLM 参数，强制重启服务：

```bash
VLLM_FORCE_RESTART=1 ./run_agent_wsl.sh
```

### `start_vllm_qwen3_4b.sh`

[start_vllm_qwen3_4b.sh](/mnt/e/LLM_Project/paper_mananger/start_vllm_qwen3_4b.sh:1) 只启动 vLLM 服务。

它不会启动 agent，也不会构建索引。适合单独保留一个 vLLM API 服务：

```bash
./start_vllm_qwen3_4b.sh
```

然后另开终端启动 agent：

```bash
source .venv/bin/activate
python -m paper_agent.cli chat \
  --model qwen3-4b \
  --base-url http://127.0.0.1:8000/v1
```

简单记：

```text
run_agent_wsl.sh          启动/复用 vLLM + 构建索引 + 启动 agent
start_vllm_qwen3_4b.sh    只启动 vLLM API 服务
```

## 日常使用

启动：

```bash
./run_agent_wsl.sh
```

交互示例：

```text
User> 总结 Zhen 这篇论文
User> 本文的核心算法是什么
User> 它是如何实现时空并行的
User> 这篇论文的算例如何验证方法有效性
```

退出：

```text
q
quit
exit
```

手动命令：

```bash
python -m paper_agent.cli --list
python -m paper_agent.cli index
python -m paper_agent.cli chat --model qwen3-4b --base-url http://127.0.0.1:8000/v1
python -m paper_agent.cli "论文文件名或唯一关键词" --model qwen3-4b --base-url http://127.0.0.1:8000/v1 --summary-mode quick
```

## 总结模式

```text
quick    默认模式，最多精选 8 个 chunk，不做多轮压缩。
standard 中等模式，最多精选 12 个 chunk，允许一轮 notes compression。
deep     深度模式，最多 30 个 chunk，允许多轮 compression。
```

普通“总结一下 / 这篇论文讲什么”默认走 `quick`。明确说“深度阅读 / 详细综述 / 完整 review”才触发 `deep`，并提示：

```text
Deep review will be slower on local GPU.
```

最终 Markdown 分章节生成：

```text
paper_card -> 摘要 / 主要内容 / 核心算法 / 算例分析 -> Python 拼接 Markdown
```

## 检索与 Embedding

构建索引会生成或复用：

```text
data/chunks/index.json
data/embeddings/chunk_embeddings.npy
data/embeddings/chunk_meta.json
```

确认 embedding 是否生效：

```bash
python - <<'PY'
import json
from pathlib import Path
meta = json.loads(Path("data/embeddings/chunk_meta.json").read_text(encoding="utf-8"))
print(meta["embedding_model"])
print(meta["embedding_backend"])
print(len(meta["chunks"]))
PY
```

检索流程：

```text
中文问题 -> query rewrite -> embedding 检索 + 关键词检索 -> hybrid rerank -> top chunks 入 prompt
```

日志字段：

```text
embedding_model
indexed_chunk_count
embedding_cache_hit
query_language
retrieved_chunks
embedding_search_time
```

## vLLM 优化方案

优化分两层：

```text
agent 层：RAG top chunks、embedding 缓存、分章节总结、短 prompt。
vLLM 层：KV cache、prefix caching、scheduler、显存比例、上下文长度。
```

默认 vLLM 参数：

```text
VLLM_MAX_MODEL_LEN=2048
VLLM_GPU_MEMORY_UTILIZATION=0.70
VLLM_KV_CACHE_DTYPE=auto
VLLM_MAX_NUM_SEQS=4
VLLM_ENABLE_PREFIX_CACHING=1
VLLM_ENABLE_CHUNKED_PREFILL=0
VLLM_ENFORCE_EAGER=1
VLLM_KV_CACHE_METRICS=0
```

常用调参：

```bash
VLLM_FORCE_RESTART=1 VLLM_GPU_MEMORY_UTILIZATION=0.80 ./run_agent_wsl.sh
VLLM_FORCE_RESTART=1 VLLM_ENFORCE_EAGER=0 ./run_agent_wsl.sh
VLLM_FORCE_RESTART=1 VLLM_MAX_MODEL_LEN=4096 AGENT_MAX_INPUT_TOKENS=1600 AGENT_MAX_TOKENS=700 ./run_agent_wsl.sh
VLLM_FORCE_RESTART=1 VLLM_KV_CACHE_METRICS=1 ./run_agent_wsl.sh
```

说明：

- `VLLM_ENABLE_PREFIX_CACHING=1` 默认开启，可复用相同 prompt 前缀的 KV cache。
- `VLLM_KV_CACHE_DTYPE=fp8` 可能降低 KV cache 显存占用，但取决于 vLLM/CUDA/模型支持情况，默认保持 `auto`。
- `VLLM_ENABLE_CHUNKED_PREFILL=1` 适合长 prompt 或多请求调度测试，本项目默认关闭。
- embedding/RAG 能减少放进 prompt 的内容，但不能突破 vLLM 的 `--max-model-len`。

日志：

```bash
tail -n 120 /tmp/paper_agent_logs/vllm_qwen3-4b_8000.log
```

## 常见问题

### `vllm: command not found`

```bash
which vllm
vllm --version
export VLLM_BIN=/path/to/vllm
```

### 无法连接 vLLM

```bash
curl --noproxy '*' http://127.0.0.1:8000/v1/models
ss -ltnp | grep 8000
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
```

WSL 中如果 `127.0.0.1` 不稳定，优先使用 `./run_agent_wsl.sh` 自动探测可访问地址。

### 修改 vLLM 参数后没生效

如果脚本显示：

```text
Reusing running vLLM API
```

说明复用了旧服务。使用：

```bash
VLLM_FORCE_RESTART=1 ./run_agent_wsl.sh
```

### 上下文超限

如果看到：

```text
maximum context length is 2048 tokens
```

降低预算：

```bash
AGENT_MAX_TOKENS=400 ./run_agent_wsl.sh
python -m paper_agent.cli chat --max-input-tokens 900 --max-tokens 400
```

或先扩大 vLLM 上下文：

```bash
VLLM_FORCE_RESTART=1 VLLM_MAX_MODEL_LEN=4096 AGENT_MAX_INPUT_TOKENS=1600 AGENT_MAX_TOKENS=700 ./run_agent_wsl.sh
```

### embedding 模型加载失败

保持 `embedding_backend: "auto"` 即可自动使用本地 hashing fallback。若要 neural embedding：

```bash
pip install -r requirements-embedding.txt \
  -i https://mirrors.aliyun.com/pypi/simple/ \
  --extra-index-url https://mirrors.aliyun.com/pytorch-wheels/
```

也可以把 `config.yaml` 中的 `embedding_model` 改成本地模型路径。

### 检索不到英文论文内容

检查：

```bash
python -m paper_agent.cli index
ls data/chunks/index.json data/embeddings/chunk_embeddings.npy
```

确认正在使用中英文 query rewrite 和 hybrid retrieval。回答仍应引用英文 chunk 来源。

## 设计约束

- 不全文翻译英文论文。
- 不把整篇论文塞进 prompt。
- 不在 chat 阶段重算所有 chunk embedding。
- 不丢弃英文原文证据。
- 不大规模重构现有 agent。
