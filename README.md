# Paper Manager

这是一个基于 vLLM OpenAI-compatible API 的论文阅读 agent。它会读取 `./paper_rep/` 中指定的 PDF、TXT 或 Markdown 文献，先对长论文分块阅读，再按审稿人的方式输出：

- 摘要
- 主要内容
- 核心算法
- 算例分析
- 不足

当前示例文献：

```text
paper_rep/Zhen 等 - 2024 - High-order space–time parallel computing of the Navier–Stokes equations.pdf
```

## 1. 环境准备

建议先在项目根目录创建虚拟环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
```

如果你的系统没有 `venv` 或 `pip`，但安装了 `uv`，也可以使用：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv venv .venv
source .venv/bin/activate
UV_CACHE_DIR=/tmp/uv-cache uv pip install -r requirements.txt
```

普通 Python 环境可以直接安装依赖：

```bash
pip install -r requirements.txt
```

依赖说明：

- `openai`：用于调用 vLLM 提供的 OpenAI-compatible API
- `pypdf`：用于读取 PDF 文本
- `python-dotenv`：用于从 `.env` 加载模型和服务地址配置

## 2. 启动 vLLM 服务

本 agent 不直接加载大模型，而是调用已经启动的 vLLM 服务。请在另一个终端启动 vLLM：

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --host 0.0.0.0 --port 8000
```

如果你的机器显存有限，可以换成更小的 instruct/chat 模型。关键是命令行中的模型名要和后面 agent 使用的 `--model` 保持一致。

vLLM 启动成功后，默认 API 地址是：

```text
http://localhost:8000/v1
```

## 3. 配置模型名

可以每次运行时传入模型名：

```bash
python3 -m paper_agent.cli "your_paper.pdf" --model Qwen/Qwen2.5-7B-Instruct
```

也可以复制 `.env.example` 创建 `.env`：

```bash
cp .env.example .env
```

`.env` 示例：

```text
VLLM_MODEL=Qwen/Qwen2.5-7B-Instruct
VLLM_BASE_URL=http://localhost:8000/v1
VLLM_API_KEY=EMPTY
```

这样后续运行时可以省略 `--model`。

## 4. 放入论文

把要分析的文献放入：

```text
paper_rep/
```

支持格式：

- `.pdf`
- `.txt`
- `.md`

如果 PDF 是扫描版图片，`pypdf` 可能无法抽取正文，需要先做 OCR，保存为可搜索 PDF 或文本文件。

## 5. 查看可用文献

```bash
python3 -m paper_agent.cli --list
```

本项目当前测试得到：

```text
Zhen 等 - 2024 - High-order space–time parallel computing of the Navier–Stokes equations.pdf
```

## 6. 运行论文审稿 agent

使用当前示例文献运行：

```bash
python3 -m paper_agent.cli "Zhen 等 - 2024 - High-order space–time parallel computing of the Navier–Stokes equations.pdf" --model Qwen/Qwen2.5-7B-Instruct
```

如果已经配置 `.env`，可以运行：

```bash
python3 -m paper_agent.cli "Zhen 等 - 2024 - High-order space–time parallel computing of the Navier–Stokes equations.pdf"
```

生成结果会写入：

```text
outputs/Zhen 等 - 2024 - High-order space–time parallel computing of the Navier–Stokes equations_review.md
```

## 7. 工作流程

agent 的处理流程如下：

1. 从 `paper_rep/` 定位指定文献，支持完整文件名或唯一的部分文件名。
2. 使用 `pypdf` 读取 PDF，或直接读取 TXT/Markdown。
3. 对长论文进行文本清洗和分块，默认每块约 `12000` 字符，相邻块重叠 `800` 字符。
4. 对每个分块生成局部审稿笔记，包括主题、方法、实验、创新点和不足。
5. 汇总所有分块笔记，生成最终报告。
6. 将报告保存到 `outputs/`。

## 8. 常用参数

- `--paper-dir`：论文目录，默认 `paper_rep`
- `--output-dir`：报告输出目录，默认 `outputs`
- `--model`：vLLM 模型名，也可用 `VLLM_MODEL` 配置
- `--base-url`：vLLM 服务地址，默认 `http://localhost:8000/v1`
- `--api-key`：API key，本地 vLLM 通常使用 `EMPTY`
- `--chunk-chars`：长论文分块长度，默认 `12000`
- `--overlap`：分块重叠长度，默认 `800`
- `--temperature`：生成温度，默认 `0.2`
- `--max-tokens`：每次模型输出上限，默认 `2048`

示例：减少分块长度，让单次上下文更小：

```bash
python3 -m paper_agent.cli "your_paper.pdf" --model Qwen/Qwen2.5-7B-Instruct --chunk-chars 8000 --overlap 500
```

## 9. 本次测试记录

测试日期：2026-06-02。

已完成的测试：

- `python3 -m compileall paper_agent`：通过
- `.venv/bin/python -m paper_agent.cli --list`：通过，能够识别 `paper_rep` 中的示例 PDF
- 使用 `paper_agent.document_loader.load_document` 读取示例 PDF：通过，抽取到 `63182` 个字符
- 使用默认分块参数处理示例 PDF：通过，分为 `6` 个文本块

完整 agent 调用测试命令：

```bash
.venv/bin/python -m paper_agent.cli "Zhen 等 - 2024 - High-order space–time parallel computing of the Navier–Stokes equations.pdf" --model Qwen/Qwen2.5-7B-Instruct
```

测试结果：

```text
Reading paper: paper_rep/Zhen 等 - 2024 - High-order space–time parallel computing of the Navier–Stokes equations.pdf
Using vLLM endpoint: http://localhost:8000/v1
Using model: Qwen/Qwen2.5-7B-Instruct
Error: Failed to call vLLM endpoint http://localhost:8000/v1. Make sure vLLM is running and the model name is correct.
```

结论：当前代码、PDF 读取、论文发现和分块逻辑已经通过测试；完整生成报告还需要先启动可访问的 vLLM 服务。

## 10. 常见问题

### 找不到论文

确认文件确实放在 `paper_rep/` 下，并且后缀是 `.pdf`、`.txt` 或 `.md`：

```bash
python3 -m paper_agent.cli --list
```

### 缺少依赖

如果看到 `openai is required` 或 `pypdf is required`，说明依赖还没有装好：

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

使用 `uv` 时：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv pip install -r requirements.txt
```

### 无法连接 vLLM

如果看到：

```text
Failed to call vLLM endpoint http://localhost:8000/v1
```

请检查：

- vLLM 服务是否已经启动
- 启动端口是否是 `8000`
- `VLLM_BASE_URL` 或 `--base-url` 是否正确
- `VLLM_MODEL` 或 `--model` 是否与 vLLM 启动时的模型名一致

### PDF 没有正文

如果提示没有可抽取文本，通常说明 PDF 是扫描版。请先 OCR，再放入 `paper_rep/`。
