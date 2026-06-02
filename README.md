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

在本项目根目录准备 agent 运行环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

如果系统没有 `venv` 或 `pip`，但安装了 `uv`，可以使用：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv venv .venv
source .venv/bin/activate
UV_CACHE_DIR=/tmp/uv-cache uv pip install -r requirements.txt
```

依赖说明：

- `openai`：调用 vLLM 的 OpenAI-compatible API
- `pypdf`：读取 PDF 文本
- `python-dotenv`：从 `.env` 加载配置

## 2. 当前机器上的 vLLM 环境

本次检查发现，当前 WSL 的系统 PATH 中没有 `vllm` 命令，系统 Python 也没有 `pip`。真正可用的 vLLM 命令位于：

```text
/mnt/e/LLM_Project/vllm_demo/.venv/bin/vllm
```

对应的 demo 启动脚本是：

```text
/mnt/e/LLM_Project/vllm_demo/serve_openai.sh
```

该脚本使用的 OpenAI API 模型名是：

```text
qwen3-4b
```

所以本项目应使用 `qwen3-4b`，而不是早期示例中的 `Qwen/Qwen2.5-7B-Instruct`。

## 3. 启动 vLLM 服务

请在另一个 WSL 终端启动 vLLM 服务。建议先只绑定本机回环地址：

```bash
cd /mnt/e/LLM_Project/vllm_demo
source .venv/bin/activate
vllm serve Qwen/Qwen3-4B-AWQ \
  --quantization awq_marlin \
  --dtype float16 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.70 \
  --enforce-eager \
  --served-model-name qwen3-4b \
  --host 127.0.0.1 \
  --port 8000
```

如果 4B 模型在 GTX 1660 Super 6GB + WSL 环境下启动很慢或迟迟不开放端口，可以先用小模型验证连通性：

```bash
cd /mnt/e/LLM_Project/vllm_demo
source .venv/bin/activate
vllm serve Qwen/Qwen3-0.6B \
  --served-model-name qwen3-0.6b \
  --host 127.0.0.1 \
  --port 8000
```

关键点：`--served-model-name` 必须和 agent 使用的 `--model` 或 `.env` 里的 `VLLM_MODEL` 完全一致。

## 4. 检查 vLLM 是否真的启动成功

确认端口监听：

```bash
ss -ltnp | grep 8000
```

确认 OpenAI-compatible API 可访问：

```bash
curl http://127.0.0.1:8000/v1/models
```

只有 `curl` 返回模型列表后，paper agent 才能正常生成报告。仅仅“安装了 vLLM”还不够，必须有一个正在运行并监听 `8000` 的 vLLM API 服务。

## 5. 配置 agent

复制配置模板：

```bash
cp .env.example .env
```

当前 `.env.example` 内容：

```text
VLLM_MODEL=qwen3-4b
VLLM_BASE_URL=http://127.0.0.1:8000/v1
VLLM_API_KEY=EMPTY
```

如果你启动的是小模型连通性测试服务，请把模型名改为：

```text
VLLM_MODEL=qwen3-0.6b
```

## 6. 放入论文

把要分析的文献放入：

```text
paper_rep/
```

支持格式：

- `.pdf`
- `.txt`
- `.md`

如果 PDF 是扫描版图片，`pypdf` 可能无法抽取正文，需要先做 OCR，保存为可搜索 PDF 或文本文件。

## 7. 查看可用文献

```bash
python3 -m paper_agent.cli --list
```

当前测试可识别：

```text
Zhen 等 - 2024 - High-order space–time parallel computing of the Navier–Stokes equations.pdf
```

## 8. 运行论文审稿 agent

如果 `.env` 中配置的是 `qwen3-4b`：

```bash
python3 -m paper_agent.cli "Zhen 等 - 2024 - High-order space–time parallel computing of the Navier–Stokes equations.pdf" --chunk-chars 3000 --overlap 300 --max-tokens 900
```

也可以显式指定模型：

```bash
python3 -m paper_agent.cli "Zhen 等 - 2024 - High-order space–time parallel computing of the Navier–Stokes equations.pdf" --model qwen3-4b --chunk-chars 3000 --overlap 300 --max-tokens 900
```

如果你先用小模型 `qwen3-0.6b` 做连通性测试：

```bash
python3 -m paper_agent.cli "Zhen 等 - 2024 - High-order space–time parallel computing of the Navier–Stokes equations.pdf" --model qwen3-0.6b --chunk-chars 2000 --overlap 200 --max-tokens 600
```

生成结果会写入：

```text
outputs/Zhen 等 - 2024 - High-order space–time parallel computing of the Navier–Stokes equations_review.md
```

## 9. 工作流程

agent 的处理流程如下：

1. 从 `paper_rep/` 定位指定文献，支持完整文件名或唯一的部分文件名。
2. 使用 `pypdf` 读取 PDF，或直接读取 TXT/Markdown。
3. 对长论文进行文本清洗和分块。
4. 对每个分块生成局部审稿笔记，包括主题、方法、实验、创新点和不足。
5. 汇总所有分块笔记，生成最终报告。
6. 将报告保存到 `outputs/`。

## 10. 常用参数

- `--paper-dir`：论文目录，默认 `paper_rep`
- `--output-dir`：报告输出目录，默认 `outputs`
- `--model`：vLLM 服务模型名，也可用 `VLLM_MODEL` 配置
- `--base-url`：vLLM 服务地址，默认 `http://localhost:8000/v1`
- `--api-key`：API key，本地 vLLM 通常使用 `EMPTY`
- `--chunk-chars`：长论文分块长度
- `--overlap`：分块重叠长度
- `--temperature`：生成温度，默认 `0.2`
- `--max-tokens`：每次模型输出上限

因为当前 vLLM demo 的 `--max-model-len` 建议设为 `2048` 或 `4096`，不建议直接使用默认 `--chunk-chars 12000`。本机建议从下面的参数开始：

```bash
--chunk-chars 3000 --overlap 300 --max-tokens 900
```

## 11. 本次环境检查记录

检查日期：2026-06-02。

已确认：

- `paper_rep` 中已有 PDF 文献，CLI 可以识别。
- PDF 文本抽取成功，抽取到 `63182` 个字符。
- 默认分块参数下可分为 `6` 个文本块。
- `python3 -m compileall paper_agent` 通过。
- 当前普通 Codex 命令运行在网络隔离沙箱中，普通 `ps/ss` 看不到真实 WSL 网络；已使用非沙箱权限检查真实 WSL 环境。
- 真实 WSL 环境中没有长期运行的 vLLM 服务进程。
- `ss -ltnp` 没有发现 `8000` 端口监听。
- `which vllm` 没有返回；vLLM 不在系统 PATH。
- 可用 vLLM 命令在 `/mnt/e/LLM_Project/vllm_demo/.venv/bin/vllm`，版本为 `0.20.2`。
- `vllm_demo/serve_openai.sh` 的 served model name 是 `qwen3-4b`。

尝试结果：

- 使用 `Qwen/Qwen3-4B-AWQ`、`qwen3-4b`、`127.0.0.1:8000` 启动 vLLM，能够进入模型加载阶段。
- GTX 1660 Super 6GB 不支持 FlashAttention 2，vLLM 自动切换到 FLASHINFER。
- 两次启动都长时间停留在 EngineCore/model loading 阶段，没有开放 `127.0.0.1:8000`。
- 继续测试 `Qwen/Qwen3-0.6B` 小模型，现象相同：进入 EngineCore/model loading 阶段后没有开放 `8000`。
- 检查 HuggingFace 缓存发现，`Qwen3-0.6B` 和 `Qwen3-4B-AWQ` 缓存目录都只有约 `16M`，且 `Qwen3-0.6B` 存在 `.incomplete` 文件，说明模型权重没有完整下载。
- 因此完整论文报告生成尚未完成，当前阻塞点是 vLLM 模型下载/加载没有完成，导致 OpenAI API 服务没有成功监听端口；不是 paper agent 的 PDF 读取或分块逻辑问题。

建议下一步：

1. 先完整下载一个小模型，例如 `Qwen/Qwen3-0.6B`。如果网络访问 HuggingFace 不稳定，可以配置 ModelScope 或代理后重新下载。
2. 删除不完整缓存后重新启动 vLLM，确认 `curl http://127.0.0.1:8000/v1/models` 可返回模型列表。
3. 再用 `--model qwen3-0.6b` 跑 agent，验证完整调用链路。
4. 确认链路正常后，再切回 `Qwen/Qwen3-4B-AWQ`，逐步调大 `--max-model-len` 和 agent 的 `--chunk-chars`。

不完整缓存路径示例：

```text
/home/zhengsq/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B
/home/zhengsq/.cache/huggingface/hub/models--Qwen--Qwen3-4B-AWQ
```

## 12. 常见问题

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
Failed to call vLLM endpoint http://127.0.0.1:8000/v1
```

请检查：

- vLLM 服务是否已经启动
- `8000` 端口是否正在监听
- `curl http://127.0.0.1:8000/v1/models` 是否返回模型列表
- `VLLM_BASE_URL` 或 `--base-url` 是否正确
- `VLLM_MODEL` 或 `--model` 是否与 `--served-model-name` 一致

### PDF 没有正文

如果提示没有可抽取文本，通常说明 PDF 是扫描版。请先 OCR，再放入 `paper_rep/`。
