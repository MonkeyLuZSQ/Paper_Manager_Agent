# Repository Instructions

This repository contains MonkeyLu's Paper Manager Agent, a local vLLM-based paper reading and review assistant.

## Scope

These instructions are for Codex and other coding agents working on this repository. They are not the runtime prompt used by the paper-review agent.

The runtime paper-review prompt is kept in `agent.md` and is loaded by the application code.

## Project Rules

- Keep changes minimally invasive.
- Do not rewrite the agent framework unless the user explicitly asks for a redesign.
- Preserve English paper chunks as evidence; do not replace them with full Chinese translations.
- Keep answers and generated reports citation-aware when they depend on paper content.
- Prefer local project patterns over new dependencies.
- When editing startup behavior, update `README.md` with concise usage notes.

## Verification

After code changes, run the smallest relevant checks:

```bash
.venv/bin/python -m compileall paper_agent
bash -n run_agent_wsl.sh
bash -n start_vllm_qwen3_4b.sh
```

For Web UI changes, also check:

```bash
.venv/bin/python -m paper_agent.web_app --help
```
