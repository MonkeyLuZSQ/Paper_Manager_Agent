from __future__ import annotations

import argparse
from pathlib import Path

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    load_dotenv = None

from paper_agent.document_loader import SUPPORTED_SUFFIXES
from paper_agent.llm_client import LLMConfig, VLLMClient
from paper_agent.reviewer import resolve_paper_path, review_paper, write_report


DEFAULT_PAPER_DIR = Path("paper_rep")
DEFAULT_OUTPUT_DIR = Path("outputs")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read a paper with a local vLLM model and generate a reviewer-style Chinese summary."
    )
    parser.add_argument(
        "paper",
        nargs="?",
        help="Paper file name in ./paper_rep, or a unique partial name. Use --list to see files.",
    )
    parser.add_argument("--list", action="store_true", help="List supported papers in paper_dir.")
    parser.add_argument("--paper-dir", type=Path, default=DEFAULT_PAPER_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", help="vLLM model name. Can also be set with VLLM_MODEL.")
    parser.add_argument(
        "--base-url",
        help="vLLM OpenAI-compatible endpoint. Default: VLLM_BASE_URL or http://localhost:8000/v1.",
    )
    parser.add_argument("--api-key", help="API key for the endpoint. Default: VLLM_API_KEY or EMPTY.")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--chunk-chars", type=int, default=12000)
    parser.add_argument("--overlap", type=int, default=800)
    return parser


def list_papers(paper_dir: Path) -> int:
    if not paper_dir.exists():
        print(f"Paper directory does not exist: {paper_dir}")
        return 1

    papers = sorted(
        path for path in paper_dir.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if not papers:
        supported = ", ".join(sorted(SUPPORTED_SUFFIXES))
        print(f"No supported papers found in {paper_dir}. Supported suffixes: {supported}")
        return 0

    for path in papers:
        print(path.name)
    return 0


def main() -> int:
    if load_dotenv:
        load_dotenv()

    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.list:
            return list_papers(args.paper_dir)

        if not args.paper:
            parser.error("paper is required unless --list is used")

        paper_path = resolve_paper_path(args.paper, args.paper_dir)
        config = LLMConfig.from_env(
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        llm = VLLMClient(config)

        print(f"Reading paper: {paper_path}")
        print(f"Using vLLM endpoint: {config.base_url}")
        print(f"Using model: {config.model}")

        result = review_paper(
            paper_path,
            llm,
            max_chars=args.chunk_chars,
            overlap=args.overlap,
        )
        output_path = write_report(result, args.output_dir)
        print(f"Reviewed {result.chunks} chunk(s).")
        print(f"Report written to: {output_path}")
        return 0
    except (FileNotFoundError, ModuleNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
