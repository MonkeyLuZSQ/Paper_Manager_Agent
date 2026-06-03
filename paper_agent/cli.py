from __future__ import annotations

import argparse
from pathlib import Path

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    load_dotenv = None

from paper_agent.document_loader import SUPPORTED_SUFFIXES
from paper_agent.agent import PaperAgent
from paper_agent.llm_client import LLMConfig, VLLMClient
from paper_agent.paper_store import DEFAULT_INDEX_PATH, build_index
from paper_agent.reviewer import resolve_paper_path, review_paper, write_report


DEFAULT_PAPER_DIR = Path("paper_rep")
DEFAULT_OUTPUT_DIR = Path("outputs")
DEFAULT_INDEX_PATH_ARG = DEFAULT_INDEX_PATH


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
    parser.add_argument("--summary-mode", choices=["quick", "standard", "deep"], default="quick")
    parser.add_argument("--index-path", type=Path, default=DEFAULT_INDEX_PATH_ARG)
    parser.add_argument("--max-input-tokens", type=int, default=1500)
    return parser


def build_chat_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interactive paper agent with local vLLM, retrieval tools, and a controlled ReAct loop."
    )
    subparsers = parser.add_subparsers(dest="command")

    index_parser = subparsers.add_parser("index", help="Build local paper chunk index.")
    index_parser.add_argument("--paper-dir", type=Path, default=DEFAULT_PAPER_DIR)
    index_parser.add_argument("--index-path", type=Path, default=DEFAULT_INDEX_PATH_ARG)
    index_parser.add_argument("--chunk-chars", type=int, default=1800)
    index_parser.add_argument("--overlap", type=int, default=180)

    chat_parser = subparsers.add_parser("chat", help="Start interactive paper agent.")
    chat_parser.add_argument("--paper-dir", type=Path, default=DEFAULT_PAPER_DIR)
    chat_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    chat_parser.add_argument("--index-path", type=Path, default=DEFAULT_INDEX_PATH_ARG)
    chat_parser.add_argument("--model", help="vLLM model name. Can also be set with VLLM_MODEL.")
    chat_parser.add_argument("--base-url", help="vLLM OpenAI-compatible endpoint.")
    chat_parser.add_argument("--api-key", help="API key for the endpoint. Default: VLLM_API_KEY or EMPTY.")
    chat_parser.add_argument("--temperature", type=float, default=0.2)
    chat_parser.add_argument("--max-tokens", type=int, default=500)
    chat_parser.add_argument("--max-input-tokens", type=int, default=1000)

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

    import sys

    if len(sys.argv) > 1 and sys.argv[1] in {"chat", "index"}:
        return chat_main(sys.argv[1:])

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
            summary_mode=args.summary_mode,
        )
        output_path = write_report(result, args.output_dir)
        print(f"Reviewed {result.chunks} selected chunk(s) in {result.mode} mode.")
        print(f"LLM calls: {result.llm_calls}")
        print(f"Report written to: {output_path}")
        return 0
    except (FileNotFoundError, ModuleNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1


def chat_main(argv: list[str]) -> int:
    parser = build_chat_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "index":
            chunks = build_index(
                args.paper_dir,
                args.index_path,
                chunk_chars=args.chunk_chars,
                overlap=args.overlap,
            )
            print(f"Indexed {len(chunks)} chunk(s).")
            print(f"Index written to: {args.index_path}")
            return 0

        if args.command == "chat":
            config = LLMConfig.from_env(
                model=args.model,
                base_url=args.base_url,
                api_key=args.api_key,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            agent = PaperAgent.create(
                config=config,
                paper_dir=args.paper_dir,
                index_path=args.index_path,
                output_dir=args.output_dir,
                max_input_tokens=args.max_input_tokens,
            )
            print(f"Using vLLM endpoint: {config.base_url}")
            print(f"Using model: {config.model}")
            print(f"Loaded {len(agent.tools.chunks)} indexed chunk(s).")
            print("Type exit, quit, or q to leave.")

            while True:
                try:
                    user_input = input("User> ").strip()
                except EOFError:
                    print()
                    return 0

                if user_input.lower() in {"exit", "quit", "q"}:
                    return 0
                if not user_input:
                    continue

                answer = agent.run(user_input)
                print(f"Agent> {answer}\n")

        parser.print_help()
        return 1
    except (FileNotFoundError, ModuleNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
