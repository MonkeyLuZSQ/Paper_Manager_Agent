from __future__ import annotations

import argparse
import json
import mimetypes
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    load_dotenv = None

from paper_agent.agent import PaperAgent
from paper_agent.llm_client import LLMConfig
from paper_agent.paper_store import DEFAULT_INDEX_PATH, build_index, list_supported_papers


ROOT_DIR = Path(__file__).resolve().parents[1]
STATIC_DIR = Path(__file__).resolve().parent / "web_static"
DEFAULT_PAPER_DIR = Path("paper_rep")
DEFAULT_OUTPUT_DIR = Path("outputs")


class WebState:
    def __init__(
        self,
        agent: PaperAgent,
        paper_dir: Path,
        output_dir: Path,
        index_path: Path,
        chunk_chars: int,
        overlap: int,
    ) -> None:
        self.agent = agent
        self.paper_dir = paper_dir
        self.output_dir = output_dir
        self.index_path = index_path
        self.chunk_chars = chunk_chars
        self.overlap = overlap

    def rebuild_index(self) -> int:
        chunks = build_index(
            self.paper_dir,
            self.index_path,
            chunk_chars=self.chunk_chars,
            overlap=self.overlap,
        )
        self.agent.tools.chunks = chunks
        return len(chunks)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Paper Manager web interface.")
    parser.add_argument("--host", default=os.getenv("WEB_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("WEB_PORT", "7860")))
    parser.add_argument("--paper-dir", type=Path, default=DEFAULT_PAPER_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--index-path", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--model", default=os.getenv("VLLM_MODEL", "qwen3-4b"))
    parser.add_argument("--base-url", default=os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--api-key", default=os.getenv("VLLM_API_KEY", "EMPTY"))
    parser.add_argument("--temperature", type=float, default=float(os.getenv("AGENT_TEMPERATURE", "0.2")))
    parser.add_argument("--max-tokens", type=int, default=int(os.getenv("AGENT_MAX_TOKENS", "500")))
    parser.add_argument("--max-input-tokens", type=int, default=int(os.getenv("AGENT_MAX_INPUT_TOKENS", "1000")))
    parser.add_argument("--chunk-chars", type=int, default=int(os.getenv("AGENT_INDEX_CHUNK_CHARS", "1800")))
    parser.add_argument("--overlap", type=int, default=int(os.getenv("AGENT_INDEX_OVERLAP", "180")))
    return parser


def create_handler(state: WebState):
    class PaperWebHandler(BaseHTTPRequestHandler):
        server_version = "PaperManagerWeb/0.1"

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send_static("index.html")
                return
            if parsed.path.startswith("/static/"):
                self._send_static(parsed.path.removeprefix("/static/"))
                return
            if parsed.path == "/api/state":
                self._send_json(self._state_payload())
                return
            if parsed.path == "/api/output":
                query = parse_qs(parsed.query)
                name = (query.get("name") or [""])[0]
                self._send_output(name)
                return
            self._send_error(HTTPStatus.NOT_FOUND, "Not found")

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/api/chat":
                payload = self._read_json()
                message = str(payload.get("message") or "").strip()
                if not message:
                    self._send_error(HTTPStatus.BAD_REQUEST, "message is required")
                    return
                try:
                    answer = state.agent.run(message)
                    self._send_json({"answer": answer, "active_paper": state.agent.active_paper_name})
                except Exception as exc:
                    self._send_json({"error": f"{type(exc).__name__}: {exc}"}, status=500)
                return
            if parsed.path == "/api/index":
                try:
                    count = state.rebuild_index()
                    self._send_json({"indexed_chunk_count": count})
                except Exception as exc:
                    self._send_json({"error": f"{type(exc).__name__}: {exc}"}, status=500)
                return
            self._send_error(HTTPStatus.NOT_FOUND, "Not found")

        def log_message(self, format: str, *args) -> None:
            print(f"[web] {self.address_string()} - {format % args}")

        def _state_payload(self) -> dict:
            papers = [path.name for path in list_supported_papers(state.paper_dir)]
            outputs = sorted(path.name for path in state.output_dir.glob("*.md")) if state.output_dir.exists() else []
            embedding_meta = ROOT_DIR / "data" / "embeddings" / "chunk_meta.json"
            embedding = {}
            if embedding_meta.exists():
                try:
                    meta = json.loads(embedding_meta.read_text(encoding="utf-8"))
                    embedding = {
                        "model": meta.get("embedding_model"),
                        "backend": meta.get("embedding_backend"),
                        "chunks": len(meta.get("chunks", [])),
                    }
                except json.JSONDecodeError:
                    embedding = {"error": "invalid chunk_meta.json"}
            return {
                "papers": papers,
                "outputs": outputs,
                "active_paper": state.agent.active_paper_name,
                "indexed_chunks": len(state.agent.tools.chunks),
                "embedding": embedding,
                "model": state.agent.llm.config.model,
                "base_url": state.agent.llm.config.base_url,
            }

        def _send_output(self, name: str) -> None:
            safe_name = Path(name).name
            path = state.output_dir / safe_name
            if not safe_name or not path.exists() or path.suffix.lower() != ".md":
                self._send_error(HTTPStatus.NOT_FOUND, "output not found")
                return
            self._send_json({"name": safe_name, "content": path.read_text(encoding="utf-8", errors="ignore")})

        def _send_static(self, relative_path: str) -> None:
            path = (STATIC_DIR / relative_path).resolve()
            if not str(path).startswith(str(STATIC_DIR.resolve())) or not path.exists() or not path.is_file():
                self._send_error(HTTPStatus.NOT_FOUND, "Not found")
                return
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            data = path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _read_json(self) -> dict:
            length = int(self.headers.get("Content-Length") or 0)
            if not length:
                return {}
            return json.loads(self.rfile.read(length).decode("utf-8"))

        def _send_json(self, payload: dict, status: int = 200) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _send_error(self, status: HTTPStatus, message: str) -> None:
            self._send_json({"error": message}, status=int(status))

    return PaperWebHandler


def main() -> int:
    if load_dotenv:
        load_dotenv()
    parser = build_parser()
    args = parser.parse_args()

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
    state = WebState(
        agent=agent,
        paper_dir=args.paper_dir,
        output_dir=args.output_dir,
        index_path=args.index_path,
        chunk_chars=args.chunk_chars,
        overlap=args.overlap,
    )
    server = ThreadingHTTPServer((args.host, args.port), create_handler(state))
    print(f"Paper Manager web UI: http://{args.host}:{args.port}")
    print(f"Using vLLM endpoint: {config.base_url}")
    print(f"Using model: {config.model}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
