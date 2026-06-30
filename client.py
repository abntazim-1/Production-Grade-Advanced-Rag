"""
Interactive CLI client for the RAG API.
Usage: python client.py --url http://localhost:8000
Supports multi-turn conversation with streaming output.
"""
import argparse
import json
import sys
import uuid
import httpx


def stream_query(base_url: str, query: str, session_id: str) -> None:
    """Send a streaming query and print tokens as they arrive."""
    url = f"{base_url}/query/stream"
    payload = {"query": query, "session_id": session_id, "stream": True}

    print("\nAssistant: ", end="", flush=True)
    sources = []

    with httpx.stream("POST", url, json=payload, timeout=60) as resp:
        for line in resp.iter_lines():
            if not line or not line.startswith("data: "):
                continue
            data_str = line[len("data: "):]
            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            if "token" in data:
                print(data["token"], end="", flush=True)

            if data.get("done"):
                sources = data.get("sources", [])

    print("\n")

    if sources:
        print("Sources:")
        for s in sources[:3]:
            src = s.get("source", "unknown")
            heading = s.get("heading") or ""
            score = s.get("rerank_score", 0)
            label = f"  [{s.get('id', '?')}] {src}"
            if heading:
                label += f" › {heading}"
            label += f"  (score: {score:.3f})"
            print(label)
    print()


def sync_query(base_url: str, query: str, session_id: str) -> None:
    """Non-streaming query — prints full response."""
    url = f"{base_url}/query"
    payload = {"query": query, "session_id": session_id}

    resp = httpx.post(url, json=payload, timeout=60)
    if resp.status_code != 200:
        print(f"Error {resp.status_code}: {resp.text}")
        return

    data = resp.json()
    print(f"\nAssistant: {data['answer']}\n")

    sources = data.get("sources", [])
    if sources:
        print("Sources:")
        for s in sources[:3]:
            src = s.get("source", "unknown")
            heading = s.get("heading") or ""
            score = s.get("rerank_score", 0)
            label = f"  [{s.get('id', '?')}] {src}"
            if heading:
                label += f" › {heading}"
            label += f"  (score: {score:.3f})"
            print(label)

    faith = data.get("faithfulness_score")
    if faith is not None:
        print(f"\n  Faithfulness: {faith:.3f}")
    print()


def main():
    parser = argparse.ArgumentParser(description="RAG System CLI Client")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--stream", action="store_true", default=True)
    parser.add_argument("--session", default=None)
    args = parser.parse_args()

    session_id = args.session or str(uuid.uuid4())
    base_url = args.url.rstrip("/")

    # Health check
    try:
        resp = httpx.get(f"{base_url}/health", timeout=5)
        health = resp.json()
        print(f"✅ Connected to RAG API | {health.get('chunks_indexed', 0)} chunks indexed")
        print(f"   Session: {session_id}\n")
    except Exception as e:
        print(f"❌ Cannot connect to {base_url}: {e}")
        sys.exit(1)

    print("Type your question (or 'quit' to exit, 'clear' to reset session)\n")

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not query:
            continue

        if query.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        if query.lower() == "clear":
            session_id = str(uuid.uuid4())
            print(f"Session reset. New session: {session_id}\n")
            continue

        try:
            if args.stream:
                stream_query(base_url, query, session_id)
            else:
                sync_query(base_url, query, session_id)
        except httpx.ReadTimeout:
            print("⏱️  Request timed out. The model may be loading.\n")
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()
