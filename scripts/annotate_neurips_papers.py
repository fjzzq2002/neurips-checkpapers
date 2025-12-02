"""
Annotate NeurIPS 2025 San Diego papers with the Safety-RSS GPT pipeline (OpenRouter).

This is adapted to the locally scraped dataset and stays dry-run by default to avoid
incurring model costs. To actually run annotations, pass --run after configuring
your .env with OPENROUTER_API_KEY (and optionally OPENROUTER_MODEL, etc.).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from openai import AsyncOpenAI

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data/neurips_2025_san_diego_posters.json"
PROMPT_PATH = ROOT / "prompts/prompt.txt"
SCHEMA_PATH = ROOT / "prompts/schema.json"
DEFAULT_OUTPUT = ROOT / f"data/analysis/neurips_2025_annotations_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.jsonl"
MAX_SUMMARY_ATTEMPTS = int(os.environ.get("ANNOTATION_MAX_ATTEMPTS", "10"))
SUMMARY_RETRY_DELAY_SECONDS = float(os.environ.get("ANNOTATION_RETRY_DELAY", "1.0"))
MIN_SUMMARY_LEN = int(os.environ.get("ANNOTATION_MIN_SUMMARY_LEN", "25"))
MIN_SUMMARY_CN_LEN = int(os.environ.get("ANNOTATION_MIN_SUMMARY_CN_LEN", "20"))


def load_env(dotenv_path: Path = ROOT / ".env") -> None:
    """Lightweight .env loader (no external dependency)."""
    if not dotenv_path.exists():
        return
    for line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def load_prompt_and_schema() -> Tuple[str, Dict[str, Any]]:
    prompt_text = PROMPT_PATH.read_text(encoding="utf-8")
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    return prompt_text, schema


def load_records(path: Path = DATA_PATH) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("records", [])


def usage_to_dict(usage_obj: Any) -> Dict[str, Any]:
    if usage_obj is None:
        return {}
    if hasattr(usage_obj, "model_dump"):
        return usage_obj.model_dump()
    if isinstance(usage_obj, dict):
        return usage_obj
    return getattr(usage_obj, "__dict__", {}) or {}


def format_sessions(sessions: Iterable[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for sess in sessions:
        name = sess.get("name") or "Unknown session"
        kind = sess.get("kind") or "poster"
        start = sess.get("start") or "unknown start"
        end = sess.get("end") or "unknown end"
        position = sess.get("poster_position") or ""
        position_str = f" | poster position: {position}" if position else ""
        lines.append(f"- {kind} | {name} ({start} to {end}){position_str}")
    return "\n".join(lines) if lines else "- None provided."


def format_authors(authors: Iterable[Any]) -> str:
    names: List[str] = []
    for author in authors:
        if isinstance(author, str):
            names.append(author)
        elif isinstance(author, dict):
            name = author.get("fullname") or author.get("name")
            if name:
                names.append(str(name))
    if not names:
        return "Unknown authors"
    return ", ".join(names)


def parse_json_content(message_obj: Any) -> Dict[str, Any]:
    """Best-effort parser for OpenRouter structured responses."""
    if message_obj is None:
        raise ValueError("Empty response received from OpenRouter.")

    parsed = getattr(message_obj, "parsed", None)
    if parsed:
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, str):
            try:
                return json.loads(parsed)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Failed to parse JSON response: {exc}") from exc

    content = getattr(message_obj, "content", message_obj)
    content_str = ""
    if isinstance(content, str):
        content_str = content.strip()
    elif isinstance(content, list):
        content_str = "".join(
            (part.get("text", "") if isinstance(part, dict) else str(part))
            for part in content
        ).strip()
    elif isinstance(content, dict):
        content_str = json.dumps(content)

    if not content_str:
        raise ValueError("Empty response received from OpenRouter.")

    try:
        return json.loads(content_str)
    except json.JSONDecodeError as exc:
        first = content_str.find("{")
        last = content_str.rfind("}")
        if first != -1 and last != -1 and last > first:
            try:
                return json.loads(content_str[first : last + 1])
            except json.JSONDecodeError:
                pass
        starts = [i for i, ch in enumerate(content_str) if ch == "{"]
        ends = [i for i, ch in enumerate(content_str) if ch == "}"]
        for s in starts:
            for e in reversed(ends):
                if e > s:
                    try:
                        return json.loads(content_str[s : e + 1])
                    except json.JSONDecodeError:
                        continue
        raise ValueError(f"Failed to parse JSON response: {exc}: {content_str[:200]}") from exc


def _clean_len(text: str) -> int:
    if not isinstance(text, str):
        return 0
    stripped = (
        text.replace(" ", "")
        .replace("\t", "")
        .replace("\n", "")
        .replace("\r", "")
        .replace(".", "")
    )
    return len(stripped)


def build_article_input(record: Dict[str, Any]) -> str:
    parts = [
        f"Title: {record.get('title') or 'Unknown title'}",
        f"Abstract:\n{record.get('abstract') or 'No abstract provided.'}",
    ]
    return "\n".join(parts)


async def annotate_record(
    client: AsyncOpenAI,
    prompt_text: str,
    schema: Dict[str, Any],
    record: Dict[str, Any],
    model: str,
    reasoning_effort: Optional[str],
) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    messages = [
        {"role": "system", "content": prompt_text},
        {"role": "user", "content": build_article_input(record)},
    ]
    request_payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "response_format": {"type": "json_schema", "json_schema": schema},
        "extra_body": {"usage": {"include": True}},
    }
    if reasoning_effort:
        request_payload["reasoning_effort"] = reasoning_effort

    last_error: Optional[Exception] = None
    for attempt in range(1, MAX_SUMMARY_ATTEMPTS + 1):
        try:
            completion = await client.chat.completions.create(**request_payload)
            choice = completion.choices[0]
            payload = parse_json_content(choice.message)
            usage = usage_to_dict(getattr(completion, "usage", None))
            used_model = getattr(completion, "model", model)

            summary_len = _clean_len(str(payload.get("summary", "")))
            summary_cn_len = _clean_len(str(payload.get("summary_cn", "")))
            if summary_len < MIN_SUMMARY_LEN and attempt < MAX_SUMMARY_ATTEMPTS:
                raise ValueError(f"Summary too short (len={summary_len})")
            if summary_cn_len < MIN_SUMMARY_CN_LEN and attempt < MAX_SUMMARY_ATTEMPTS:
                raise ValueError(f"Summary CN too short (len={summary_cn_len})")

            return payload, usage, used_model
        except Exception as exc:
            print(f"Error annotating record {record.get('id')}: {exc}")
            last_error = exc
        if attempt == MAX_SUMMARY_ATTEMPTS:
            raise last_error  # type: ignore[misc]
        await asyncio.sleep(SUMMARY_RETRY_DELAY_SECONDS * attempt)


async def annotate_records(
    records: List[Dict[str, Any]],
    prompt_text: str,
    schema: Dict[str, Any],
    model: str,
    reasoning_effort: Optional[str],
    concurrency: int,
    api_key: str,
    base_url: str,
) -> List[Tuple[Dict[str, Any], Dict[str, Any], str]]:
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    semaphore = asyncio.Semaphore(max(1, concurrency))
    results: List[Optional[Tuple[Dict[str, Any], Dict[str, Any], str]]] = [None] * len(records)
    total = len(records)
    completed = 0

    async def run(idx: int, record: Dict[str, Any]) -> None:
        nonlocal completed
        async with semaphore:
            annotation, usage, used_model = await annotate_record(
                client, prompt_text, schema, record, model, reasoning_effort
            )
            results[idx] = (annotation, usage, used_model)
            completed += 1
            print(f"Progress: {completed}/{total}", end="\r", flush=True)

    await asyncio.gather(*[run(idx, rec) for idx, rec in enumerate(records)])
    if total:
        print(f"Progress: {total}/{total}")
    return [r for r in results if r is not None]


def write_results(
    records: List[Dict[str, Any]],
    annotations: List[Tuple[Dict[str, Any], Dict[str, Any], str]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    with output_path.open("a", encoding="utf-8") as handle:
        for record, (analysis, usage, used_model) in zip(records, annotations):
            handle.write(
                json.dumps(
                    {
                        "timestamp": now,
                        "record_id": record.get("id"),
                        "title": record.get("title"),
                        "presentation": record.get("presentation"),
                        "decision": record.get("decision"),
                        "topic": record.get("topic"),
                        "sessions": record.get("sessions"),
                        "paper_url": record.get("paper_url"),
                        "virtualsite_url": record.get("virtualsite_url"),
                        "analysis": analysis,
                        "usage": usage,
                        "model": used_model,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Annotate NeurIPS 2025 San Diego papers using the Safety-RSS GPT pipeline."
    )
    parser.add_argument("--limit", type=int, default=5, help="How many papers to annotate (default: 5).")
    parser.add_argument("--offset", type=int, default=0, help="Skip this many records before annotating.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Where to append JSONL analysis records.",
    )
    parser.add_argument("--model", type=str, default=None, help="Override model (defaults to OPENROUTER_MODEL or gpt-oss).")
    parser.add_argument("--reasoning", type=str, default=None, help="Set OpenRouter reasoning_effort (e.g., medium).")
    parser.add_argument("--concurrency", type=int, default=None, help="Number of concurrent API calls.")
    parser.add_argument(
        "--run",
        action="store_true",
        help="Actually call OpenRouter. Without this flag, the script stays dry-run and shows a preview.",
    )
    return parser.parse_args()


def main() -> None:
    load_env()
    args = parse_args()
    prompt_text, schema = load_prompt_and_schema()
    records = load_records()

    if args.offset:
        records = records[args.offset :]
    if args.limit is not None:
        records = records[: args.limit]

    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    model = args.model or os.environ.get("OPENROUTER_MODEL") or "openai/gpt-oss-120b"
    reasoning_effort = args.reasoning or os.environ.get("OPENROUTER_REASONING")
    concurrency = args.concurrency or int(os.environ.get("OPENROUTER_CONCURRENCY", "20"))

    if not args.run:
        sample = records[0] if records else {}
        preview_input = build_article_input(sample) if sample else "(no records)"
        print("Dry run: not calling OpenRouter. Preview of first prompt:\n")
        print(preview_input)
        print(
            "\nTo annotate, set OPENROUTER_API_KEY in .env and rerun with --run "
            f"(limit={args.limit}, model={model}, concurrency={concurrency})."
        )
        return

    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY (or OPENAI_API_KEY) is required to run annotations.")

    print(f"Annotating {len(records)} records with model={model}, concurrency={concurrency}")
    annotations = asyncio.run(
        annotate_records(records, prompt_text, schema, model, reasoning_effort, concurrency, api_key, base_url)
    )
    write_results(records, annotations, args.output)
    print(f"Wrote {len(annotations)} annotations to {args.output}")


if __name__ == "__main__":
    main()
