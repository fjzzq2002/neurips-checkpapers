"""
Re-annotate specific NeurIPS records and update the JSONL annotations file.

Example:
  python scripts/reannotate_ids.py --ids 116378 119515 --input data/analysis/neurips_2025_annotations_20251201_215608.jsonl --in-place
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from annotate_neurips_papers import (
    load_prompt_and_schema,
    load_records,
    load_env,
    annotate_records,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-annotate specific records and patch the JSONL file.")
    parser.add_argument("--ids", nargs="+", required=True, help="Record IDs to re-annotate.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/analysis/neurips_2025_annotated.jsonl"),
        help="Input JSONL annotations file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL file. If omitted and --in-place is set, overwrites input.",
    )
    parser.add_argument("--model", type=str, default=None, help="Override model (defaults to env/OpenRouter model).")
    parser.add_argument("--reasoning", type=str, default=None, help="Reasoning effort (env OPENROUTER_REASONING).")
    parser.add_argument("--concurrency", type=int, default=5, help="Concurrency for API calls.")
    parser.add_argument("--in-place", action="store_true", help="Overwrite the input file.")
    return parser.parse_args()


def main() -> None:
    load_env()
    args = parse_args()
    target_ids = {str(int(float(x))) for x in args.ids}

    prompt_text, schema = load_prompt_and_schema()
    all_records = load_records()
    records = [r for r in all_records if str(r.get("id")) in target_ids]
    if not records:
        raise SystemExit(f"No matching records found for IDs: {target_ids}")

    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    model = args.model or os.environ.get("OPENROUTER_MODEL") or "openai/gpt-oss-120b"
    reasoning = args.reasoning or os.environ.get("OPENROUTER_REASONING")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY (or OPENAI_API_KEY) is required.")

    annotations = asyncio.run(
        annotate_records(records, prompt_text, schema, model, reasoning, args.concurrency, api_key, base_url)
    )
    updated_map: Dict[str, Dict[str, object]] = {}
    for rec, (analysis, usage, used_model) in zip(records, annotations):
        rid = str(rec.get("id"))
        updated_map[rid] = {
            "analysis": analysis,
            "usage": usage,
            "model": used_model,
        }

    input_path: Path = args.input
    output_path: Path = args.output or (input_path if args.in_place else input_path.with_suffix(".patched.jsonl"))
    timestamp = datetime.now(timezone.utc).isoformat()

    lines_out: List[str] = []
    patched = set()
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            rec = json.loads(line)
            rid = rec.get("record_id")
            key = str(int(rid)) if rid is not None else None
            if key and key in updated_map:
                rec["analysis"] = updated_map[key]["analysis"]
                rec["usage"] = updated_map[key]["usage"]
                rec["model"] = updated_map[key]["model"]
                rec["timestamp"] = timestamp
                patched.add(key)
            lines_out.append(json.dumps(rec, ensure_ascii=False))

    missing = target_ids - patched
    if missing:
        raise SystemExit(f"Did not patch IDs: {missing}")

    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines_out) + "\n")

    print(f"Patched {len(patched)} records into {output_path}")


if __name__ == "__main__":
    main()
