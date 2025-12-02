"""
Scrape NeurIPS 2025 San Diego poster/oral metadata.

Fetches the event JSON the virtual site uses, filters to San Diego sessions,
and writes a compact JSON file with the key fields (title, authors, abstract,
session, and oral/spotlight flags).
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import requests


SOURCE_URL = "https://neurips.cc/static/virtual/data/neurips-2025-orals-posters.json"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


def fetch_events() -> List[Dict[str, Any]]:
    resp = requests.get(SOURCE_URL, headers={"User-Agent": USER_AGENT}, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    return payload.get("results", [])


def is_san_diego_event(entry: Dict[str, Any]) -> bool:
    session = (entry.get("session") or "").lower()
    event_type = (entry.get("event_type") or entry.get("eventtype") or "").lower()
    return "san diego" in session or "san diego" in event_type


def classify_presentation(entry: Dict[str, Any]) -> Dict[str, bool]:
    decision = entry.get("decision") or ""
    decision_lower = decision.lower()
    eventtype = (entry.get("eventtype") or "").lower()
    event_type = (entry.get("event_type") or "").lower()

    is_spotlight = (
        "spotlight" in decision_lower
        or "spotlight" in eventtype
        or "spotlight" in event_type
    )
    is_oral = (
        (
            "oral" in decision_lower
            and "poster" not in decision_lower
            and "spotlight" not in decision_lower
        )
        or eventtype == "oral"
        or (
            "oral" in event_type
            and "poster" not in event_type
            and "spotlight" not in event_type
        )
    )

    presentation = "poster"
    if is_oral:
        presentation = "oral"
    elif is_spotlight:
        presentation = "spotlight"

    # Recompute flags to avoid accidental oral from "oral poster".
    is_oral = presentation == "oral"
    is_spotlight = presentation == "spotlight"

    return {
        "is_oral": is_oral,
        "is_spotlight": is_spotlight,
        "is_oral_or_spotlight": is_oral or is_spotlight,
        "presentation": presentation,
    }


def normalize_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    decision = entry.get("decision") or ""
    presentation_flags = classify_presentation(entry)

    virtual_url = entry.get("virtualsite_url")
    if virtual_url and virtual_url.startswith("/"):
        virtual_url = f"https://neurips.cc{virtual_url}"

    eventtype_lower = (entry.get("eventtype") or "").lower()
    event_type_lower = (entry.get("event_type") or "").lower()

    session_kind = "poster"
    if "poster" in eventtype_lower or "poster" in event_type_lower:
        session_kind = "poster"
    elif presentation_flags["is_oral"]:
        session_kind = "oral"
    elif presentation_flags["is_spotlight"]:
        session_kind = "spotlight"

    session_record = None
    if entry.get("session") or entry.get("starttime") or entry.get("endtime"):
        session_record = {
            "name": entry.get("session"),
            "start": entry.get("starttime"),
            "end": entry.get("endtime"),
            "kind": session_kind,
            "poster_position": entry.get("poster_position"),
            "event_type": entry.get("event_type") or entry.get("eventtype"),
        }

    return {
        "id": entry.get("id"),
        "title": entry.get("name"),
        "authors": entry.get("authors") or [],
        "abstract": entry.get("abstract"),
        "topic": entry.get("topic"),
        "session": entry.get("session"),
        "sessions": [session_record] if session_record else [],
        "event_type": entry.get("event_type") or entry.get("eventtype"),
        "decision": decision,
        "poster_position": entry.get("poster_position"),
        "virtualsite_url": virtual_url,
        "paper_url": entry.get("paper_url"),
        "paper_pdf_url": entry.get("paper_pdf_url"),
        **presentation_flags,
    }


def transform(events: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    san_diego_events = [e for e in events if is_san_diego_event(e)]
    grouped: Dict[str, Dict[str, Any]] = {}

    for entry in san_diego_events:
        key = (entry.get("name") or "").strip().lower()
        normalized = normalize_entry(entry)

        if key not in grouped:
            grouped[key] = normalized
            grouped[key]["ids"] = [entry.get("id")]
            grouped[key]["decisions"] = [normalized.get("decision")]
        else:
            existing = grouped[key]
            existing["sessions"].extend(normalized["sessions"])
            existing["ids"].append(entry.get("id"))
            existing["decisions"].append(normalized.get("decision"))
            # Upgrade presentation flags if any session warrants it.
            for flag in ("is_oral", "is_spotlight", "is_oral_or_spotlight"):
                existing[flag] = existing.get(flag) or normalized.get(flag)
            # Highest priority: oral > spotlight > poster.
            if normalized["presentation"] == "oral" or (
                normalized["presentation"] == "spotlight"
                and existing.get("presentation") == "poster"
            ):
                existing["presentation"] = normalized["presentation"]

    # Normalize and sort session lists.
    records = []
    for rec in grouped.values():
        seen_sessions = []
        deduped_sessions = []
        for sess in rec["sessions"]:
            key = (sess.get("name"), sess.get("start"), sess.get("end"), sess.get("kind"))
            if key in seen_sessions:
                continue
            seen_sessions.append(key)
            deduped_sessions.append(sess)
        deduped_sessions.sort(key=lambda s: (s.get("start") or "", s.get("name") or ""))
        rec["sessions"] = deduped_sessions
        rec["ids"] = sorted([i for i in rec["ids"] if i is not None])
        rec["decisions"] = sorted({d for d in rec["decisions"] if d})
        records.append(rec)

    records.sort(
        key=lambda e: (
            e["sessions"][0].get("start") if e["sessions"] else "",
            e.get("title") or "",
        )
    )
    return records


def write_output(records: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "source_url": SOURCE_URL,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "count": len(records),
        "records": records,
    }
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scrape NeurIPS 2025 San Diego posters metadata.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/neurips_2025_san_diego_posters.json"),
        help="Where to write the JSON payload.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    events = fetch_events()
    records = transform(events)
    write_output(records, args.output)


if __name__ == "__main__":
    main()
