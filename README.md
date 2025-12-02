## NeurIPS 2025 San Diego Posters – Scrape & Annotation Toolkit

This repo holds two main workflows:

1) **Scraping** NeurIPS 2025 San Diego posters/orals metadata from the official virtual data feed.
2) **Annotating** papers with a GPT pipeline (OpenRouter) and exporting to CSV/XLSX.

Everything runs locally; API calls only happen when you pass `--run`.

---

### Key Artifacts

- `data/neurips_2025_san_diego_posters.json` — scraped poster/oral metadata (with merged sessions).
- `data/analysis/neurips_2025_annotated.jsonl` — current model annotations (one JSON per paper). New runs from the annotator are timestamped; rename the latest to this path to keep things consistent.
- `example_workbook.xlsx` — formatted workbook with summaries, scores, flags, and abstracts.

---

### Environment

Use the provided Python (e.g., `/Users/ziqianz/.pyenv/shims/python`). For OpenRouter calls:
```
OPENROUTER_API_KEY=...
OPENROUTER_MODEL=openai/gpt-oss-120b   # optional override
OPENROUTER_REASONING=medium            # optional
OPENROUTER_CONCURRENCY=5               # optional
```
Put these in `.env` (not checked in).

---

### Scrape NeurIPS metadata

```
python scripts/neurips_2025_san_diego_scraper.py --output data/neurips_2025_san_diego_posters.json
```

Outputs merged records with sessions (oral/spotlight/poster), poster positions, links, topic, and flags.

---

### Annotate papers (OpenRouter prompt)

Dry run preview (no cost):
```
python scripts/annotate_neurips_papers.py --limit 3
```

Real run (calls OpenRouter):
```
python scripts/annotate_neurips_papers.py --limit 100 --run
```
Options: `--offset`, `--model`, `--reasoning`, `--concurrency`, `--output` (defaults to a timestamped JSONL in `data/analysis/`).

Retries are enabled; too-short summaries trigger re-tries.

Re-annotate specific IDs and patch the main JSONL:
```
python scripts/reannotate_ids.py --ids 116378 119515 --input data/analysis/neurips_2025_annotated.jsonl --in-place
```

---

### Export to CSV (weighted score)

```
python scripts/export_annotations_csv.py --safety-weight 2.5 \
  --input data/analysis/neurips_2025_annotated.jsonl \
  --output data/analysis/neurips_2025_annotations_sorted.csv
```

Fields include IDs, title, authors, decision/presentation, topic, sessions, poster #, summaries (EN/中文), scores, categories, verdict flags, and weighted total (`interp + understanding + safety * weight`). Sorted by session, then weighted score.

---

### Build formatted workbook

```
python scripts/build_example_workbook.py
```

Produces `example_workbook.xlsx` with:
- Columns: ID, Title, Authors, Decision, Topic, Session, Poster #, Safety?, MI?, Link (NeurIPS virtual), Summary, Summary (CN), Abstract, Focus, Failure Mode, scores, Keywords, Weighted Score.
- Sorting: Session asc, then weighted score desc (safety weight 2.5).
- Styling: header bold; row fills (Safety-only green, MI-only orange, both pink); oral/spotlight IDs bold.

---

### Notes
- The prompt and schema are copied from the previous pipeline (see `prompts/`).
- All network/API calls are gated by `--run`; scraping and exports are offline.
- If you regenerate annotations, rerun CSV and workbook steps to stay in sync.
