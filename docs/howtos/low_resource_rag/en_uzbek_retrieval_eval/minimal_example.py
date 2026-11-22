from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = {
    "id",
    "language",
    "domain",
    "question",
    "gold_answer",
    "source_doc_ids",
    "answerable",
    "cultural_specificity",
    "source_title",
    "difficulty",
    "quality_flag",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL row") from exc
            rows.append(row)
    return rows


def validate_rows(rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("No rows loaded")

    missing_by_id: dict[str, list[str]] = {}
    for index, row in enumerate(rows):
        row_id = str(row.get("id", f"row_{index}"))
        missing = sorted(REQUIRED_FIELDS.difference(row))
        if missing:
            missing_by_id[row_id] = missing

    if missing_by_id:
        details = "; ".join(f"{row_id}: {fields}" for row_id, fields in missing_by_id.items())
        raise ValueError(f"Rows missing required fields: {details}")


def to_ragas_preview(row: dict[str, Any]) -> dict[str, Any]:
    """Return a RAGAS-style record shape without claiming metrics are runnable yet."""

    return {
        "user_input": row["question"],
        "reference": row["gold_answer"],
        "retrieved_contexts": [],
        "response": "",
        "metadata": {
            "id": row["id"],
            "language": row["language"],
            "domain": row["domain"],
            "source_doc_ids": row["source_doc_ids"],
            "source_title": row["source_title"],
            "difficulty": row["difficulty"],
            "quality_flag": row["quality_flag"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate SOAS English-Uzbek evaluation rows and preview a RAGAS-style "
            "adapter shape. This does not run RAGAS metrics because retrieved contexts "
            "and generated answers are pipeline outputs, not dataset fields."
        )
    )
    parser.add_argument("--input", default="data/manual_eval_v5_sample.jsonl")
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()

    rows = load_jsonl(Path(args.input))
    validate_rows(rows)

    language_counts = Counter(row["language"] for row in rows)
    domain_counts = Counter(row["domain"] for row in rows)

    print(f"Loaded rows: {len(rows)}")
    print(f"Languages: {dict(sorted(language_counts.items()))}")
    print(f"Domains: {dict(sorted(domain_counts.items()))}")
    print("Preview records:")
    for row in rows[: max(args.limit, 0)]:
        print(json.dumps(to_ragas_preview(row), ensure_ascii=False))


if __name__ == "__main__":
    main()
