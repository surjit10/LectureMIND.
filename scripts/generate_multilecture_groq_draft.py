"""Generate a source-grounded, multi-lecture evaluation draft with Groq.

This deliberately labels every record as auto-generated/pending human review.
It never invents chunk IDs: each generated answer is tied to the source chunk
included in its prompt.  The output is a draft for review, not release-grade
ground truth.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from local.llm.provider_registry import get_provider_registry


ROOT = Path(__file__).resolve().parents[1]


def _json_from_answer(text: str) -> list[dict[str, Any]]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I | re.S)
    parsed = json.loads(text)
    if not isinstance(parsed, list):
        raise ValueError("Groq response was not a JSON array")
    return parsed


def _chunk_prompt(chunks: list[dict[str, Any]], batch_no: int) -> str:
    source = []
    for c in chunks:
        transcript = " ".join(str(c.get("transcript", "")).split())[:1400]
        visual = " ".join(str(c.get("visual_context", "")).split())[:900]
        source.append({
            "chunk_id": c["chunk_id"],
            "timestamp": c.get("timestamp", 0),
            "transcript": transcript,
            "visual_context": visual,
        })
    return f"""You are preparing a retrieval evaluation draft (batch {batch_no}).
Use ONLY the supplied lecture evidence. Create exactly one distinct question for
each source chunk. Questions must be answerable from that chunk alone, and the
answer must be concise (one or two sentences). Include visual questions only when the
source has meaningful visual_context; otherwise use factual/conceptual questions.
Use difficulty values easy, medium, or hard. Use question_type values factual,
definition, conceptual, summary, or visual. Return ONLY valid JSON, an array of
objects with exactly these keys: query, ground_truth_answer, question_type,
difficulty, topic, keywords, need_visual. Do not include chunk_id; the caller
will attach the verified source chunk ID.

SOURCE EVIDENCE:
{json.dumps(source, ensure_ascii=False)}"""


def _normalise(item: dict[str, Any], chunk: dict[str, Any]) -> dict[str, Any]:
    query = str(item.get("query", "")).strip()
    answer = str(item.get("ground_truth_answer", "")).strip()
    if not query or not answer:
        raise ValueError("Groq produced an empty query or answer")
    qtype = str(item.get("question_type", "conceptual")).lower().strip()
    if qtype not in {"factual", "definition", "conceptual", "summary", "visual"}:
        qtype = "conceptual"
    difficulty = str(item.get("difficulty", "medium")).lower().strip()
    if difficulty not in {"easy", "medium", "hard"}:
        difficulty = "medium"
    keywords = item.get("keywords", [])
    if not isinstance(keywords, list):
        keywords = []
    return {
        "lecture_id": chunk["runtime_lecture_id"],
        "query": query,
        "ground_truth_answer": answer,
        "expected_chunk_ids": [chunk["chunk_id"]],
        "expected_route": "vector_only",
        "question_type": qtype,
        "difficulty": difficulty,
        "topic": str(item.get("topic", "lecture concept")).strip() or "lecture concept",
        "keywords": [str(k).strip() for k in keywords if str(k).strip()],
        "need_visual": bool(item.get("need_visual", False) or qtype == "visual"),
        "answerable": True,
        "review_status": "auto_generated_pending_human_review",
        "source_evidence": {
            "chunk_id": chunk["chunk_id"],
            "timestamp": chunk.get("timestamp", 0),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="evaluation/datasets/multilecture_groq_draft.json")
    parser.add_argument("--per-lecture", type=int, default=40)
    parser.add_argument("--lecture-ids", default="", help="Comma-separated registered lecture IDs to process")
    args = parser.parse_args()

    registry = json.loads((ROOT / "data/lecture_registry.json").read_text())
    # Exclude duplicate CS162 runtime packages: they point to the same lecture.
    selected = [
        (lid, entry) for lid, entry in registry["lectures"].items()
        if entry.get("status") == "READY" and lid != "lecture_05086e79"
    ]
    requested = {x.strip() for x in args.lecture_ids.split(",") if x.strip()}
    if requested:
        selected = [(lid, entry) for lid, entry in selected if lid in requested]
    backend = get_provider_registry().get_active_backend()
    records: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []

    for lecture_id, entry in selected:
        chunks = json.loads((Path(entry["package_path"]) / "multimodal_chunks.json").read_text())
        usable = [c for c in chunks if str(c.get("transcript", "")).strip()]
        stride = max(1, len(usable) // args.per_lecture)
        chosen = usable[::stride][: args.per_lecture]
        # Guarantee the requested count when integer stride rounding underfills.
        if len(chosen) < args.per_lecture:
            seen = {c["chunk_id"] for c in chosen}
            chosen.extend(c for c in usable if c["chunk_id"] not in seen)
            chosen = chosen[: args.per_lecture]
        for c in chosen:
            c["runtime_lecture_id"] = lecture_id

        lecture_records = []
        # Small batches avoid provider completion truncation while keeping each
        # request independently grounded in a compact evidence window.
        for offset in range(0, len(chosen), 5):
            batch = chosen[offset : offset + 5]
            prompt = _chunk_prompt(batch, offset // 5 + 1)
            result = backend.generate_with_metadata([
                {"role": "system", "content": "You produce strict JSON evaluation data."},
                {"role": "user", "content": prompt},
            ])
            generated = _json_from_answer(result.answer)
            if len(generated) != len(batch):
                raise ValueError(f"Expected {len(batch)} records, got {len(generated)} for {lecture_id}")
            lecture_records.extend(_normalise(item, chunk) for item, chunk in zip(generated, batch))
        records.extend(lecture_records)
        manifest.append({
            "lecture_id": lecture_id,
            "display_name": entry.get("display_name", ""),
            "record_count": len(lecture_records),
        })

    payload = {
        "dataset_name": "LectureMind multi-lecture Groq evaluation draft",
        "dataset_version": "draft-1",
        "label_status": "auto_generated_pending_human_review",
        "generation_provider": getattr(backend, "provider_name", "unknown"),
        "generation_model": getattr(backend, "_model", "unknown"),
        "lectures": manifest,
        "records": records,
    }
    # DatasetLoader consumes a top-level list; retain the manifest beside it.
    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n")
    manifest_path = out.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(out), "manifest": str(manifest_path), "records": len(records), "lectures": manifest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
