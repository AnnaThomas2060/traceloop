import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tracing.client import LangfuseClientWrapper


def compute_query_hash(query: str) -> str:
    cleaned = query.strip().lower()
    return hashlib.sha256(cleaned.encode("utf-8")).hexdigest()


def load_existing_hashes(file_path: Path) -> set[str]:
    hashes: set[str] = set()
    if not file_path.exists():
        return hashes

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                q_hash = record.get("query_hash")
                if q_hash:
                    hashes.add(q_hash)
            except Exception:
                pass
    return hashes


def parse_trace(trace: dict[str, Any]) -> dict[str, Any] | None:
    # Extract query
    input_data = trace.get("input", {})
    if isinstance(input_data, dict):
        query = input_data.get("query") or input_data.get("question") or input_data.get("prompt")
    elif isinstance(input_data, str):
        query = input_data
    else:
        query = None

    if not query or not isinstance(query, str) or not query.strip():
        return None

    query = query.strip()
    observations = trace.get("observations", [])

    tool_used = "knowledge_retriever_tool"
    retrieved_chunk_ids: list[str] = []
    retrieved_context = ""
    generated_answer = ""

    # Parse nested observations
    for obs in observations:
        obs_type = obs.get("type", "")
        obs_name = obs.get("name", "")

        if obs_type == "TOOL" or obs_name == "knowledge_retriever_tool":
            tool_used = "knowledge_retriever_tool"
            meta = obs.get("metadata", {})
            retrieved_chunk_ids = meta.get("retrieved_chunk_ids", [])
            retrieved_context = str(obs.get("output", ""))

        elif obs_name == "fallback_direct_tool":
            tool_used = "fallback_direct_tool"
            retrieved_chunk_ids = []
            retrieved_context = str(obs.get("output", ""))

        elif obs_type == "GENERATION" or obs_name == "response_synthesis":
            generated_answer = str(obs.get("output", ""))

    if not generated_answer:
        output_data = trace.get("output", {})
        if isinstance(output_data, dict):
            generated_answer = str(output_data.get("answer", output_data.get("output", "")))
        elif isinstance(output_data, str):
            generated_answer = output_data

    return {
        "query_hash": compute_query_hash(query),
        "question": query,
        "generated_answer": generated_answer,
        "tool_used": tool_used,
        "retrieved_chunk_ids": retrieved_chunk_ids,
        "retrieved_context": retrieved_context,
        "promoted_at": datetime.now(UTC).isoformat(),
        "trace_id": trace.get("id", ""),
    }


def sync_traces_to_dataset(
    output_path: str = "data/processed/promoted_traces.jsonl",
    score_threshold: float = 0.5,
    client: LangfuseClientWrapper | None = None,
) -> dict[str, int]:
    wrapper = client or LangfuseClientWrapper()
    dest_path = Path(output_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    existing_hashes = load_existing_hashes(dest_path)

    raw_negative = wrapper.get_traces_with_negative_feedback()
    raw_errors = wrapper.get_traces_with_errors()
    candidate_traces = raw_negative + raw_errors

    total_scanned = len(candidate_traces)
    candidate_count = 0
    new_appended = 0
    duplicates_skipped = 0

    new_records: list[dict[str, Any]] = []

    for trace in candidate_traces:
        candidate_count += 1
        parsed = parse_trace(trace)
        if parsed is None:
            continue

        q_hash = parsed["query_hash"]
        if q_hash in existing_hashes:
            duplicates_skipped += 1
            continue

        existing_hashes.add(q_hash)
        new_records.append(parsed)
        new_appended += 1

    if new_records:
        with open(dest_path, "a", encoding="utf-8") as f:
            f.writelines(json.dumps(rec) + "\n" for rec in new_records)

    print(
        f"Trace synchronization completed successfully. Scanned {total_scanned} production traces "
        f"and identified {candidate_count} candidates. Appended {new_appended} new regression cases "
        f"to {output_path} while skipping {duplicates_skipped} duplicate queries."
    )

    return {
        "total_scanned": total_scanned,
        "candidates": candidate_count,
        "new_appended": new_appended,
        "duplicates_skipped": duplicates_skipped,
    }


def main():
    parser = argparse.ArgumentParser(description="Synchronize Langfuse traces to promoted evaluation dataset.")
    parser.add_argument("--output", default="data/processed/promoted_traces.jsonl")
    parser.add_argument("--threshold", type=float, default=0.5)

    args = parser.parse_args()
    sync_traces_to_dataset(output_path=args.output, score_threshold=args.threshold)


if __name__ == "__main__":
    main()
