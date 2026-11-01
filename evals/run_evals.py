import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent import TraceLoopAgent
from app.config import get_settings
from evals.baseline_manager import BaselineManager
from evals.fault_classifier import EvalRecord, FaultClassifier, FaultResult
from evals.mlflow_logger import MLflowLogger
from evals.ragas_evaluator import DatasetEvaluationResult, EvalSample, RagasEvaluator


def load_dataset_cases(dataset_path: str) -> list[dict[str, Any]]:
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    cases: list[dict[str, Any]] = []
    if path.suffix == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    cases.append(json.loads(line))
    else:
        with open(path, "r", encoding="utf-8") as f:
            cases = json.load(f)
    return cases


def run_evaluation(
    dataset_path: str = "evals/golden_dataset.json",
    baseline_path: str = "evals/baseline.json",
    mock_judge: bool = True,
    update_baseline: bool = False,
    fail_on_regression: bool = True,
) -> int:
    cases = load_dataset_cases(dataset_path)
    agent = TraceLoopAgent()
    evaluator = RagasEvaluator(mock_judge=mock_judge)
    classifier = FaultClassifier()
    baseline_mgr = BaselineManager(baseline_file=baseline_path)
    mlflow_logger = MLflowLogger()

    eval_samples: list[EvalSample] = []
    agent_outputs: list[dict[str, Any]] = []

    for item in cases:
        qid = item.get("question_id", "q_unknown")
        question = item.get("question", "")
        ground_truth = item.get("ground_truth_answer", "")
        expected_tool = item.get("expected_tool", "knowledge_retriever_tool")
        expected_chunks = item.get("expected_chunk_ids", [])

        # Run agent
        res = agent.run(question)
        agent_outputs.append({
            "question_id": qid,
            "answer": res.answer,
            "tool_used": res.tool_used,
            "retrieved_chunk_ids": res.retrieved_chunk_ids,
        })

        eval_samples.append(
            EvalSample(
                question_id=qid,
                question=question,
                ground_truth_answer=ground_truth,
                expected_tool=expected_tool,
                expected_chunk_ids=expected_chunks,
                generated_answer=res.answer,
                tool_used=res.tool_used,
                retrieved_chunk_ids=res.retrieved_chunk_ids,
                retrieved_context=res.answer,
            )
        )

    # 1. Compute RAGAS metrics
    dataset_results: DatasetEvaluationResult = evaluator.evaluate_dataset(eval_samples)

    # 2. Classify faults
    fault_results: list[FaultResult] = []
    for sample, metrics in zip(eval_samples, dataset_results.sample_scores):
        record = EvalRecord(
            sample_id=sample.question_id,
            tool_used=sample.tool_used,
            expected_tool=sample.expected_tool,
            context_recall=metrics.context_recall,
            context_precision=metrics.context_precision,
            faithfulness=metrics.faithfulness,
            answer_relevancy=metrics.answer_relevancy,
            retrieved_chunk_ids=sample.retrieved_chunk_ids,
            expected_chunk_ids=sample.expected_chunk_ids,
        )
        fault_res = classifier.classify(record)
        fault_results.append(fault_res)

    fault_distribution = classifier.aggregate_faults(fault_results)

    current_metrics = {
        "faithfulness": dataset_results.mean_faithfulness,
        "context_precision": dataset_results.mean_context_precision,
        "context_recall": dataset_results.mean_context_recall,
        "answer_relevancy": dataset_results.mean_answer_relevancy,
    }

    # 3. Compare with Baseline
    comparison = baseline_mgr.compare(current_metrics)

    # 4. Log to MLflow
    settings = get_settings()
    params = {
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "chunk_size": 500,
        "chunk_overlap": 60,
        "model_name": settings.OPENAI_MODEL_NAME,
        "dataset_path": dataset_path,
        "total_samples": len(cases),
    }
    metrics_to_log = dict(current_metrics)
    metrics_to_log["delta_faithfulness"] = comparison.delta_faithfulness
    metrics_to_log["delta_context_precision"] = comparison.delta_context_precision
    metrics_to_log["total_faults"] = sum(
        v for k, v in fault_distribution.items() if k != "PASS"
    )

    artifacts = {
        "eval_scores.json": [s.model_dump() for s in dataset_results.sample_scores],
        "fault_summary.json": fault_distribution,
        "baseline_comparison.json": comparison.model_dump(),
    }
    mlflow_logger.log_evaluation_run("ragas-evaluation", params, metrics_to_log, artifacts)

    # 5. Emit summary table
    print("\nTraceLoop Continuous Evaluation Summary")
    print("========================================")
    print(f"{'Metric':<24} | {'Current':<8} | {'Baseline':<8} | {'Delta':<8}")
    print("---------------------------------------------------------------")
    for metric_name in ["faithfulness", "context_precision", "context_recall", "answer_relevancy"]:
        cur_val = current_metrics.get(metric_name, 0.0)
        base_val = comparison.baseline_metrics.get(metric_name, 0.0)
        delta_val = round(base_val - cur_val, 4)
        print(f"{metric_name:<24} | {cur_val:<8.4f} | {base_val:<8.4f} | {delta_val:<8.4f}")

    print("\nFault Attribution Distribution")
    print("------------------------------")
    for fault_key, count in fault_distribution.items():
        print(f"{fault_key:<24}: {count}")

    print(f"\nRegression Status: {comparison.status}")
    if comparison.regressed_metrics:
        print("Flagged Regressions:")
        for r in comparison.regressed_metrics:
            print(f"- {r}")

    if update_baseline:
        baseline_mgr.update_baseline(current_metrics)
        print("Baseline metrics successfully updated in evals/baseline.json.")

    if comparison.status == "REGRESSION_DETECTED" and fail_on_regression:
        print("\nCI Gate Failure: Evaluation detected regression exceeding tolerance threshold (0.05).")
        return 1

    return 0


def main():
    parser = argparse.ArgumentParser(description="TraceLoop evaluation runner")
    parser.add_argument("--dataset", default="evals/golden_dataset.json")
    parser.add_argument("--baseline", default="evals/baseline.json")
    parser.add_argument("--mock-judge", action="store_true", default=True)
    parser.add_argument("--update-baseline", action="store_true", default=False)
    parser.add_argument("--no-fail", action="store_true", default=False)

    args = parser.parse_args()

    exit_code = run_evaluation(
        dataset_path=args.dataset,
        baseline_path=args.baseline,
        mock_judge=args.mock_judge,
        update_baseline=args.update_baseline,
        fail_on_regression=not args.no_fail,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
