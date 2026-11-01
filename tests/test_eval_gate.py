import json

from evals.run_evals import run_evaluation


def test_eval_gate_passes_on_golden_dataset():
    # Running on default golden dataset should pass against baseline and return 0
    exit_code = run_evaluation(
        dataset_path="evals/golden_dataset.json",
        baseline_path="evals/baseline.json",
        mock_judge=True,
        fail_on_regression=True,
    )
    assert exit_code == 0


def test_eval_gate_fails_when_baseline_is_unreachable(tmp_path):
    # Create an artificially inflated baseline that forces a regression failure
    high_baseline_file = tmp_path / "high_baseline.json"
    high_baseline = {
        "faithfulness": 0.99,
        "context_precision": 0.99,
        "context_recall": 0.99,
        "answer_relevancy": 0.99,
    }
    high_baseline_file.write_text(json.dumps(high_baseline), encoding="utf-8")

    exit_code = run_evaluation(
        dataset_path="evals/golden_dataset.json",
        baseline_path=str(high_baseline_file),
        mock_judge=True,
        fail_on_regression=True,
    )
    assert exit_code == 1
