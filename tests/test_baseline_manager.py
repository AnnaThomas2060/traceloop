import json

from evals.baseline_manager import BaselineManager, RegressionReport


def test_passing_comparison(tmp_path):
    baseline_file = tmp_path / "baseline.json"
    baseline_data = {
        "faithfulness": 0.88,
        "context_precision": 0.85,
        "context_recall": 0.82,
        "answer_relevancy": 0.84,
    }
    baseline_file.write_text(json.dumps(baseline_data), encoding="utf-8")

    mgr = BaselineManager(baseline_file=str(baseline_file), regression_threshold=0.05)

    # Current metrics close to or better than baseline
    current = {
        "faithfulness": 0.86,  # delta = 0.02 (<= 0.05, pass)
        "context_precision": 0.87,  # delta = -0.02 (improved, pass)
        "context_recall": 0.83,
        "answer_relevancy": 0.85,
    }
    res = mgr.compare(current)
    assert res.status == "PASSED"
    assert len(res.regressed_metrics) == 0


def test_regression_detection_over_threshold(tmp_path):
    baseline_file = tmp_path / "baseline.json"
    baseline_data = {
        "faithfulness": 0.88,
        "context_precision": 0.85,
    }
    baseline_file.write_text(json.dumps(baseline_data), encoding="utf-8")

    mgr = BaselineManager(baseline_file=str(baseline_file), regression_threshold=0.05)

    # Significant drop in context precision (0.85 -> 0.58, delta = 0.27)
    regressed_current = {
        "faithfulness": 0.87,
        "context_precision": 0.58,
    }
    res = mgr.compare(regressed_current)
    assert res.status == "REGRESSION_DETECTED"
    assert any("context_precision" in m for m in res.regressed_metrics)
    assert res.delta_context_precision == 0.27


def test_update_baseline(tmp_path):
    baseline_file = tmp_path / "baseline_update.json"
    mgr = BaselineManager(baseline_file=str(baseline_file))

    new_metrics = {
        "faithfulness": 0.92,
        "context_precision": 0.90,
        "context_recall": 0.88,
        "answer_relevancy": 0.89,
    }
    mgr.update_baseline(new_metrics)

    loaded = mgr.get_baseline()
    assert loaded["faithfulness"] == 0.92
    assert loaded["context_precision"] == 0.90
    assert "created_at" in loaded


def test_check_regression_method_and_report_alias(tmp_path):
    baseline_file = tmp_path / "baseline_check.json"
    baseline_data = {
        "faithfulness": 0.88,
        "context_precision": 0.85,
    }
    baseline_file.write_text(json.dumps(baseline_data), encoding="utf-8")

    mgr = BaselineManager(baseline_file=str(baseline_file))
    report = mgr.check_regression({"faithfulness": 0.85, "context_precision": 0.85}, threshold=0.05)
    assert isinstance(report, RegressionReport)
    assert report.status == "PASSED"

    report_fail = mgr.check_regression({"faithfulness": 0.70, "context_precision": 0.85}, threshold=0.05)
    assert report_fail.status == "REGRESSION_DETECTED"

