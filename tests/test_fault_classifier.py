from evals.fault_classifier import (
    EvalRecord,
    EvalResultSample,
    FaultClassification,
    FaultClassifier,
    FaultType,
)


def test_tool_selection_fault():
    classifier = FaultClassifier()
    record = EvalResultSample(
        sample_id="case_tool_fail",
        tool_used="fallback_direct_tool",
        expected_tool="knowledge_retriever_tool",
        context_recall=1.0,
        context_precision=1.0,
        faithfulness=1.0,
        answer_relevancy=1.0,
    )
    result = classifier.classify(record)
    assert isinstance(result, FaultClassification)
    assert result.fault_type == FaultType.TOOL_SELECTION_FAULT
    assert result.primary_failing_metric == "tool_used"
    assert "mismatch" in result.explanation


def test_retriever_fault_low_recall():
    classifier = FaultClassifier()
    record = EvalRecord(
        sample_id="case_retriever_recall",
        tool_used="knowledge_retriever_tool",
        expected_tool="knowledge_retriever_tool",
        context_recall=0.40,
        context_precision=0.80,
        faithfulness=0.90,
        answer_relevancy=0.90,
    )
    result = classifier.classify(record)
    assert result.fault_type == FaultType.RETRIEVER_FAULT
    assert result.primary_failing_metric == "context_recall"
    assert "recall" in result.explanation


def test_retriever_fault_low_precision():
    classifier = FaultClassifier()
    record = EvalRecord(
        sample_id="case_retriever_precision",
        tool_used="knowledge_retriever_tool",
        expected_tool="knowledge_retriever_tool",
        context_recall=0.85,
        context_precision=0.55,
        faithfulness=0.90,
        answer_relevancy=0.90,
    )
    result = classifier.classify(record)
    assert result.fault_type == FaultType.RETRIEVER_FAULT
    assert "precision" in result.explanation


def test_generator_fault_low_faithfulness():
    classifier = FaultClassifier()
    record = EvalRecord(
        sample_id="case_gen_faith",
        tool_used="knowledge_retriever_tool",
        expected_tool="knowledge_retriever_tool",
        context_recall=0.90,
        context_precision=0.85,
        faithfulness=0.50,
        answer_relevancy=0.85,
    )
    result = classifier.classify(record)
    assert result.fault_type == FaultType.GENERATOR_FAULT
    assert "faithfulness" in result.explanation


def test_generator_fault_low_relevancy():
    classifier = FaultClassifier()
    record = EvalRecord(
        sample_id="case_gen_relevancy",
        tool_used="knowledge_retriever_tool",
        expected_tool="knowledge_retriever_tool",
        context_recall=0.90,
        context_precision=0.85,
        faithfulness=0.95,
        answer_relevancy=0.55,
    )
    result = classifier.classify(record)
    assert result.fault_type == FaultType.GENERATOR_FAULT
    assert "relevancy" in result.explanation


def test_passing_record_and_aggregation():
    classifier = FaultClassifier()
    pass_record = EvalRecord(
        sample_id="case_pass",
        tool_used="knowledge_retriever_tool",
        expected_tool="knowledge_retriever_tool",
        context_recall=0.85,
        context_precision=0.80,
        faithfulness=0.90,
        answer_relevancy=0.88,
    )
    res_pass = classifier.classify(pass_record)
    assert res_pass.fault_type == FaultType.PASS

    fail_record = EvalRecord(
        sample_id="case_fail",
        tool_used="fallback_direct_tool",
        expected_tool="knowledge_retriever_tool",
        context_recall=0.0,
        context_precision=0.0,
        faithfulness=1.0,
        answer_relevancy=1.0,
    )
    res_fail = classifier.classify(fail_record)

    summary = classifier.aggregate_faults([res_pass, res_fail])
    assert summary[FaultType.PASS.value] == 1
    assert summary[FaultType.TOOL_SELECTION_FAULT.value] == 1
    assert summary[FaultType.RETRIEVER_FAULT.value] == 0
    assert summary[FaultType.GENERATOR_FAULT.value] == 0
