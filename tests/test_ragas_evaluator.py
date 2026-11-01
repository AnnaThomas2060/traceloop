from evals.ragas_evaluator import DatasetEvaluationResult, EvalSample, RagasEvaluator


def test_evaluate_sample_grounded_match():
    evaluator = RagasEvaluator(mock_judge=True)
    sample = EvalSample(
        question_id="test_001",
        question="How does agent memory consolidation work?",
        ground_truth_answer="Memory consolidation transforms episodic interaction logs into semantic representations.",
        expected_tool="knowledge_retriever_tool",
        expected_chunk_ids=["chunk_1", "chunk_2"],
        generated_answer="Memory consolidation transforms episodic interaction logs into semantic representations.",
        tool_used="knowledge_retriever_tool",
        retrieved_chunk_ids=["chunk_1", "chunk_2", "chunk_3"],
        retrieved_context="Memory consolidation transforms episodic interaction logs into semantic representations.",
    )

    metrics = evaluator.evaluate_sample(sample)
    assert metrics.context_recall == 1.0
    assert metrics.context_precision == 1.0
    assert 0.70 <= metrics.faithfulness <= 1.0
    assert 0.0 <= metrics.answer_relevancy <= 1.0


def test_empty_retrieved_chunks_division_by_zero():
    evaluator = RagasEvaluator(mock_judge=True)
    sample = EvalSample(
        question_id="test_empty",
        question="What is the answer?",
        ground_truth_answer="Some ground truth.",
        expected_tool="knowledge_retriever_tool",
        expected_chunk_ids=["chunk_1"],
        generated_answer="No answer available.",
        tool_used="knowledge_retriever_tool",
        retrieved_chunk_ids=[],
        retrieved_context="",
    )

    metrics = evaluator.evaluate_sample(sample)
    assert metrics.context_recall == 0.0
    assert metrics.context_precision == 0.0
    assert 0.0 <= metrics.faithfulness <= 1.0
    assert 0.0 <= metrics.answer_relevancy <= 1.0


def test_conversational_sample_scores():
    evaluator = RagasEvaluator(mock_judge=True)
    sample = EvalSample(
        question_id="test_conv",
        question="Hello assistant!",
        ground_truth_answer="Hello, how can I assist you?",
        expected_tool="fallback_direct_tool",
        expected_chunk_ids=[],
        generated_answer="Hello, how can I assist you?",
        tool_used="fallback_direct_tool",
        retrieved_chunk_ids=[],
        retrieved_context="",
    )

    metrics = evaluator.evaluate_sample(sample)
    assert metrics.context_recall == 1.0
    assert metrics.context_precision == 1.0
    assert metrics.faithfulness == 1.0
    assert metrics.answer_relevancy == 1.0


def test_dataset_evaluation_aggregation():
    evaluator = RagasEvaluator(mock_judge=True)
    samples = [
        EvalSample(
            question_id=f"test_{i}",
            question=f"Question {i}",
            ground_truth_answer=f"Answer {i}",
            expected_tool="fallback_direct_tool",
            expected_chunk_ids=[],
            generated_answer=f"Answer {i}",
            tool_used="fallback_direct_tool",
            retrieved_chunk_ids=[],
            retrieved_context="",
        )
        for i in range(3)
    ]

    res = evaluator.evaluate_dataset(samples)
    assert isinstance(res, DatasetEvaluationResult)
    assert res.mean_context_recall == 1.0
    assert res.mean_context_precision == 1.0
    assert res.mean_faithfulness == 1.0
    assert res.mean_answer_relevancy == 1.0
    assert len(res.sample_scores) == 3


def test_mock_judge_flag_and_score_bounds():
    evaluator_mock = RagasEvaluator(mock_judge=True)
    assert evaluator_mock.mock_judge is True

    evaluator_real = RagasEvaluator(mock_judge=False)
    assert evaluator_real.mock_judge is False

    sample = EvalSample(
        question_id="test_bounds",
        question="What is Qdrant?",
        ground_truth_answer="Qdrant is an open-source vector search engine.",
        expected_tool="knowledge_retriever_tool",
        expected_chunk_ids=["chunk_a"],
        generated_answer="Qdrant provides vector indexing with HNSW graphs.",
        tool_used="knowledge_retriever_tool",
        retrieved_chunk_ids=["chunk_a"],
        retrieved_context="Qdrant is an open-source vector search engine with HNSW indexing.",
    )
    metrics = evaluator_mock.evaluate_sample(sample)
    assert 0.0 <= metrics.context_recall <= 1.0
    assert 0.0 <= metrics.context_precision <= 1.0
    assert 0.0 <= metrics.faithfulness <= 1.0
    assert 0.0 <= metrics.answer_relevancy <= 1.0
