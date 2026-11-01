import json
from pathlib import Path


def test_golden_dataset_structure_and_count():
    path = Path("evals/golden_dataset.json")
    assert path.exists()

    with open(path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    assert len(cases) == 35

    expected_categories = {"memory", "evaluation", "vector_db", "observability", "conversational"}
    found_categories = set()

    for item in cases:
        assert "question_id" in item
        assert "question" in item
        assert "ground_truth_answer" in item
        assert "expected_tool" in item
        assert "expected_chunk_ids" in item
        assert "category" in item

        assert item["question_id"].startswith("gold_")
        assert len(item["question"].strip()) > 0
        assert len(item["ground_truth_answer"].strip()) > 0
        assert item["expected_tool"] in ["knowledge_retriever_tool", "fallback_direct_tool"]
        assert isinstance(item["expected_chunk_ids"], list)
        assert item["category"] in expected_categories
        found_categories.add(item["category"])

        if item["category"] == "conversational":
            assert item["expected_tool"] == "fallback_direct_tool"
            assert item["expected_chunk_ids"] == []
        else:
            assert item["expected_tool"] == "knowledge_retriever_tool"
            assert len(item["expected_chunk_ids"]) > 0

    assert found_categories == expected_categories
