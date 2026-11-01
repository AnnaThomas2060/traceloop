from enum import Enum

from pydantic import BaseModel, Field


class FaultType(str, Enum):
    TOOL_SELECTION_FAULT = "TOOL_SELECTION_FAULT"
    RETRIEVER_FAULT = "RETRIEVER_FAULT"
    GENERATOR_FAULT = "GENERATOR_FAULT"
    PASS = "PASS"


class EvalRecord(BaseModel):
    sample_id: str
    tool_used: str
    expected_tool: str
    context_recall: float = Field(..., ge=0.0, le=1.0)
    context_precision: float = Field(..., ge=0.0, le=1.0)
    faithfulness: float = Field(..., ge=0.0, le=1.0)
    answer_relevancy: float = Field(..., ge=0.0, le=1.0)
    retrieved_chunk_ids: list[str] | None = None
    expected_chunk_ids: list[str] | None = None


class FaultResult(BaseModel):
    sample_id: str
    fault_type: FaultType
    explanation: str
    primary_failing_metric: str | None = None


# Type aliases matching Part 2 specification
FaultClassification = FaultResult
EvalResultSample = EvalRecord


class FaultClassifier:
    """Classifies evaluation failures into tool, retriever, or generator faults."""

    def classify(self, record: EvalRecord) -> FaultResult:
        # 1. Tool Selection Fault
        if record.tool_used != record.expected_tool:
            return FaultResult(
                sample_id=record.sample_id,
                fault_type=FaultType.TOOL_SELECTION_FAULT,
                explanation=(
                    f"Tool selection mismatch: agent invoked '{record.tool_used}' "
                    f"but expected '{record.expected_tool}'."
                ),
                primary_failing_metric="tool_used",
            )

        # 2. Retriever Fault (applies when knowledge retriever is used)
        if record.tool_used == "knowledge_retriever_tool":
            if record.context_recall < 0.70 or record.context_precision < 0.65:
                reasons = []
                failing_metric = "context_recall" if record.context_recall < 0.70 else "context_precision"
                if record.context_recall < 0.70:
                    reasons.append(f"low context recall ({record.context_recall:.2f} < 0.70)")
                if record.context_precision < 0.65:
                    reasons.append(f"poor context precision ranking ({record.context_precision:.2f} < 0.65)")
                return FaultResult(
                    sample_id=record.sample_id,
                    fault_type=FaultType.RETRIEVER_FAULT,
                    explanation=f"Retriever defect: {', '.join(reasons)}.",
                    primary_failing_metric=failing_metric,
                )

        # 3. Generator Fault (applies when faithfulness or relevancy degrades)
        if record.faithfulness < 0.70 or record.answer_relevancy < 0.70:
            reasons = []
            failing_metric = "faithfulness" if record.faithfulness < 0.70 else "answer_relevancy"
            if record.faithfulness < 0.70:
                reasons.append(f"low faithfulness ({record.faithfulness:.2f} < 0.70)")
            if record.answer_relevancy < 0.70:
                reasons.append(f"low answer relevancy ({record.answer_relevancy:.2f} < 0.70)")
            return FaultResult(
                sample_id=record.sample_id,
                fault_type=FaultType.GENERATOR_FAULT,
                explanation=f"Generator breakdown: {', '.join(reasons)}.",
                primary_failing_metric=failing_metric,
            )

        # 4. Pass
        return FaultResult(
            sample_id=record.sample_id,
            fault_type=FaultType.PASS,
            explanation="Evaluation passed all metric and routing criteria.",
            primary_failing_metric=None,
        )

    def aggregate_faults(self, results: list[FaultResult]) -> dict[str, int]:
        counts = {
            FaultType.TOOL_SELECTION_FAULT.value: 0,
            FaultType.RETRIEVER_FAULT.value: 0,
            FaultType.GENERATOR_FAULT.value: 0,
            FaultType.PASS.value: 0,
        }
        for res in results:
            counts[res.fault_type.value] += 1
        return counts
