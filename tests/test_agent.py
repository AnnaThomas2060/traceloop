from app.agent import AgentExecutionResult, TraceLoopAgent
from app.qdrant_client import SearchResult
from app.tools import FallbackDirectTool, KnowledgeRetrieverTool


def test_conversational_routing():
    agent = TraceLoopAgent()
    res = agent.run("Hello there, what can you do?")

    assert isinstance(res, AgentExecutionResult)
    assert res.tool_used == "fallback_direct_tool"
    assert "no retrieval" in res.answer.lower()
    assert res.retrieved_chunk_ids == []
    assert res.trace_id
    assert "traces" in res.trace_url


def test_technical_query_routing_with_citations():
    class MockRetrieverClient:
        def search(self, query_vector, top_k, score_threshold):
            return [
                SearchResult(
                    chunk_id="doc_001_chunk_0000",
                    score=0.89,
                    text="Agent memory consolidation is modeled after cognitive processes.",
                    doc_id="doc_001",
                    title="Agent Memory Consolidation",
                    source_url="https://docs.traceloop.dev/memory",
                )
            ]

    retriever_tool = KnowledgeRetrieverTool(vector_client=MockRetrieverClient())
    agent = TraceLoopAgent(retriever_tool=retriever_tool)
    res = agent.run("Explain agent memory consolidation mechanisms")

    assert res.tool_used == "knowledge_retriever_tool"
    assert "doc_001_chunk_0000" in res.retrieved_chunk_ids
    assert "[Chunk: doc_001_chunk_0000]" in res.answer
    assert "Agent memory consolidation" in res.answer


def test_insufficient_context_handling():
    class EmptyRetrieverClient:
        def search(self, query_vector, top_k, score_threshold):
            return []

    retriever_tool = KnowledgeRetrieverTool(vector_client=EmptyRetrieverClient())
    agent = TraceLoopAgent(retriever_tool=retriever_tool)
    res = agent.run("What is quantum entanglement in black holes?")

    assert res.tool_used == "knowledge_retriever_tool"
    assert res.retrieved_chunk_ids == []
    assert "insufficient" in res.answer.lower()


def test_nested_spans_in_trace_tree():
    class MockRetrieverClient:
        def search(self, query_vector, top_k, score_threshold):
            return [
                SearchResult(
                    chunk_id="doc_004_chunk_0001",
                    score=0.91,
                    text="Traces and evaluation metrics must be coupled.",
                    doc_id="doc_004",
                    title="LLM Observability",
                    source_url="https://docs.traceloop.dev/observability",
                )
            ]

    retriever_tool = KnowledgeRetrieverTool(vector_client=MockRetrieverClient())
    agent = TraceLoopAgent(retriever_tool=retriever_tool)
    query = "How does LLM observability integrate with evaluation?"
    session_id = "test_sess_42"
    res = agent.run(query=query, session_id=session_id)

    # 1. Verify root trace
    tree = agent.get_last_trace()
    assert tree is not None
    assert tree["id"] == res.trace_id
    assert tree["user_id"] == "traceloop_user"
    assert tree["session_id"] == session_id
    assert tree["input"]["query"] == query
    assert tree["output"]["answer"] == res.answer

    observations = tree["observations"]
    assert len(observations) == 3

    # 2. Verify agent planning span
    agent_span = observations[0]
    assert agent_span["type"] == "SPAN"
    assert agent_span["name"] == "agent_plan"
    assert agent_span["input"]["query"] == query
    assert agent_span["output"]["tool_used"] == "knowledge_retriever_tool"
    assert agent_span["metadata"]["max_iterations"] == 5

    # 3. Verify tool execution span
    tool_span = observations[1]
    assert tool_span["type"] == "TOOL"
    assert tool_span["name"] == "knowledge_retriever_tool"
    assert tool_span["input"]["query"] == query
    assert "retrieved_chunk_ids" in tool_span["metadata"]
    assert "doc_004_chunk_0001" in tool_span["metadata"]["retrieved_chunk_ids"]
    assert tool_span["metadata"]["retrieval_used"] is True

    # 4. Verify generation span
    gen_span = observations[2]
    assert gen_span["type"] == "GENERATION"
    assert gen_span["name"] == "response_synthesis"
    assert gen_span["model"] == agent.settings.OPENAI_MODEL_NAME
    assert gen_span["output"] == res.answer
    assert "token_count" in gen_span
    assert gen_span["token_count"]["total_tokens"] > 0


def test_agent_executor_properties():
    agent = TraceLoopAgent()
    assert agent.max_iterations == 5
    assert len(agent.tools) == 2
    assert any(isinstance(t, KnowledgeRetrieverTool) for t in agent.tools)
    assert any(isinstance(t, FallbackDirectTool) for t in agent.tools)
    assert "knowledge_retriever_tool" in agent.SYSTEM_PROMPT
    assert "fallback_direct_tool" in agent.SYSTEM_PROMPT
    assert "insufficient" in agent.SYSTEM_PROMPT


def test_callback_handler_attachment():
    received_traces = []

    class DummyCallbackHandler:
        def on_agent_finish(self, trace_tree):
            received_traces.append(trace_tree)

    handler = DummyCallbackHandler()
    agent = TraceLoopAgent(callback_handler=handler)
    res = agent.run("Hello there!")

    assert len(received_traces) == 1
    assert received_traces[0]["id"] == res.trace_id
    assert received_traces[0]["output"]["answer"] == res.answer
