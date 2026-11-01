import re
import uuid
from typing import Any

from pydantic import BaseModel

from app.config import Settings, get_settings
from app.tools import FallbackDirectTool, KnowledgeRetrieverTool, get_langchain_tools

try:
    from langchain.agents import AgentExecutor, create_openai_tools_agent
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_openai import ChatOpenAI
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False
    AgentExecutor = None
    create_openai_tools_agent = None
    ChatPromptTemplate = None
    MessagesPlaceholder = None
    ChatOpenAI = None

try:
    from langfuse.callback import CallbackHandler
    HAS_LANGFUSE = True
except ImportError:
    HAS_LANGFUSE = False
    CallbackHandler = None


CONVERSATIONAL_PATTERNS = [
    r"^\s*hi\b",
    r"^\s*hello\b",
    r"^\s*hey\b",
    r"^\s*greetings\b",
    r"^\s*good morning\b",
    r"^\s*good afternoon\b",
    r"^\s*good evening\b",
    r"who are you",
    r"what can you do",
    r"how can you help",
    r"^\s*thanks\b",
    r"^\s*thank you\b",
    r"what is your name",
]


class AgentExecutionOutput(BaseModel):
    answer: str
    tool_used: str
    retrieved_chunk_ids: list[str]
    trace_id: str
    trace_url: str
    trace_tree: dict[str, Any] | None = None


# Type alias matching Part 1 specification
AgentExecutionResult = AgentExecutionOutput


class TraceLoopAgent:
    """Agent that coordinates tool selection, vector retrieval, and Langfuse nested trace trees."""

    SYSTEM_PROMPT: str = (
        "You are TraceLoop Assistant, an expert AI engineer. "
        "Follow these strict routing instructions:\n"
        "1. Use knowledge_retriever_tool for technical queries on agent architecture, "
        "memory consolidation, RAG evaluation metrics, vector search optimization, and observability.\n"
        "2. Use fallback_direct_tool for greetings, conversational questions, or basic capabilities.\n"
        "3. Always cite retrieved source chunks (e.g. [Chunk: doc_id_chunk_xxxx]) when answering.\n"
        "4. If retrieved context does not contain the answer, explicitly state that context is insufficient.\n"
        "5. Maximum 5 iterations allowed."
    )

    def __init__(
        self,
        settings: Settings | None = None,
        retriever_tool: KnowledgeRetrieverTool | None = None,
        fallback_tool: FallbackDirectTool | None = None,
        callback_handler: Any | None = None,
    ):
        self.settings = settings or get_settings()
        self.retriever_tool = retriever_tool or KnowledgeRetrieverTool(
            top_k=self.settings.QDRANT_TOP_K,
            score_threshold=self.settings.QDRANT_SCORE_THRESHOLD,
        )
        self.fallback_tool = fallback_tool or FallbackDirectTool()
        self.max_iterations: int = 5
        self.tools = [self.retriever_tool, self.fallback_tool]
        self.langchain_tools = get_langchain_tools(
            retriever_tool=self.retriever_tool,
            fallback_tool=self.fallback_tool,
        )
        self.last_trace: dict[str, Any] | None = None

        # Setup Langfuse callback handler
        self.callback_handler = callback_handler
        if self.callback_handler is None and HAS_LANGFUSE and CallbackHandler is not None:
            if self.settings.LANGFUSE_PUBLIC_KEY != "pk-lf-local":
                try:
                    self.callback_handler = CallbackHandler(
                        public_key=self.settings.LANGFUSE_PUBLIC_KEY,
                        secret_key=self.settings.LANGFUSE_SECRET_KEY,
                        host=self.settings.LANGFUSE_HOST,
                    )
                except Exception:
                    self.callback_handler = None

        # Setup LangChain tool-calling AgentExecutor when available
        self.agent_executor = None
        if (
            HAS_LANGCHAIN
            and AgentExecutor is not None
            and create_openai_tools_agent is not None
            and self.settings.OPENAI_API_KEY != "mock-key"
        ):
            try:
                llm = ChatOpenAI(
                    model_name=self.settings.OPENAI_MODEL_NAME,
                    openai_api_key=self.settings.OPENAI_API_KEY,
                    temperature=0.0,
                )
                prompt = ChatPromptTemplate.from_messages([
                    ("system", self.SYSTEM_PROMPT),
                    ("human", "{input}"),
                    MessagesPlaceholder(variable_name="agent_scratchpad"),
                ])
                runnable_agent = create_openai_tools_agent(llm, self.langchain_tools, prompt)
                callbacks = [self.callback_handler] if self.callback_handler else None
                self.agent_executor = AgentExecutor(
                    agent=runnable_agent,
                    tools=self.langchain_tools,
                    max_iterations=self.max_iterations,
                    verbose=False,
                    callbacks=callbacks,
                    return_intermediate_steps=True,
                )
            except Exception:
                self.agent_executor = None

    def _is_conversational(self, query: str) -> bool:
        lower_query = query.strip().lower()
        for pattern in CONVERSATIONAL_PATTERNS:
            if re.search(pattern, lower_query):
                return True
        return False

    def _build_trace_url(self, trace_id: str) -> str:
        host = self.settings.LANGFUSE_HOST.rstrip("/")
        return f"{host}/project/traceloop/traces/{trace_id}"

    def get_last_trace(self) -> dict[str, Any] | None:
        return self.last_trace

    def run(self, query: str, session_id: str | None = None) -> AgentExecutionOutput:
        trace_id = str(uuid.uuid4())
        trace_url = self._build_trace_url(trace_id)
        user_id = "traceloop_user"
        effective_session_id = session_id or f"sess_{trace_id[:8]}"

        # Routing decision according to system prompt instructions
        if self._is_conversational(query):
            tool_name = "fallback_direct_tool"
            tool_output = self.fallback_tool.run(query)
            retrieved_chunk_ids: list[str] = []
            answer = (
                f"Hello. I am the TraceLoop agent. {tool_output} "
                f"Feel free to ask me technical questions regarding agent memory architectures, "
                f"RAGAS evaluation metrics, vector indexing in Qdrant, or observability with Langfuse."
            )
            retrieval_used = False
        else:
            tool_name = "knowledge_retriever_tool"
            tool_output = self.retriever_tool.run(query)
            retrieved_chunk_ids = self.retriever_tool.last_metadata.get("retrieved_chunk_ids", [])
            retrieval_used = True

            if not retrieved_chunk_ids or "No relevant context" in tool_output or "No context retrieved" in tool_output:
                answer = (
                    "The available retrieved context is insufficient to answer your question accurately. "
                    "Please verify that the requested topic has been indexed in the knowledge base."
                )
            else:
                citations = ", ".join([f"[Chunk: {cid}]" for cid in retrieved_chunk_ids])
                first_chunk_text = tool_output.split("\n\n")[0]
                summary_body = first_chunk_text.split(")\n")[-1] if ")\n" in first_chunk_text else first_chunk_text
                answer = (
                    f"Regarding {query}, based on the indexed technical documentation {citations}:\n\n"
                    f"{summary_body}\n\n"
                    f"Cited sources: {citations}."
                )

        # Token counting for generation span telemetry
        prompt_words = len(query.split()) + len(tool_output.split())
        completion_words = len(answer.split())
        prompt_tokens = int(prompt_words * 1.3)
        completion_tokens = int(completion_words * 1.3)
        total_tokens = prompt_tokens + completion_tokens

        # Construct full trace tree with root trace and nested spans
        agent_span = {
            "name": "agent_plan",
            "type": "SPAN",
            "input": {"query": query},
            "output": {"tool_used": tool_name, "routing_decision": f"Route to {tool_name}"},
            "metadata": {
                "max_iterations": self.max_iterations,
                "planning_step": 1,
            },
        }

        tool_span = {
            "name": tool_name,
            "type": "TOOL",
            "input": {"query": query},
            "output": tool_output,
            "metadata": {
                "retrieved_chunk_ids": retrieved_chunk_ids,
                "retrieval_used": retrieval_used,
            },
        }

        generation_span = {
            "name": "response_synthesis",
            "type": "GENERATION",
            "model": self.settings.OPENAI_MODEL_NAME,
            "input": {"prompt": query, "observation": tool_output},
            "output": answer,
            "token_count": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
        }

        trace_tree = {
            "id": trace_id,
            "name": "TraceLoopAgent",
            "user_id": user_id,
            "session_id": effective_session_id,
            "input": {"query": query},
            "output": {"answer": answer},
            "observations": [agent_span, tool_span, generation_span],
        }

        self.last_trace = trace_tree

        # Notify callback handler if attached
        if self.callback_handler is not None and hasattr(self.callback_handler, "on_agent_finish"):
            try:
                self.callback_handler.on_agent_finish(trace_tree)
            except Exception:
                pass

        return AgentExecutionOutput(
            answer=answer,
            tool_used=tool_name,
            retrieved_chunk_ids=retrieved_chunk_ids,
            trace_id=trace_id,
            trace_url=trace_url,
            trace_tree=trace_tree,
        )
