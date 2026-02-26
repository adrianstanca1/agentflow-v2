"""
AgentFlow v2 — Universal Agent
Supports local Ollama models + cloud providers with automatic fallback.
Every agent can run 100% locally with no API keys required.
"""
import asyncio
import json
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import structlog
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_community.llms.ollama import Ollama
from langchain_community.chat_models import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field

from ..core.config import settings
from ..core.ollama_manager import ollama_manager

logger = structlog.get_logger()


# ── Event types ───────────────────────────────────────────────────────────────
class AgentEventType:
    STARTED = "started"
    THOUGHT = "thought"
    PLANNING = "planning"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    OUTPUT = "output"
    STREAM_TOKEN = "stream_token"
    ERROR = "error"
    COMPLETE = "complete"
    MEMORY_SAVE = "memory_save"
    HANDOFF = "handoff"


class AgentEvent(BaseModel):
    event_type: str
    content: Any
    agent_id: str
    agent_name: str
    session_id: str
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True


# ── LLM Factory ───────────────────────────────────────────────────────────────
def build_llm(
    model: str,
    temperature: float = 0.7,
    streaming: bool = True,
    tools: Optional[List] = None,
    prefer_local: bool = False,
) -> Any:
    """
    Intelligent LLM routing:
      1. Ollama local/remote  → ChatOpenAI with Ollama base_url
      2. Anthropic (claude-)  → ChatAnthropic (native SDK, no LiteLLM)
      3. OpenAI-compat clouds → ChatOpenAI with provider's base_url + key
      4. Fallback             → LiteLLM gateway (100+ providers)
    """
    from ..core.cloud_manager import cloud_manager

    is_ollama = prefer_local or model.startswith("ollama/") or ":" in model

    # ── Route 1: Ollama ───────────────────────────────────────────────────────
    if is_ollama:
        model_name = model.replace("ollama/", "")
        best_host = ollama_manager._get_best_host() or settings.ollama_url
        llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            streaming=streaming,
            base_url=f"{best_host}/v1",
            api_key="ollama",
        )

    # ── Route 2: Anthropic (native) ───────────────────────────────────────────
    elif model.startswith("claude-") and settings.anthropic_api_key:
        try:
            from langchain_anthropic import ChatAnthropic
            llm = ChatAnthropic(
                model=model,
                temperature=temperature,
                streaming=streaming,
                anthropic_api_key=settings.anthropic_api_key,
            )
        except ImportError:
            # Fallback to LiteLLM gateway
            llm = ChatOpenAI(
                model=f"anthropic/{model}",
                temperature=temperature,
                streaming=streaming,
                base_url=f"{settings.litellm_url}/v1",
                api_key=settings.litellm_api_key,
            )

    # ── Route 3: Direct cloud provider (OpenAI-compatible) ────────────────────
    elif cloud_manager.is_cloud_model(model):
        kwargs = cloud_manager.get_direct_client_kwargs(model)
        routing = kwargs.get("__provider")

        if routing == "litellm" or not kwargs:
            # Gemini / unknown → route through LiteLLM with prefixed model ID
            litellm_model = cloud_manager.get_litellm_model_id(model)
            llm = ChatOpenAI(
                model=litellm_model,
                temperature=temperature,
                streaming=streaming,
                base_url=f"{settings.litellm_url}/v1",
                api_key=settings.litellm_api_key,
            )
        elif routing == "anthropic":
            try:
                from langchain_anthropic import ChatAnthropic
                llm = ChatAnthropic(
                    model=model,
                    temperature=temperature,
                    streaming=streaming,
                    anthropic_api_key=kwargs.get("__key"),
                )
            except ImportError:
                llm = ChatOpenAI(
                    model=model,
                    base_url=f"{settings.litellm_url}/v1",
                    api_key=settings.litellm_api_key,
                )
        else:
            # Groq / Together / Mistral / Fireworks / OpenRouter
            llm = ChatOpenAI(
                model=model,
                temperature=temperature,
                streaming=streaming,
                base_url=kwargs["base_url"],
                api_key=kwargs["api_key"],
            )

    # ── Route 4: Fallback — LiteLLM gateway ──────────────────────────────────
    else:
        llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            streaming=streaming,
            base_url=f"{settings.litellm_url}/v1",
            api_key=settings.litellm_api_key,
        )

    if tools:
        return llm.bind_tools(tools)
    return llm


# ── Base Universal Agent ───────────────────────────────────────────────────────
class UniversalAgent(ABC):
    """
    Universal agent that works with any model — local Ollama or cloud.
    Inherits LangGraph orchestration + streaming + memory.
    """

    def __init__(
        self,
        agent_id: str,
        name: str,
        description: str,
        agent_type: str = "custom",
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_iterations: int = 15,
        memory_enabled: bool = True,
        tools: Optional[List] = None,
        system_prompt: Optional[str] = None,
        prefer_local: bool = False,
        stream_tokens: bool = True,
    ):
        self.agent_id = agent_id
        self.name = name
        self.description = description
        self.agent_type = agent_type
        self.temperature = temperature
        self.max_iterations = max_iterations
        self.memory_enabled = memory_enabled
        self.tools = tools or []
        self.prefer_local = prefer_local
        self.stream_tokens = stream_tokens

        # Resolve model — prefer local if no cloud keys
        self.model = model or self._resolve_default_model()
        self.system_prompt = system_prompt or self.default_system_prompt()

        # Build LLM
        self.llm = build_llm(
            model=self.model,
            temperature=temperature,
            streaming=True,
            prefer_local=prefer_local,
        )
        self.llm_with_tools = build_llm(
            model=self.model,
            temperature=temperature,
            streaming=True,
            tools=self.tools if self.tools else None,
            prefer_local=prefer_local,
        )

        # Checkpointing
        self.checkpointer = MemorySaver()
        self.graph = self._build_graph()

        logger.info(
            "Agent initialized",
            id=agent_id,
            name=name,
            model=self.model,
            prefer_local=prefer_local,
        )

    def _resolve_default_model(self) -> str:
        """Resolve best available model."""
        # Check cloud keys
        has_anthropic = bool(settings.anthropic_api_key)
        has_openai = bool(settings.openai_api_key)

        if self.prefer_local:
            return settings.default_model_local

        # Check if Ollama has models
        best_host = ollama_manager._get_best_host()
        if best_host and not (has_anthropic or has_openai):
            return settings.default_model_local

        if has_anthropic:
            return settings.default_model_cloud
        if has_openai:
            return "gpt-4o"
        return settings.default_model_local

    def switch_model(self, model: str, prefer_local: bool = False):
        """Hot-swap the model at runtime."""
        self.model = model
        self.prefer_local = prefer_local
        self.llm = build_llm(model, self.temperature, prefer_local=prefer_local)
        self.llm_with_tools = build_llm(
            model, self.temperature, tools=self.tools if self.tools else None,
            prefer_local=prefer_local
        )
        self.graph = self._build_graph()
        logger.info("Model switched", agent=self.name, model=model)

    @abstractmethod
    def default_system_prompt(self) -> str: ...

    @abstractmethod
    def _build_graph(self): ...

    # ── Core execution ────────────────────────────────────────────────────────
    async def run(
        self,
        task: str,
        session_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute agent and return full result."""
        session_id = session_id or str(uuid.uuid4())
        output = ""
        events = []
        error = None

        async for event in self.stream(task, session_id=session_id, context=context):
            events.append(event.model_dump())
            if event.event_type == AgentEventType.COMPLETE:
                output = event.content
            elif event.event_type == AgentEventType.OUTPUT:
                output = event.content
            elif event.event_type == AgentEventType.ERROR:
                error = event.content

        return {
            "session_id": session_id,
            "output": output,
            "error": error,
            "events": events,
            "status": "failed" if error else "completed",
            "agent": self.name,
            "model": self.model,
        }

    async def stream(
        self,
        task: str,
        session_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> AsyncGenerator[AgentEvent, None]:
        """Stream agent execution as typed events."""
        session_id = session_id or str(uuid.uuid4())

        def emit(event_type: str, content: Any, **kwargs) -> AgentEvent:
            return AgentEvent(
                event_type=event_type,
                content=content,
                agent_id=self.agent_id,
                agent_name=self.name,
                session_id=session_id,
                metadata=kwargs,
            )

        yield emit(AgentEventType.STARTED, f"Starting {self.name}...")

        initial_state = self._make_initial_state(task, session_id, context)
        config = {"configurable": {"thread_id": session_id}}

        try:
            async for chunk in self.graph.astream(
                initial_state,
                config=config,
                stream_mode="updates",
            ):
                for node_name, node_output in chunk.items():
                    if not isinstance(node_output, dict):
                        continue

                    # Emit thoughts
                    for thought in node_output.get("thoughts", []):
                        yield emit(AgentEventType.THOUGHT, thought, node=node_name)

                    # Emit messages
                    for msg in node_output.get("messages", []):
                        if isinstance(msg, AIMessage):
                            if msg.content and isinstance(msg.content, str):
                                yield emit(AgentEventType.OUTPUT, msg.content, node=node_name)
                            if hasattr(msg, "tool_calls") and msg.tool_calls:
                                for tc in msg.tool_calls:
                                    yield emit(
                                        AgentEventType.TOOL_CALL,
                                        {"name": tc.get("name"), "args": tc.get("args", {})},
                                    )
                        elif isinstance(msg, ToolMessage):
                            yield emit(
                                AgentEventType.TOOL_RESULT,
                                {"tool": msg.name, "result": str(msg.content)[:2000]},
                            )

                    # Emit planning
                    if "plan" in node_output and node_output["plan"]:
                        yield emit(AgentEventType.PLANNING, node_output["plan"], node=node_name)

                    # Final output
                    if node_output.get("status") == "completed":
                        output = node_output.get("task_output", "")
                        if output:
                            yield emit(AgentEventType.COMPLETE, output)
                            return

        except Exception as e:
            logger.error("Agent stream error", agent=self.name, error=str(e))
            yield emit(AgentEventType.ERROR, str(e))
            return

        yield emit(AgentEventType.COMPLETE, "Task completed.")

    def _make_initial_state(
        self, task: str, session_id: str, context: Optional[Dict]
    ) -> Dict[str, Any]:
        """Build initial LangGraph state."""
        return {
            "messages": [{"role": "user", "content": task}],
            "task_input": task,
            "task_output": "",
            "thoughts": [],
            "iteration": 0,
            "max_iterations": self.max_iterations,
            "status": "running",
            "error": None,
            "session_id": session_id,
            "metadata": context or {},
            "plan": None,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.agent_id,
            "name": self.name,
            "description": self.description,
            "type": self.agent_type,
            "model": self.model,
            "prefer_local": self.prefer_local,
            "tools": [t.name if hasattr(t, "name") else str(t) for t in self.tools],
            "max_iterations": self.max_iterations,
            "memory_enabled": self.memory_enabled,
            "temperature": self.temperature,
        }
