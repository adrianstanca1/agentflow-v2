"""
AgentFlow v2 — Orchestrator & Agent Registry
Master coordinator with dynamic routing, CrewAI teams, and agent registry.
"""
import asyncio
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional

import structlog
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from .base import UniversalAgent, AgentEvent, AgentEventType
from .specialists import CodingAgent, ResearchAgent, BusinessAgent
from ..core.config import settings

logger = structlog.get_logger()


# ─────────────────────────────────────────────────────────
# AGENT REGISTRY
# ─────────────────────────────────────────────────────────
class AgentRegistry:
    def __init__(self):
        self._agents: Dict[str, UniversalAgent] = {}

    def register(self, key: str, agent: UniversalAgent):
        self._agents[key] = agent
        logger.info("Registered agent", key=key, name=agent.name, model=agent.model)

    def get(self, key: str) -> Optional[UniversalAgent]:
        return self._agents.get(key)

    def list(self) -> List[Dict[str, Any]]:
        return [a.to_dict() for a in self._agents.values()]

    def keys(self) -> List[str]:
        return list(self._agents.keys())

    def create_dynamic(
        self,
        agent_id: str,
        name: str,
        description: str,
        model: str,
        system_prompt: str,
        prefer_local: bool = False,
        agent_type: str = "custom",
    ) -> "DynamicAgent":
        agent = DynamicAgent(
            agent_id=agent_id,
            name=name,
            description=description,
            model=model,
            system_prompt=system_prompt,
            prefer_local=prefer_local,
            agent_type=agent_type,
        )
        self.register(agent_id, agent)
        return agent


registry = AgentRegistry()


# ─────────────────────────────────────────────────────────
# DYNAMIC AGENT (no-code agent creation)
# ─────────────────────────────────────────────────────────
class DynamicAgentState(TypedDict):
    messages: List[Any]
    task_input: str
    task_output: str
    thoughts: List[str]
    iteration: int
    max_iterations: int
    status: str
    error: Optional[str]
    session_id: str
    metadata: Dict[str, Any]
    plan: Optional[str]


class DynamicAgent(UniversalAgent):
    """User-configurable agent with custom system prompt and model."""

    def __init__(self, agent_id: str, name: str, description: str,
                 model: str, system_prompt: str, prefer_local: bool = False,
                 agent_type: str = "custom"):
        self._custom_prompt = system_prompt
        super().__init__(
            agent_id=agent_id, name=name, description=description,
            agent_type=agent_type, model=model, prefer_local=prefer_local,
        )

    def default_system_prompt(self): return self._custom_prompt

    def _build_graph(self):
        llm = self.llm

        async def process(state: DynamicAgentState) -> DynamicAgentState:
            msgs = state["messages"]
            if not msgs or not isinstance(msgs[0], SystemMessage):
                msgs = [SystemMessage(content=self.system_prompt)] + list(msgs)
            resp = await llm.ainvoke(msgs)
            return {**state, "messages": msgs + [resp],
                    "task_output": resp.content if isinstance(resp.content, str) else "",
                    "iteration": state.get("iteration", 0) + 1, "status": "completed"}

        g = StateGraph(DynamicAgentState)
        g.add_node("process", process)
        g.add_edge(START, "process")
        g.add_edge("process", END)
        return g.compile(checkpointer=self.checkpointer)


# ─────────────────────────────────────────────────────────
# ORCHESTRATOR TOOLS
# ─────────────────────────────────────────────────────────
@tool
async def delegate(agent_key: str, task: str, context: str = "") -> str:
    """Delegate a subtask to a specialist agent.
    Args:
        agent_key: One of 'coding', 'research', 'business' or any registered agent
        task: Specific task for the agent
        context: Context from previous steps
    """
    agent = registry.get(agent_key)
    if not agent:
        return f"Agent '{agent_key}' not found. Available: {registry.keys()}"
    full_task = f"{context}\n\nTask: {task}" if context else task
    result = await agent.run(task=full_task, session_id=str(uuid.uuid4()))
    return result.get("output", "No output") or "Completed without output"


@tool
async def parallel_run(tasks: List[Dict[str, str]]) -> str:
    """Run multiple agent tasks in parallel.
    Args:
        tasks: List of {agent_key, task} dicts
    """
    async def run_one(t):
        agent = registry.get(t.get("agent_key", "research"))
        if not agent: return {"error": f"Agent {t.get('agent_key')} not found"}
        r = await agent.run(task=t.get("task", ""), session_id=str(uuid.uuid4()))
        return {"agent": t.get("agent_key"), "output": r.get("output", "")}

    results = await asyncio.gather(*[run_one(t) for t in tasks], return_exceptions=True)
    return "\n\n---\n\n".join(
        r.get("output", str(r)) if isinstance(r, dict) else str(r)
        for r in results
    )


ORCH_TOOLS = [delegate, parallel_run]

ORCH_SYSTEM = """You are the master orchestrator of a multi-agent AI platform.

Available specialists:
- 'coding': Software engineering — write, debug, test code
- 'research': Deep research — search web, academic papers, synthesize
- 'business': Business analysis — data, reports, process automation

Your job:
1. Analyze the task complexity
2. Break it into focused subtasks
3. Delegate each subtask to the right specialist
4. Run independent tasks in parallel when possible
5. Synthesize all results into a cohesive final output

For simple tasks, delegate directly. For complex tasks, use parallel_run.
Always produce a complete, high-quality final answer."""


class OrchestratorState(TypedDict):
    messages: List[Any]
    task_input: str
    task_output: str
    thoughts: List[str]
    iteration: int
    max_iterations: int
    status: str
    error: Optional[str]
    session_id: str
    metadata: Dict[str, Any]
    plan: Optional[str]
    subtask_results: List[str]


class OrchestratorAgent(UniversalAgent):
    def __init__(self, **kwargs):
        kwargs.setdefault("agent_id", "orchestrator")
        kwargs.setdefault("name", "Orchestrator")
        kwargs.setdefault("description", "Coordinates multiple specialist agents for complex tasks")
        kwargs.setdefault("agent_type", "orchestrator")
        kwargs.setdefault("max_iterations", 20)
        super().__init__(tools=ORCH_TOOLS, **kwargs)

    def default_system_prompt(self): return ORCH_SYSTEM

    def _build_graph(self):
        llm = self.llm_with_tools
        tool_node = ToolNode(ORCH_TOOLS)

        def route(state):
            last = (state["messages"] or [None])[-1]
            if state.get("iteration",0) >= state.get("max_iterations",20): return "end"
            if last and hasattr(last,"tool_calls") and last.tool_calls: return "tools"
            return "end"

        async def orchestrate(state: OrchestratorState):
            msgs = state["messages"]
            if not msgs or not isinstance(msgs[0], SystemMessage):
                msgs = [SystemMessage(content=self.system_prompt)] + list(msgs)
            resp = await llm.ainvoke(msgs)
            done = not(hasattr(resp,"tool_calls") and resp.tool_calls)
            return {**state,"messages":msgs+[resp],"iteration":state.get("iteration",0)+1,
                    "task_output":resp.content if isinstance(resp.content,str) else state.get("task_output",""),
                    "status":"completed" if done else "running"}

        async def tools(state): return {**state, **(await tool_node.ainvoke(state))}
        g = StateGraph(OrchestratorState)
        g.add_node("orchestrate", orchestrate); g.add_node("tools", tools)
        g.add_edge(START, "orchestrate")
        g.add_conditional_edges("orchestrate", route, {"tools":"tools","end":END})
        g.add_edge("tools", "orchestrate")
        return g.compile(checkpointer=self.checkpointer)

    async def run_crew(
        self, task: str, agents: List[str] = None, session_id: str = None
    ) -> AsyncGenerator[AgentEvent, None]:
        """CrewAI-style coordinated team execution."""
        session_id = session_id or str(uuid.uuid4())
        agents = agents or ["research", "coding"]

        def emit(t, c, **kw):
            return AgentEvent(event_type=t, content=c, agent_id=self.agent_id,
                              agent_name=self.name, session_id=session_id, metadata=kw)

        yield emit(AgentEventType.STARTED, f"Starting crew with {len(agents)} agents: {', '.join(agents)}")

        try:
            from crewai import Agent as CA, Task as CT, Crew, Process
            from langchain_openai import ChatOpenAI

            best_host = __import__('backend.core.ollama_manager', fromlist=['ollama_manager']).ollama_manager._get_best_host()
            llm_url = f"{best_host or settings.ollama_url}/v1" if self.prefer_local else f"{settings.litellm_url}/v1"
            llm_key = "ollama" if self.prefer_local else settings.litellm_api_key

            crew_llm = ChatOpenAI(model=self.model, base_url=llm_url, api_key=llm_key)
            crew_agents, crew_tasks = [], []

            for key in agents:
                ag = registry.get(key)
                if not ag: continue
                ca = CA(role=ag.name, goal=ag.description,
                        backstory=f"Expert {ag.agent_type} agent.", verbose=False, llm=crew_llm)
                ct = CT(description=f"Handle the {key} aspects of: {task}",
                        expected_output=f"Complete {key} output", agent=ca)
                crew_agents.append(ca); crew_tasks.append(ct)

            yield emit(AgentEventType.THOUGHT, f"Crew assembled: {[a.role for a in crew_agents]}")
            crew = Crew(agents=crew_agents, tasks=crew_tasks, process=Process.sequential, verbose=False)
            result = await asyncio.get_event_loop().run_in_executor(None, crew.kickoff)
            yield emit(AgentEventType.COMPLETE, str(result))

        except ImportError:
            yield emit(AgentEventType.THOUGHT, "CrewAI not available, using sequential orchestration")
            async for event in self.stream(task, session_id=session_id):
                yield event
        except Exception as e:
            yield emit(AgentEventType.ERROR, str(e))


# ─────────────────────────────────────────────────────────
# INITIALIZE ALL AGENTS
# ─────────────────────────────────────────────────────────
def initialize_agents(prefer_local: bool = False) -> AgentRegistry:
    """Boot all default agents. prefer_local=True forces Ollama."""
    common = {"prefer_local": prefer_local}

    registry.register("coding", CodingAgent(**common))
    registry.register("research", ResearchAgent(**common))
    registry.register("business", BusinessAgent(**common))
    registry.register("orchestrator", OrchestratorAgent(**common))

    logger.info("All agents initialized", agents=registry.keys(), prefer_local=prefer_local)
    return registry
