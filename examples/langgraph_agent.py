import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, TypedDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from toolscout import (
    SkillRegistry,
    SkillRetriever,
    ToolExecutor,
    ToolGraph,
    ToolRegistry,
    ToolRetriever,
)


def news_search(topic: str) -> Dict[str, object]:
    return {
        "topic": topic,
        "articles": [
            "{0} announces a new AI platform".format(topic),
            "Analysts publish a fresh outlook on {0}".format(topic),
        ],
        "source": "demo-news-index",
    }


def news_summary(topic: str) -> Dict[str, str]:
    return {
        "topic": topic,
        "summary": "{0} is receiving strong news coverage with a focus on AI and growth.".format(
            topic
        ),
        "source": "demo-summarizer",
    }


def web_search(query: str) -> Dict[str, object]:
    return {
        "query": query,
        "results": [
            "Public web result for {0}".format(query),
            "Background article on {0}".format(query),
        ],
        "source": "demo-web-search",
    }


def finance_api(symbol: str) -> Dict[str, object]:
    return {
        "symbol": symbol.upper(),
        "price": 132.45,
        "currency": "USD",
        "source": "demo-market-feed",
    }


def build_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register_tool(
        name="news_search",
        description="Find news articles and source reports for a topic or company.",
        args=["topic"],
        handler=news_search,
        category="news",
        examples=["latest Nvidia news", "find AI chip coverage"],
    )
    registry.register_tool(
        name="news_summary",
        description="Summarize news articles and produce a short briefing for a topic.",
        args=["topic"],
        handler=news_summary,
        category="news",
        dependencies=["news_search"],
        examples=["summarize recent Nvidia articles", "brief me on AI headlines"],
    )
    registry.register_tool(
        name="web_search",
        description="Search the public web for articles, documentation, and general information.",
        args=["query"],
        handler=web_search,
        category="search",
        examples=["search the web for Nvidia latest news"],
    )
    registry.register_tool(
        name="finance_api",
        description="Retrieve stock and market information for a company ticker.",
        args=["symbol"],
        handler=finance_api,
        category="finance",
        examples=["NVDA stock price"],
    )
    return registry


def build_skill_registry() -> SkillRegistry:
    skills = SkillRegistry()
    skills.register_skill(
        name="news_research",
        description="Find and summarize news articles for a topic or company.",
        tools=["news_search", "news_summary"],
        examples=["latest Nvidia news", "summarize AI chip headlines"],
    )
    skills.register_skill(
        name="market_intelligence",
        description="Research company news, market context, and web coverage.",
        tools=["finance_api", "web_search", "news_search"],
        examples=["Nvidia market update", "latest company outlook"],
    )
    return skills


class AgentState(TypedDict, total=False):
    query: str
    routed_skill: str
    retrieved_tools: List[str]
    chosen_tool: str
    tool_arguments: Dict[str, str]
    tool_result: Dict[str, object]
    runtime: str


def derive_topic(query: str) -> str:
    lowered = query.lower()
    for marker in ("about ", "on ", "for "):
        if marker in lowered:
            start = lowered.index(marker) + len(marker)
            return query[start:].strip(" ?.")
    return "Nvidia" if "nvidia" in lowered else query.strip(" ?.")


def retrieve_node(state: AgentState, context: Dict[str, object]) -> AgentState:
    routed = context["skill_retriever"].route(
        state["query"],
        tool_registry=context["tool_registry"],
        tool_retriever=context["tool_retriever"],
        tool_graph=context["tool_graph"],
        top_k_tools=4,
        graph_aware=True,
    )
    return {
        **state,
        "routed_skill": routed.skill.name,
        "retrieved_tools": [hit.tool.name for hit in routed.tools],
    }


def plan_node(state: AgentState, context: Dict[str, object]) -> AgentState:
    query = state["query"].lower()
    chosen_tool = "news_search"
    arguments: Dict[str, str]

    if "summarize" in query or "summary" in query or "brief" in query:
        chosen_tool = "news_summary"
        arguments = {"topic": derive_topic(state["query"])}
    elif "stock" in query or "price" in query or "market" in query:
        chosen_tool = "finance_api"
        arguments = {"symbol": "NVDA"}
    elif "web" in query:
        chosen_tool = "web_search"
        arguments = {"query": state["query"]}
    else:
        arguments = {"topic": derive_topic(state["query"])}

    if chosen_tool not in state["retrieved_tools"]:
        chosen_tool = state["retrieved_tools"][0]
        if chosen_tool == "web_search":
            arguments = {"query": state["query"]}
        elif chosen_tool == "finance_api":
            arguments = {"symbol": "NVDA"}
        else:
            arguments = {"topic": derive_topic(state["query"])}

    return {
        **state,
        "chosen_tool": chosen_tool,
        "tool_arguments": arguments,
    }


def execute_node(state: AgentState, context: Dict[str, object]) -> AgentState:
    result = context["executor"].execute(
        state["chosen_tool"], state["tool_arguments"]
    )
    return {
        **state,
        "tool_result": result,
    }


def run_sequential(state: AgentState, context: Dict[str, object]) -> AgentState:
    state = retrieve_node(state, context)
    state = plan_node(state, context)
    state = execute_node(state, context)
    state["runtime"] = "sequential fallback"
    return state


def run_with_langgraph(state: AgentState, context: Dict[str, object]) -> AgentState:
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError:
        return run_sequential(state, context)

    graph = StateGraph(AgentState)
    graph.add_node("retrieve", lambda current: retrieve_node(current, context))
    graph.add_node("plan", lambda current: plan_node(current, context))
    graph.add_node("execute", lambda current: execute_node(current, context))
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "plan")
    graph.add_edge("plan", "execute")
    graph.add_edge("execute", END)
    app = graph.compile()
    final_state = app.invoke(state)
    final_state["runtime"] = "langgraph"
    return final_state


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Demonstrate ToolScout inside a LangGraph-style agent node."
    )
    parser.add_argument(
        "--query",
        default="summarize recent Nvidia articles",
        help="User query for the agent.",
    )
    args = parser.parse_args()

    tool_registry = build_tool_registry()
    skill_registry = build_skill_registry()
    tool_retriever = ToolRetriever(registry=tool_registry)
    tool_retriever.fit()
    skill_retriever = SkillRetriever(registry=skill_registry)
    skill_retriever.fit()
    context = {
        "tool_registry": tool_registry,
        "tool_retriever": tool_retriever,
        "skill_retriever": skill_retriever,
        "tool_graph": ToolGraph.from_registry(tool_registry),
        "executor": ToolExecutor(tool_registry),
    }

    state: AgentState = {"query": args.query}
    final_state = run_with_langgraph(state, context)
    print(json.dumps(final_state, indent=2))


if __name__ == "__main__":
    main()
