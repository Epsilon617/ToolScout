import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from toolscout import ToolExecutor, ToolRegistry, ToolRetriever


def weather_api(city: str) -> Dict[str, str]:
    return {
        "city": city,
        "forecast": "18C, cloudy, light wind",
        "source": "demo-weather-service",
    }


def news_search(topic: str) -> Dict[str, object]:
    return {
        "topic": topic,
        "headlines": [
            "{0} announces a new product update".format(topic),
            "Analysts discuss the latest outlook for {0}".format(topic),
        ],
        "source": "demo-news-service",
    }


def stock_quote(ticker: str) -> Dict[str, object]:
    return {
        "ticker": ticker.upper(),
        "price": 132.45,
        "currency": "USD",
        "source": "demo-market-feed",
    }


def calendar_lookup(date: str) -> Dict[str, object]:
    return {
        "date": date,
        "events": [
            "09:00 Product review",
            "14:30 Team sync",
        ],
        "source": "demo-calendar",
    }


def build_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register_tool(
        name="weather_api",
        description="Get weather for a city",
        args=["city"],
        handler=weather_api,
        tags=["weather", "forecast"],
        examples=["weather in London", "forecast for Tokyo tomorrow"],
    )
    registry.register_tool(
        name="news_search",
        description="Find the latest news about a topic, company, or person",
        args=["topic"],
        handler=news_search,
        tags=["news", "search"],
        examples=["latest news about Nvidia", "recent coverage on EV batteries"],
    )
    registry.register_tool(
        name="stock_quote",
        description="Get the latest stock price for a company ticker",
        args=["ticker"],
        handler=stock_quote,
        tags=["finance", "stocks"],
        examples=["NVDA stock price", "quote for MSFT"],
    )
    registry.register_tool(
        name="calendar_lookup",
        description="Look up calendar events for a date",
        args=["date"],
        handler=calendar_lookup,
        tags=["calendar", "schedule"],
        examples=["meetings on Friday", "agenda for 2026-03-15"],
    )
    return registry


def extract_after_marker(query: str, markers: List[str]) -> str:
    lowered = query.lower()
    for marker in markers:
        if marker in lowered:
            start = lowered.index(marker) + len(marker)
            return query[start:].strip(" ?.")
    return query.strip(" ?.")


def guess_ticker(query: str) -> str:
    uppercase_tokens = re.findall(r"\b[A-Z]{2,5}\b", query)
    if uppercase_tokens:
        return uppercase_tokens[0]

    companies = {
        "nvidia": "NVDA",
        "microsoft": "MSFT",
        "apple": "AAPL",
        "tesla": "TSLA",
    }
    lowered = query.lower()
    for company, ticker in companies.items():
        if company in lowered:
            return ticker
    return "NVDA"


def mock_llm_plan(query: str, candidate_names: List[str]) -> Dict[str, object]:
    lowered = query.lower()

    if "weather" in lowered and "weather_api" in candidate_names:
        city = extract_after_marker(query, [" in ", " for "]) or "London"
        return {"tool_name": "weather_api", "arguments": {"city": city}}

    if ("stock" in lowered or "price" in lowered) and "stock_quote" in candidate_names:
        return {
            "tool_name": "stock_quote",
            "arguments": {"ticker": guess_ticker(query)},
        }

    if "calendar" in lowered or "meeting" in lowered or "schedule" in lowered:
        if "calendar_lookup" in candidate_names:
            date = extract_after_marker(query, [" on ", " for "]) or "today"
            return {"tool_name": "calendar_lookup", "arguments": {"date": date}}

    if "news_search" in candidate_names:
        topic = extract_after_marker(query, [" about ", " on "]) or query
        return {"tool_name": "news_search", "arguments": {"topic": topic}}

    best = candidate_names[0]
    fallback_argument = "topic"
    if best == "weather_api":
        fallback_argument = "city"
    elif best == "stock_quote":
        fallback_argument = "ticker"
    elif best == "calendar_lookup":
        fallback_argument = "date"

    return {
        "tool_name": best,
        "arguments": {fallback_argument: query},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="ToolScout simple agent demo.")
    parser.add_argument(
        "--query",
        default="latest news about Nvidia",
        help="User query to run through the agent pipeline.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of retrieved tools to show the planner.",
    )
    args = parser.parse_args()

    registry = build_registry()
    retriever = ToolRetriever(registry=registry)
    retriever.fit()

    results = retriever.search(args.query, top_k=args.top_k)
    candidate_names = [result.tool.name for result in results]
    plan = mock_llm_plan(args.query, candidate_names)

    executor = ToolExecutor(registry)
    execution_result = executor.execute(
        plan["tool_name"], plan["arguments"]
    )

    payload = {
        "query": args.query,
        "retrieval_backend": retriever.backend_summary,
        "retrieved_tools": [
            {
                "rank": result.rank,
                "tool": result.tool.name,
                "score": round(result.score, 4),
            }
            for result in results
        ],
        "planner_output": plan,
        "tool_result": execution_result,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

