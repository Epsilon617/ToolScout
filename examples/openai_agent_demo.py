import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from toolscout import ToolExecutor, ToolRegistry, ToolRetriever


def news_search(topic: str) -> Dict[str, object]:
    return {
        "topic": topic,
        "headlines": [
            "{0} launches a new AI platform".format(topic),
            "{0} remains a major focus for analysts".format(topic),
        ],
        "source": "demo-news-service",
    }


def weather_api(city: str) -> Dict[str, str]:
    return {
        "city": city,
        "forecast": "17C and partly sunny",
        "source": "demo-weather-service",
    }


def stock_quote(ticker: str) -> Dict[str, object]:
    return {
        "ticker": ticker.upper(),
        "price": 132.45,
        "currency": "USD",
        "source": "demo-market-feed",
    }


def build_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register_tool(
        name="news_search",
        description="Find the latest news about a topic, company, or person",
        args=["topic"],
        handler=news_search,
        tags=["news", "search"],
    )
    registry.register_tool(
        name="weather_api",
        description="Get weather for a city",
        args=["city"],
        handler=weather_api,
        tags=["weather", "forecast"],
    )
    registry.register_tool(
        name="stock_quote",
        description="Get the latest stock price for a company ticker",
        args=["ticker"],
        handler=stock_quote,
        tags=["finance", "stocks"],
    )
    return registry


def build_prompt(query: str, tools: List[object]) -> str:
    tool_blocks = "\n\n".join(tool.tool.to_prompt_text() for tool in tools)
    schema = '{"tool_name": "tool_name_here", "arguments": {"arg_name": "value"}}'
    return "\n".join(
        [
            "You are selecting a single tool for the user.",
            "Only choose from the candidate tools below.",
            "Return JSON only and do not wrap it in markdown.",
            "Use this schema exactly: {0}".format(schema),
            "",
            "Candidate tools:",
            tool_blocks,
            "",
            "User query: {0}".format(query),
        ]
    )


def extract_json_object(text: str) -> Dict[str, object]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("Model output did not include a JSON object.")
    return json.loads(match.group(0))


def main() -> None:
    parser = argparse.ArgumentParser(description="ToolScout OpenAI agent demo.")
    parser.add_argument(
        "--query",
        default="latest news about Nvidia",
        help="User query to run through the retrieved-tool prompt.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4.1-mini",
        help="OpenAI model name.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of candidate tools to send to the model.",
    )
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required for this demo.")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit(
            "openai is not installed. Run `pip install -e .` first."
        ) from exc

    registry = build_registry()
    retriever = ToolRetriever(registry=registry)
    retriever.fit()
    hits = retriever.search(args.query, top_k=args.top_k)

    client = OpenAI()
    response = client.responses.create(
        model=args.model,
        input=build_prompt(args.query, hits),
    )
    output_text = getattr(response, "output_text", "").strip()
    if not output_text:
        raise SystemExit("The model returned an empty response.")

    plan = extract_json_object(output_text)
    executor = ToolExecutor(registry)
    result = executor.execute(plan["tool_name"], plan["arguments"])

    payload = {
        "query": args.query,
        "retrieved_tools": [
            {
                "rank": hit.rank,
                "tool": hit.tool.name,
                "score": round(hit.score, 4),
            }
            for hit in hits
        ],
        "model_output": plan,
        "tool_result": result,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

