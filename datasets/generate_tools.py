import json
from pathlib import Path
from typing import Dict, List


DATASET_PATH = Path(__file__).resolve().parent / "tools_1000.json"
TOTAL_TOOLS = 1000

CATEGORY_CONFIGS = {
    "weather": {
        "core": {
            "name": "weather_api",
            "description": "Get weather information for a city including temperature, forecast, and conditions.",
            "arguments": ["city"],
            "examples": [
                "weather in Tokyo",
                "temperature in London",
                "forecast for New York",
                "rain forecast for Seattle",
            ],
        },
        "names": ["forecast", "temperature", "climate", "rain", "storm"],
        "nouns": ["city", "region", "airport", "destination", "coast"],
        "verbs": ["Get", "Fetch", "Look up", "Retrieve", "Check"],
    },
    "finance": {
        "prefix": "finance",
        "core": {
            "name": "finance_api",
            "description": "Retrieve stock, currency, and market information for companies and tickers.",
            "arguments": ["symbol"],
            "examples": [
                "NVDA stock price",
                "latest finance update for Nvidia",
                "USD to EUR rate",
            ],
        },
        "names": ["market", "stocks", "fx", "equity", "pricing"],
        "nouns": ["ticker", "company", "portfolio", "currency", "market"],
        "verbs": ["Get", "Fetch", "Look up", "Track", "Retrieve"],
    },
    "news": {
        "prefix": "headline",
        "core_tools": [
            {
                "name": "news_api",
                "description": "Get breaking news and recent headlines for a topic, company, or person.",
                "arguments": ["topic"],
                "examples": [
                    "latest Nvidia news",
                    "breaking news about Apple",
                    "recent AI headlines",
                ],
            },
            {
                "name": "news_search",
                "description": "Find news articles and source reports for a topic or company.",
                "arguments": ["topic"],
                "examples": [
                    "find news articles about Nvidia",
                    "search news for chip industry updates",
                    "news search for Tesla coverage",
                ],
            },
            {
                "name": "news_summary",
                "description": "Summarize news articles and produce a short briefing for a topic.",
                "arguments": ["topic"],
                "dependencies": ["news_search"],
                "examples": [
                    "summarize recent Nvidia articles",
                    "brief me on the latest AI headlines",
                    "news summary for Microsoft",
                ],
            },
        ],
        "names": ["headline", "press", "bulletin", "coverage", "media"],
        "nouns": ["company", "topic", "industry", "person", "story"],
        "verbs": ["Get", "Fetch", "Search", "Track", "Monitor"],
    },
    "maps": {
        "prefix": "places",
        "core": {
            "name": "maps_places_api",
            "description": "Find nearby places, restaurants, and local businesses using location queries.",
            "arguments": ["query", "location"],
            "examples": [
                "find restaurants near me",
                "coffee shops in Soho",
                "hotels near Times Square",
            ],
        },
        "names": ["places", "restaurants", "route", "geocode", "nearby"],
        "nouns": ["location", "city", "route", "restaurant", "business"],
        "verbs": ["Find", "Search", "Locate", "Discover", "Look up"],
    },
    "math": {
        "prefix": "math",
        "core": {
            "name": "math_solver",
            "description": "Solve calculations, algebra, square roots, and arithmetic expressions.",
            "arguments": ["expression"],
            "examples": [
                "calculate square root",
                "solve 2x + 5 = 11",
                "evaluate 144 / 12",
            ],
        },
        "names": ["solver", "algebra", "calculus", "equation", "statistics"],
        "nouns": ["expression", "equation", "formula", "matrix", "number"],
        "verbs": ["Solve", "Compute", "Calculate", "Evaluate", "Derive"],
    },
    "code": {
        "prefix": "code",
        "core": {
            "name": "code_search",
            "description": "Search source code, APIs, and programming examples across repositories.",
            "arguments": ["query"],
            "examples": [
                "find Python AST parser example",
                "search for FastAPI middleware code",
                "look up FAISS integration",
            ],
        },
        "names": ["search", "snippet", "repo", "api", "debug"],
        "nouns": ["repository", "codebase", "function", "module", "symbol"],
        "verbs": ["Search", "Inspect", "Find", "Scan", "Locate"],
    },
    "translation": {
        "prefix": "lingua",
        "core": {
            "name": "translation_service",
            "description": "Translate text between languages for chat, documents, and phrases.",
            "arguments": ["text", "target_language"],
            "examples": [
                "translate hello to Spanish",
                "French translation for this email",
                "convert sentence to Japanese",
            ],
        },
        "names": ["localize", "multilingual", "language", "phrase", "lingual"],
        "nouns": ["text", "document", "message", "sentence", "phrase"],
        "verbs": ["Translate", "Convert", "Rewrite", "Localize", "Render"],
    },
    "search": {
        "prefix": "web",
        "core": {
            "name": "web_search",
            "description": "Search the public web for articles, documentation, reviews, and general information.",
            "arguments": ["query"],
            "examples": [
                "search the web for Nvidia latest news",
                "best restaurants near me",
                "find information about vector search",
            ],
        },
        "names": ["web", "internet", "discovery", "lookup", "query"],
        "nouns": ["query", "topic", "website", "article", "result"],
        "verbs": ["Search", "Find", "Discover", "Look up", "Browse"],
    },
    "social": {
        "prefix": "social",
        "core": {
            "name": "social_search",
            "description": "Search social posts, discussions, and trending conversations for a topic.",
            "arguments": ["topic"],
            "examples": [
                "social posts about ToolScout",
                "Reddit discussion on Nvidia",
                "trending posts about AI chips",
            ],
        },
        "names": ["trend", "post", "mention", "community", "engagement"],
        "nouns": ["brand", "topic", "hashtag", "campaign", "community"],
        "verbs": ["Track", "Search", "Monitor", "Analyze", "Discover"],
    },
    "calendar": {
        "prefix": "calendar",
        "core": {
            "name": "calendar_lookup",
            "description": "Look up meetings, events, and schedules for a date or time window.",
            "arguments": ["date"],
            "examples": [
                "what meetings do I have tomorrow",
                "calendar for Friday",
                "show events next week",
            ],
        },
        "names": ["schedule", "meeting", "event", "agenda", "planner"],
        "nouns": ["date", "day", "week", "meeting", "schedule"],
        "verbs": ["Check", "Look up", "Find", "Review", "List"],
    },
}

QUERY_SPECS = [
    {
        "query": "latest Nvidia news",
        "expected_tools": ["news_api", "web_search", "finance_api"],
    },
    {
        "query": "weather in Tokyo",
        "expected_tools": ["weather_api"],
    },
    {
        "query": "calculate square root",
        "expected_tools": ["math_solver"],
    },
    {
        "query": "translate hello to Spanish",
        "expected_tools": ["translation_service"],
    },
    {
        "query": "find restaurants near me",
        "expected_tools": ["maps_places_api", "web_search"],
    },
    {
        "query": "find Python AST parser example",
        "expected_tools": ["code_search", "web_search"],
    },
    {
        "query": "show posts about ToolScout on social media",
        "expected_tools": ["social_search", "web_search"],
    },
    {
        "query": "what meetings do I have tomorrow",
        "expected_tools": ["calendar_lookup"],
    },
    {
        "query": "USD to EUR exchange rate",
        "expected_tools": ["finance_api", "web_search"],
    },
    {
        "query": "rain forecast for Seattle",
        "expected_tools": ["weather_api"],
    },
]


def build_core_tools() -> List[Dict[str, object]]:
    tools: List[Dict[str, object]] = []
    for category in CATEGORY_CONFIGS:
        config = CATEGORY_CONFIGS[category]
        records = config.get("core_tools")
        if records is None:
            records = [config["core"]]

        for record in records:
            core = dict(record)
            core["category"] = category
            tools.append(core)
    return tools


def build_variant_tool(
    category: str, category_index: int, variant_index: int
) -> Dict[str, object]:
    config = CATEGORY_CONFIGS[category]
    name_token = config["names"][variant_index % len(config["names"])]
    noun = config["nouns"][variant_index % len(config["nouns"])]
    verb = config["verbs"][variant_index % len(config["verbs"])]
    ordinal = variant_index + 1

    if category == "weather":
        arguments = ["city"]
        description = (
            "{0} {1} and live {2} conditions for a {3}.".format(
                verb, name_token, category, noun
            )
        )
        examples = [
            "{0} in city {1}".format(name_token, ordinal),
            "weather for metro zone {0}".format(ordinal),
        ]
    elif category == "finance":
        arguments = ["symbol"]
        description = (
            "{0} {1} data, price movement, and {2} analytics for a {3}.".format(
                verb, name_token, category, noun
            )
        )
        examples = [
            "{0} update for symbol {1}".format(name_token, ordinal),
            "market data for company {0}".format(ordinal),
        ]
    elif category == "news":
        arguments = ["topic"]
        description = (
            "{0} {1} digests and recent {2} coverage for a {3}.".format(
                verb, name_token, category, noun
            )
        )
        examples = [
            "{0} on company sector {1}".format(name_token, ordinal),
            "press coverage on topic cluster {0}".format(ordinal),
        ]
    elif category == "maps":
        arguments = ["query", "location"]
        description = (
            "{0} {1} results and {2} recommendations for a {3}.".format(
                verb, name_token, category, noun
            )
        )
        examples = [
            "restaurants near district {0}".format(ordinal),
            "{0} around block {1}".format(name_token, ordinal),
        ]
    elif category == "math":
        arguments = ["expression"]
        description = (
            "{0} {1} problems and advanced {2} operations on a {3}.".format(
                verb, name_token, category, noun
            )
        )
        examples = [
            "solve expression {0} * 7".format(ordinal),
            "compute square root of {0}".format(ordinal * ordinal),
        ]
    elif category == "code":
        arguments = ["query"]
        description = (
            "{0} {1} references and {2} examples inside a {3}.".format(
                verb, name_token, category, noun
            )
        )
        examples = [
            "find {0} helper in repo {1}".format(name_token, ordinal),
            "search codebase for module {0}".format(ordinal),
        ]
    elif category == "translation":
        arguments = ["text", "target_language"]
        description = (
            "{0} {1} text and {2} phrases for a {3}.".format(
                verb, name_token, category, noun
            )
        )
        examples = [
            "localize sentence {0} into French".format(ordinal),
            "render message {0} in German".format(ordinal),
        ]
    elif category == "search":
        arguments = ["query"]
        description = (
            "{0} {1} results across the web and broad {2} sources for a {3}.".format(
                verb, name_token, category, noun
            )
        )
        examples = [
            "search web for topic {0}".format(ordinal),
            "find article about trend {0}".format(ordinal),
        ]
    elif category == "social":
        arguments = ["topic"]
        description = (
            "{0} {1} posts, mentions, and {2} signals for a {3}.".format(
                verb, name_token, category, noun
            )
        )
        examples = [
            "social chatter on brand {0}".format(ordinal),
            "posts about hashtag {0}".format(ordinal),
        ]
    else:
        arguments = ["date"]
        description = (
            "{0} {1} information and {2} planning details for a {3}.".format(
                verb, name_token, category, noun
            )
        )
        examples = [
            "calendar for day {0}".format(ordinal),
            "meetings in week {0}".format(ordinal),
        ]

    prefix = config.get("prefix", category)

    return {
        "name": "{0}_{1}_{2:03d}".format(prefix, name_token, category_index * 100 + ordinal),
        "description": description,
        "arguments": arguments,
        "category": category,
        "examples": examples,
        "dependencies": [],
    }


def build_tools() -> List[Dict[str, object]]:
    tools = build_core_tools()
    remaining = TOTAL_TOOLS - len(tools)
    category_count = len(CATEGORY_CONFIGS)
    variants_per_category = remaining // category_count
    remainder = remaining % category_count

    for category_index, category in enumerate(CATEGORY_CONFIGS):
        extra_variant = 1 if category_index < remainder else 0
        for variant_index in range(variants_per_category + extra_variant):
            tools.append(build_variant_tool(category, category_index, variant_index))

    if len(tools) != TOTAL_TOOLS:
        raise ValueError("Expected {0} tools, generated {1}.".format(TOTAL_TOOLS, len(tools)))

    return tools


def build_dataset() -> Dict[str, object]:
    tools = build_tools()
    return {
        "metadata": {
            "generator": "datasets/generate_tools.py",
            "tool_count": len(tools),
            "categories": list(CATEGORY_CONFIGS.keys()),
        },
        "tools": tools,
        "queries": QUERY_SPECS,
    }


def main() -> None:
    dataset = build_dataset()
    DATASET_PATH.write_text(json.dumps(dataset, indent=2), encoding="utf-8")
    print("Wrote {0} tools to {1}".format(len(dataset["tools"]), DATASET_PATH))


if __name__ == "__main__":
    main()
