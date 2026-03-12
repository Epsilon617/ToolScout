import argparse
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.eval_utils import (
    CROSS_LINGUAL_QUERIES,
    build_runtime,
    dataset_statistics,
    expand_methods,
    format_dataset_statistics,
    load_json,
    method_label,
    search_with_method,
)


def extract_json_object(text: str) -> Dict[str, object]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("Judge output did not include a JSON object.")
    return json.loads(match.group(0))


def build_judge_prompt(query: str, tools: List[object]) -> str:
    tool_lines = []
    for rank, hit in enumerate(tools, start=1):
        tool_lines.append("{0}. {1}".format(rank, hit.tool.to_prompt_text()))
    schema = '{"score": 1, "adequacy": true, "reason": "brief explanation"}'
    return "\n".join(
        [
            "You are evaluating retrieved tools for an agent query.",
            "Score how adequate the retrieved tools are for answering the query.",
            "Return JSON only using this schema: {0}".format(schema),
            "Use score 1-10 where 10 means the best tool is present and the shortlist is highly usable.",
            "",
            "User query: {0}".format(query),
            "",
            "Retrieved tools:",
            "\n\n".join(tool_lines),
        ]
    )


def mock_judge(
    query: str,
    hits: List[object],
    correct_tool: Optional[str] = None,
) -> Dict[str, object]:
    names = [hit.tool.name for hit in hits]
    if not hits:
        return {"score": 1, "adequacy": False, "reason": "No tools were retrieved."}

    if correct_tool and correct_tool in names:
        rank = names.index(correct_tool) + 1
        score_map = {1: 10, 2: 8, 3: 7, 4: 6, 5: 5}
        score = score_map.get(rank, 4)
        return {
            "score": score,
            "adequacy": True,
            "reason": "The correct tool is present at rank {0}.".format(rank),
        }

    if correct_tool:
        return {
            "score": 2,
            "adequacy": False,
            "reason": "The correct tool is missing from the shortlist.",
        }

    return {
        "score": 4,
        "adequacy": True,
        "reason": "Mock judge used lexical shortlist quality only.",
    }


def openai_judge(
    query: str,
    hits: List[object],
    model: str,
) -> Dict[str, object]:
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required for OpenAI judge mode.")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("openai is not installed. Run `pip install -e .` first.") from exc

    client = OpenAI()
    response = client.responses.create(
        model=model,
        input=build_judge_prompt(query, hits),
    )
    output_text = getattr(response, "output_text", "").strip()
    if not output_text:
        raise ValueError("The judge model returned an empty response.")
    payload = extract_json_object(output_text)
    payload["score"] = int(payload["score"])
    payload["adequacy"] = bool(payload["adequacy"])
    payload["reason"] = str(payload["reason"])
    return payload


def evaluate_judge(
    tool_dataset: Dict[str, object],
    query_dataset: List[Dict[str, object]],
    top_k: int = 5,
    method: str = "toolscout",
    judge_mode: str = "mock",
    model: str = "gpt-4.1-mini",
) -> Dict[str, object]:
    runtime = build_runtime(
        tool_dataset,
        feedback_queries=query_dataset,
    )
    try:
        judgments = []
        latencies_ms: List[float] = []
        for record in query_dataset:
            start = time.perf_counter()
            hits = search_with_method(
                record["query"],
                runtime=runtime,
                top_k=top_k,
                method=method,
            )
            latencies_ms.append((time.perf_counter() - start) * 1000.0)

            if judge_mode == "openai":
                judgment = openai_judge(record["query"], hits, model=model)
            else:
                judgment = mock_judge(
                    record["query"],
                    hits,
                    correct_tool=record.get("correct_tool"),
                )
            judgments.append(
                {
                    "query": record["query"],
                    "retrieved_tools": [hit.tool.name for hit in hits],
                    "judgment": judgment,
                }
            )

        scores = [item["judgment"]["score"] for item in judgments]
        adequacy_rate = sum(
            1 for item in judgments if item["judgment"]["adequacy"]
        ) / len(judgments)
        return {
            "average_adequacy_score": statistics.mean(scores),
            "adequacy_rate": adequacy_rate,
            "avg_latency_ms": statistics.mean(latencies_ms),
            "judgments": judgments,
        }
    finally:
        runtime.cleanup()


def render_results(results: List[Dict[str, object]]) -> str:
    lines = [
        "| Method | Avg Adequacy Score | Adequacy Rate | Avg Latency |",
        "| ------ | -----------------: | ------------: | ----------: |",
    ]
    for row in results:
        lines.append(
            "| {method} | {average_adequacy_score:.2f}/10 | {adequacy_rate:.1%} | {avg_latency_ms:.3f} ms |".format(
                **row
            )
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate ToolScout retrieval using an LLM-as-a-judge workflow."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "tools_1000.json",
        help="Path to the tool dataset.",
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "tool_queries.json",
        help="Path to the query dataset.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of tools shown to the judge.",
    )
    parser.add_argument(
        "--method",
        choices=["random", "semantic", "semantic_rerank", "toolscout", "baseline", "all"],
        default="all",
        help="Method to evaluate.",
    )
    parser.add_argument(
        "--judge-mode",
        choices=["mock", "openai"],
        default="mock",
        help="Judge backend. Use mock mode for offline testing.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4.1-mini",
        help="OpenAI model name for judge mode.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-query judgments.",
    )
    args = parser.parse_args()

    tool_dataset = load_json(args.dataset)
    query_dataset = load_json(args.queries)
    stats = dataset_statistics(
        tool_dataset=tool_dataset,
        query_records=query_dataset,
        extra_query_records=CROSS_LINGUAL_QUERIES,
    )

    results = []
    per_method_judgments = []
    for method in expand_methods(args.method):
        metrics = evaluate_judge(
            tool_dataset=tool_dataset,
            query_dataset=query_dataset,
            top_k=args.top_k,
            method=method,
            judge_mode=args.judge_mode,
            model=args.model,
        )
        results.append(
            {
                "method": method_label(method),
                "average_adequacy_score": metrics["average_adequacy_score"],
                "adequacy_rate": metrics["adequacy_rate"],
                "avg_latency_ms": metrics["avg_latency_ms"],
            }
        )
        per_method_judgments.append((method_label(method), metrics["judgments"]))

    print("ToolScout Judge Evaluation")
    print("judge mode: {0}".format(args.judge_mode))
    print(format_dataset_statistics(stats))
    print("")
    print(render_results(results))

    if args.verbose:
        print("")
        for method_name, judgments in per_method_judgments:
            print("Method: {0}".format(method_name))
            for item in judgments:
                print("Query: {0}".format(item["query"]))
                print("Tools: {0}".format(", ".join(item["retrieved_tools"])))
                print("Judgment: {0}".format(json.dumps(item["judgment"], ensure_ascii=False)))
                print("")


if __name__ == "__main__":
    main()
