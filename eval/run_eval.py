import json
import time
from pathlib import Path
from agent.graph import research_graph
from ingestion.unified_context import build_unified_context
from storage.db import init_db, save_research_record


SAMPLE_QUERIES = [
    "What are the latest advancements in quantum computing in 2024?",
    "Explain the impact of large language models on software development.",
    "What are the key differences between RAG and fine-tuning for LLMs?",
    "Summarize recent breakthroughs in CRISPR gene editing.",
    "What is the current state of fusion energy research?",
    "How does the transformer architecture work in modern AI models?",
    "What are the environmental impacts of Bitcoin mining?",
    "Explain the key principles behind reinforcement learning from human feedback.",
    "What are the latest developments in autonomous vehicle technology?",
    "How is AI being used in drug discovery and pharmaceutical research?",
]


def run_single(query: str, verbose: bool = True) -> dict:
    """
    Runs a single query through the full research pipeline.
    Returns the result dict with timing info.
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print("="*60)

    context = build_unified_context(query=query)

    initial_state = {
        "query": query,
        "text_context": context["text_context"],
        "images": context["images"],
        "messages": [],
        "tool_outputs": [],
        "report": "",
        "scorecard": {},
        "trace": [],
    }

    start = time.time()
    result = research_graph.invoke(initial_state)
    latency = round(time.time() - start, 2)

    result["latency_seconds"] = latency

    if verbose:
        scorecard = result.get("scorecard", {})
        print(f"Latency     : {latency}s")
        print(f"Faithfulness: {scorecard.get('faithfulness', 'N/A')}")
        print(f"Relevance   : {scorecard.get('relevance', 'N/A')}")
        print(f"Completeness: {scorecard.get('completeness', 'N/A')}")
        print(f"Coherence   : {scorecard.get('coherence', 'N/A')}")
        print(f"Overall     : {scorecard.get('overall', 'N/A')}")

    return result


def run_batch(queries: list[str] = None, output_path: str = "eval/results.json") -> dict:
    """
    Runs all queries and saves results to a JSON file.
    Returns aggregated metrics.
    """
    init_db()
    queries = queries or SAMPLE_QUERIES
    results = []
    failed = []

    print(f"\nRunning batch eval on {len(queries)} queries...\n")

    for i, query in enumerate(queries, start=1):
        print(f"[{i}/{len(queries)}] Running: {query[:60]}...")
        try:
            result = run_single(query, verbose=True)
            results.append({
                "query": query,
                "scorecard": result.get("scorecard", {}),
                "latency_seconds": result.get("latency_seconds", 0),
                "report_length": len(result.get("report", "")),
            })
        except Exception as e:
            print(f"  ERROR: {e}")
            failed.append({"query": query, "error": str(e)})

    # Aggregate metrics
    metrics = _aggregate(results)
    metrics["failed_count"] = len(failed)
    metrics["total_queries"] = len(queries)
    metrics["successful_queries"] = len(results)

    # Save to file
    output = {
        "summary": metrics,
        "results": results,
        "failed": failed,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print("BATCH EVAL SUMMARY")
    print("="*60)
    print(f"Total queries     : {metrics['total_queries']}")
    print(f"Successful        : {metrics['successful_queries']}")
    print(f"Failed            : {metrics['failed_count']}")
    print(f"Avg Faithfulness  : {metrics['avg_faithfulness']}")
    print(f"Avg Relevance     : {metrics['avg_relevance']}")
    print(f"Avg Completeness  : {metrics['avg_completeness']}")
    print(f"Avg Coherence     : {metrics['avg_coherence']}")
    print(f"Avg Overall       : {metrics['avg_overall']}")
    print(f"Avg Latency       : {metrics['avg_latency_seconds']}s")
    print(f"\nResults saved to  : {output_path}")

    return metrics


def _aggregate(results: list[dict]) -> dict:
    if not results:
        return {}

    dims = ["faithfulness", "relevance", "completeness", "coherence", "overall"]
    aggregated = {}

    for dim in dims:
        scores = [r["scorecard"].get(dim, 0) for r in results if r.get("scorecard")]
        aggregated[f"avg_{dim}"] = round(sum(scores) / len(scores), 2) if scores else 0

    latencies = [r["latency_seconds"] for r in results if r.get("latency_seconds")]
    aggregated["avg_latency_seconds"] = round(sum(latencies) / len(latencies), 2) if latencies else 0

    return aggregated


if __name__ == "__main__":
    run_batch()