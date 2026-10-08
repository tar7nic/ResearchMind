import os
import re
import json

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage


JUDGE_SYSTEM_PROMPT = """You are an expert evaluator of research reports.

Given a research query, the tool outputs used to generate a report, and the final report itself,
score the report on the following dimensions from 0 to 10:

1. Faithfulness — Are all claims grounded in the retrieved tool outputs?
2. Relevance — Does the report directly address the original query?
3. Completeness — Are all key aspects of the query covered?
4. Coherence — Is the report logically structured and readable?

Return ONLY a valid JSON object with this exact structure, no preamble, no markdown:
{
  "faithfulness": <0-10>,
  "relevance": <0-10>,
  "completeness": <0-10>,
  "coherence": <0-10>,
  "overall": <average of the four>,
  "justifications": {
    "faithfulness": "<one-line justification>",
    "relevance": "<one-line justification>",
    "completeness": "<one-line justification>",
    "coherence": "<one-line justification>"
  }
}"""


DIMENSIONS = ["faithfulness", "relevance", "completeness", "coherence"]


def run_judge(query: str, report: str, tool_outputs: list) -> dict:
    """
    Runs the LLM-as-judge evaluation.
    Uses a different model from the generator (JUDGE_MODEL) to avoid self-scoring bias.
    Returns a scorecard dict.
    """
    llm = ChatGroq(
    model=os.getenv("JUDGE_MODEL", "openai/gpt-oss-20b"),
    temperature=0,
    max_tokens=4096,
    reasoning_effort="low",
    timeout=60,
    max_retries=2,
    )

    tool_summary = "\n\n".join([
        f"Tool: {t['tool']}\nOutput (truncated): {t['output'][:2000]}"
        for t in tool_outputs
    ])

    user_message = f"""Query: {query}

--- Tool Outputs Used ---
{tool_summary}

--- Generated Report ---
{report}

Now evaluate the report and return the JSON scorecard."""

    try:
        response = llm.invoke([
            SystemMessage(content=JUDGE_SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ])
    except Exception as e:
        return _error_scorecard(f"Judge call failed: {e}")

    return _parse_scorecard(response.content)


def _error_scorecard(message: str) -> dict:
    return {
        "faithfulness": 0,
        "relevance": 0,
        "completeness": 0,
        "coherence": 0,
        "overall": 0,
        "justifications": {d: "Parse error" for d in DIMENSIONS},
        "error": message[:300],
    }


def _parse_scorecard(raw: str) -> dict:
    """
    Safely parses the JSON scorecard from the judge response.
    Tolerates markdown fences and surrounding text.
    Falls back to a default scorecard on failure.
    """
    try:
        # Strip markdown code fences if present
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()

        # Grab the outermost JSON object even if the model added extra text
        match = re.search(r"\{.*\}", clean, re.DOTALL)
        scorecard = json.loads(match.group(0))

        # Coerce scores to floats, clamp to 0-10, and recompute overall as a safeguard
        scores = []
        for d in DIMENSIONS:
            value = max(0.0, min(10.0, float(scorecard.get(d, 0))))
            scorecard[d] = value
            scores.append(value)
        scorecard["overall"] = round(sum(scores) / len(scores), 2)

        if not isinstance(scorecard.get("justifications"), dict):
            scorecard["justifications"] = {d: "" for d in DIMENSIONS}

        return scorecard

    except (json.JSONDecodeError, KeyError, AttributeError, TypeError, ValueError):
        return _error_scorecard(f"Failed to parse judge response: {raw[:200]}")