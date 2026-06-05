from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
import json
import re


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


def run_judge(query: str, report: str, tool_outputs: list) -> dict:
    """
    Runs the LLM-as-judge evaluation.
    Returns a scorecard dict.
    """
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    
    tool_summary = "\n\n".join([
        f"Tool: {t['tool']}\nOutput (truncated): {t['output'][:500]}"
        for t in tool_outputs
    ])

    user_message = f"""Query: {query}

--- Tool Outputs Used ---
{tool_summary}

--- Generated Report ---
{report}

Now evaluate the report and return the JSON scorecard."""

    response = llm.invoke([
        SystemMessage(content=JUDGE_SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ])

    return _parse_scorecard(response.content)


def _parse_scorecard(raw: str) -> dict:
    """
    Safely parses the JSON scorecard from the judge response.
    Falls back to a default scorecard on failure.
    """
    try:
        # Strip markdown code fences if present
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        scorecard = json.loads(clean)

        # Recompute overall as a safeguard
        dims = ["faithfulness", "relevance", "completeness", "coherence"]
        scores = [scorecard.get(d, 0) for d in dims]
        scorecard["overall"] = round(sum(scores) / len(scores), 2)

        return scorecard

    except (json.JSONDecodeError, KeyError):
        return {
            "faithfulness": 0,
            "relevance": 0,
            "completeness": 0,
            "coherence": 0,
            "overall": 0,
            "justifications": {
                "faithfulness": "Parse error",
                "relevance": "Parse error",
                "completeness": "Parse error",
                "coherence": "Parse error",
            },
            "error": f"Failed to parse judge response: {raw[:200]}",
        }