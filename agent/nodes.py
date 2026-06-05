from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from agent.tools import get_all_tools
from agent.judge import run_judge
from storage.db import save_research_record
from typing import TypedDict, Annotated
import json
import operator


# --- Agent State ---

class AgentState(TypedDict):
    query: str
    text_context: str
    images: list
    messages: Annotated[list, operator.add]
    tool_outputs: list
    report: str
    scorecard: dict
    trace: list


# --- Helpers ---

def _tc_to_dict(tc) -> dict:
    if isinstance(tc, dict):
        return tc
    try:
        return tc.model_dump()
    except AttributeError:
        return {"name": str(tc), "args": {}, "id": ""}


def _get_llm(tools: list = None):
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    if tools:
        return llm.bind_tools(tools)
    return llm


# --- Node 1: Supervisor / Tool Router ---

def supervisor_node(state: AgentState, config: RunnableConfig) -> AgentState:
    tools = get_all_tools()
    llm_with_tools = _get_llm(tools=tools)

    system_prompt = """You are a research assistant with access to the following tools:
1. tavily_search_results_json — search the web for current information
2. python_repl — execute Python code for calculations or data processing
3. read_document — read and extract text from an uploaded PDF

Your job is to:
- Analyze the user's query and context
- Call the appropriate tools to gather information
- Call multiple tools if needed to fully answer the query
- Be thorough and systematic

Always cite which tool provided which information."""

    messages = [SystemMessage(content=system_prompt)]
    messages.append(HumanMessage(content=state["text_context"]))
    messages.extend(state["messages"])

    response = _get_llm(tools=tools).invoke(messages, config=config)

    trace_entry = {
        "node": "supervisor",
        "tool_calls": [_tc_to_dict(tc) for tc in response.tool_calls] if response.tool_calls else [],
    }

    return {
        "messages": [response],
        "trace": state.get("trace", []) + [trace_entry],
    }


# --- Node 2: Tool Executor ---

def tool_executor_node(state: AgentState, config: RunnableConfig) -> AgentState:
    tools = get_all_tools()
    tools_by_name = {tool.name: tool for tool in tools}

    last_message = state["messages"][-1]
    tool_outputs = state.get("tool_outputs", [])
    new_messages = []
    trace_entries = []

    for raw_tc in last_message.tool_calls:
        # Normalize to dict regardless of whether it's a dict or object
        tc = _tc_to_dict(raw_tc)
        tool_name = tc.get("name", "")
        tool_args = tc.get("args", {})
        tool_id = tc.get("id", "")

        tool = tools_by_name.get(tool_name)
        if not tool:
            result_str = f"Error: Tool '{tool_name}' not found."
        else:
            try:
                result = tool.invoke(tool_args)
                result_str = result if isinstance(result, str) else json.dumps(result, indent=2)
            except Exception as e:
                result_str = f"Error executing {tool_name}: {str(e)}"

        tool_outputs.append({
            "tool": tool_name,
            "args": tool_args,
            "output": result_str,
        })

        new_messages.append(
            ToolMessage(content=result_str, tool_call_id=tool_id)
        )

        trace_entries.append({
            "node": "tool_executor",
            "tool": tool_name,
            "args": tool_args,
            "output_preview": result_str[:300],
        })

    return {
        "messages": new_messages,
        "tool_outputs": tool_outputs,
        "trace": state.get("trace", []) + trace_entries,
    }


# --- Node 3: Synthesis ---

def synthesis_node(state: AgentState, config: RunnableConfig) -> AgentState:
    llm = _get_llm()

    tool_outputs_text = "\n\n".join([
        f"### Tool: {t['tool']}\n**Args:** {t['args']}\n**Output:**\n{t['output']}"
        for t in state.get("tool_outputs", [])
    ])

    system_prompt = """You are a research report writer. Given a query and the outputs
from various research tools, write a structured markdown report.

The report must follow this structure:
# Research Report: [Query Title]

## Summary
(2-3 sentence executive summary)

## Findings
(Detailed findings organized by topic, cite sources inline as [Source: tool_name])

## Analysis
(Your synthesis and interpretation of the findings)

## Conclusion
(Key takeaways)

## Sources
(List all sources used with tool name and query used)

Be factual, cite every claim to a tool output, and be thorough."""

    user_message = f"""Query: {state['query']}

Tool Outputs:
{tool_outputs_text}

Write the research report now."""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message),
    ], config=config)

    trace_entry = {
        "node": "synthesis",
        "report_length": len(response.content),
    }

    return {
        "report": response.content,
        "trace": state.get("trace", []) + [trace_entry],
    }


# --- Node 4: LLM-as-Judge ---

def judge_node(state: AgentState, config: RunnableConfig) -> AgentState:
    scorecard = run_judge(
        query=state["query"],
        report=state["report"],
        tool_outputs=state.get("tool_outputs", []),
    )

    trace_entry = {
        "node": "judge",
        "scorecard": scorecard,
    }

    return {
        "scorecard": scorecard,
        "trace": state.get("trace", []) + [trace_entry],
    }


# --- Node 5: Storage ---

def storage_node(state: AgentState, config: RunnableConfig) -> AgentState:
    save_research_record(
        query=state["query"],
        report=state["report"],
        scorecard=state["scorecard"],
    )

    trace_entry = {"node": "storage", "status": "saved"}

    return {
        "trace": state.get("trace", []) + [trace_entry],
    }


# --- Routing Logic ---

def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "synthesize"