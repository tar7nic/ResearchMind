from langgraph.graph import StateGraph, END
from agent.nodes import (
    AgentState,
    supervisor_node,
    tool_executor_node,
    synthesis_node,
    judge_node,
    storage_node,
    should_continue,
)


def build_graph() -> StateGraph:
    """
    Builds and compiles the LangGraph research agent graph.

    Flow:
        supervisor → (tool_calls?) → tool_executor → supervisor (loop)
                                   → (no tool_calls) → synthesis → judge → storage → END
    """
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("tools", tool_executor_node)
    graph.add_node("synthesis", synthesis_node)
    graph.add_node("judge", judge_node)
    graph.add_node("storage", storage_node)

    # Entry point
    graph.set_entry_point("supervisor")

    # Conditional routing after supervisor
    graph.add_conditional_edges(
        "supervisor",
        should_continue,
        {
            "tools": "tools",
            "synthesize": "synthesis",
        },
    )

    # After tool execution, loop back to supervisor
    graph.add_edge("tools", "supervisor")

    # Linear flow after synthesis
    graph.add_edge("synthesis", "judge")
    graph.add_edge("judge", "storage")
    graph.add_edge("storage", END)

    return graph.compile()


# Singleton compiled graph
research_graph = build_graph()