from langgraph.graph import StateGraph, END

from .state import BlogState
from agents.researcher import researcher_node
from agents.generator import generator_node
from agents.critic import critic_node
from agents.orchestrator import orchestrator_node


def orchestrator_router(state: BlogState) -> str:
    """Route from orchestrator based on iteration and route decision."""
    route = state.get("route", "continue")
    
    if route == "done":
        return "done"
    
    # Only do research on the first iteration
    iteration = state.get("iteration", 0)
    if iteration <= 1:
        return "researcher"
    else:
        # Skip research, go directly to generator for refinement
        return "generator"


def build_blog_graph():
    graph = StateGraph(BlogState)

    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("generator", generator_node)
    graph.add_node("critic", critic_node)

    graph.set_entry_point("orchestrator")

    graph.add_conditional_edges(
        "orchestrator",
        orchestrator_router,
        {
            "researcher": "researcher",
            "generator": "generator",
            "done": END,
        },
    )

    graph.add_edge("researcher", "generator")
    graph.add_edge("generator", "critic")
    graph.add_edge("critic", "orchestrator")

    return graph.compile()