# agents/planner.py
from typing import List
from langchain_core.prompts import ChatPromptTemplate

from config.settings import get_llm
from utils.prompt_loader import load_prompt
from graph.state import BlogState

planner_llm = get_llm(temperature=0.2)


def planner_node(state: BlogState) -> BlogState:
    """
    Produce a prioritized list of web-search queries (strings) that the Researcher
    will run with Tavily. Output: state['search_queries'] = List[str]
    """
    topic = state["topic"]
    tone = state.get("tone", "")
    constraints = state.get("constraints", "")
    word_count = state.get("word_count", 800)

    system_prompt = load_prompt("planner_system")

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "human",
                "You are a planning agent. Given a blog Topic: {topic}, Tone: {tone}, "
                "Constraints: {constraints}, and Target word count: {word_count}, "
                "produce a prioritized list of 6-10 concise web search queries that will "
                "yield the best material for writing this blog. Return only a JSON array "
                "of query strings, ordered by priority.",
            ),
        ]
    )

    messages = prompt.format_messages(
        topic=topic,
        tone=tone,
        constraints=constraints,
        word_count=word_count,
    )

    # Ask the LLM for a JSON array of queries (simple, deterministic-ish)
    response = planner_llm.invoke(messages)
    text = response.content.strip()

    # Simple robust parsing: try to extract JSON array; fallback to newline-split
    import json
    search_queries: List[str] = []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            search_queries = [str(q).strip() for q in parsed if str(q).strip()]
    except Exception:
        # fallback: split by lines and clean bullets
        lines = [l.strip(" -•\t") for l in text.splitlines() if l.strip()]
        search_queries = [l for l in lines if len(l) > 5]

    # Limit to reasonable number (e.g. top 8)
    state["search_queries"] = search_queries[:8]
    return state
