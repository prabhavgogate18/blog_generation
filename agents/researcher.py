from typing import Dict, Any
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate

from config.settings import get_llm
from utils.prompt_loader import load_prompt
from graph.state import BlogState


# Tavily tool (web search)
tavily_search = TavilySearchResults(
    max_results=5,
    include_answer=True,
)


researcher_llm = get_llm(temperature=0.2)


def researcher_node(state: BlogState) -> BlogState:
    topic = state["topic"]
    constraints = state.get("constraints", "")
    tone = state.get("tone", "")
    word_count = state.get("word_count", 800)

    query = f"Research for in-depth blog on: {topic}. Tone: {tone}. " \
            f"Constraints: {constraints}. Word count target: {word_count}."

    search_results = tavily_search.invoke({"query": query})

    system_prompt = load_prompt("researcher_system")

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "human",
                "Topic: {topic}\nTone: {tone}\nWord count target: {word_count}\n"
                "Additional constraints: {constraints}\n\n"
                "Raw web search results:\n{search_results}",
            ),
        ]
    )

    messages = prompt.format_messages(
        topic=topic,
        tone=tone,
        word_count=word_count,
        constraints=constraints,
        search_results=search_results,
    )

    response = researcher_llm.invoke(messages)

    state["research_notes"] = response.content
    return state
