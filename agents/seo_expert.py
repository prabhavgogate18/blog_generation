from langchain_core.prompts import ChatPromptTemplate

from config.settings import get_llm
from utils.prompt_loader import load_prompt
from graph.state import BlogState


seo_llm = get_llm(temperature=0.4)


def seo_expert_node(state: BlogState) -> BlogState:
    topic = state["topic"]
    tone = state.get("tone", "")
    constraints = state.get("constraints", "")
    draft = state.get("draft", "")

    system_prompt = load_prompt("seo_system")

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "human",
                "Here is the current blog draft. Improve it for SEO.\n\n"
                "Topic: {topic}\nTone: {tone}\nConstraints: {constraints}\n\n"
                "Draft:\n{draft}",
            ),
        ]
    )

    messages = prompt.format_messages(
        topic=topic,
        tone=tone,
        constraints=constraints,
        draft=draft,
    )

    response = seo_llm.invoke(messages)

    # Response should contain improved draft + SEO metadata.
    state["seo_suggestions"] = response.content

    # Optionally: let SEO expert directly rewrite the draft.
    # For simplicity, assume the first part of the response is the improved draft.
    # You can change this logic based on how you design the prompt.
    state["draft"] = response.content
    return state
