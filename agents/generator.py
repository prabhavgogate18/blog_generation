from langchain_core.prompts import ChatPromptTemplate

from config.settings import get_llm
from utils.prompt_loader import load_prompt
from graph.state import BlogState

generator_llm = get_llm(temperature=0.7)


def generator_node(state: BlogState) -> BlogState:
    topic = state["topic"]
    tone = state.get("tone", "")
    word_count = state.get("word_count", 800)
    constraints = state.get("constraints", "")
    research_notes = state.get("research_notes", "")

    iteration = state.get("iteration", 1)
    critic_feedback = state.get("critic_feedback", "")
    previous_draft = state.get("draft", "")

    # 🧠 list of past mistakes / feedback across iterations
    mistake_memory = state.get("mistake_memory", [])
    # To avoid super-long prompts, just use the last few
    recent_mistakes = mistake_memory[-3:] if mistake_memory else []
    mistake_memory_text = "\n\n".join(recent_mistakes) if recent_mistakes else "None yet."

    system_prompt = load_prompt("generator_system")

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "human",
                "You are writing or revising a blog.\n\n"
                "Topic: {topic}\n"
                "Tone: {tone}\n"
                "Target word count: {word_count}\n"
                "Additional constraints: {constraints}\n"
                "Iteration: {iteration}\n\n"
                "Research notes (source of truth):\n{research_notes}\n\n"
                "GLOBAL MISTAKE MEMORY (mistakes you must NOT repeat):\n"
                "{mistake_memory_text}\n\n"
                "{revision_instructions}"
            ),
        ]
    )

    if iteration <= 1 or not previous_draft or not critic_feedback:
        revision_instructions = (
            "This is the FIRST iteration. Write a complete, polished blog from scratch "
            "using the research notes. Do not mention that this is a draft or an iteration."
        )
    else:
        revision_instructions = (
            "This is a LATER iteration.\n\n"
            "You are given the previous draft and critic feedback.\n"
            "Your job is to REVISE and IMPROVE the blog, not to ignore it.\n\n"
            "Previous draft:\n"
            "---------------------\n"
            f"{previous_draft}\n"
            "---------------------\n\n"
            "Critic feedback from last iteration:\n"
            "---------------------\n"
            f"{critic_feedback}\n"
            "---------------------\n\n"
            "Using BOTH the research notes and the GLOBAL MISTAKE MEMORY above, "
            "rewrite or refine the blog so that you DO NOT repeat any prior mistakes. "
            "Preserve strengths, fix weaknesses, and keep the requested tone and lengths."
        )

    messages = prompt.format_messages(
        topic=topic,
        tone=tone,
        word_count=word_count,
        constraints=constraints,
        iteration=iteration,
        research_notes=research_notes,
        mistake_memory_text=mistake_memory_text,
        revision_instructions=revision_instructions,
    )

    response = generator_llm.invoke(messages)
    state["draft"] = response.content
    return state
