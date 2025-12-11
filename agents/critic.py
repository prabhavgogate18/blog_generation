from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from config.settings import get_critic_llm
from utils.prompt_loader import load_prompt
from graph.state import BlogState

critic_llm = get_critic_llm(temperature=0.0)


class CriticScore(BaseModel):
    """Structured evaluation of the blog for confidence scoring."""
    overall_score: float = Field(
        description="Overall confidence score between 0 and 1."
    )
    grammar_score: int = Field(
        description="Grammar/clarity score between 1 and 10."
    )
    depth_score: int = Field(
        description="Depth of topic coverage score between 1 and 10."
    )
    structure_score: int = Field(
        description="Blog structure and flow score between 1 and 10."
    )
    seo_alignment_score: int = Field(
        description="SEO alignment score between 1 and 10."
    )
    short_feedback: str = Field(
        description="Short constructive feedback on how to improve the blog."
    )


def critic_node(state: BlogState) -> BlogState:
    draft = state.get("draft", "")
    topic = state["topic"]
    tone = state.get("tone", "")
    constraints = state.get("constraints", "")
    iteration = state.get("iteration", 0)

    system_prompt = load_prompt("critic_system")

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "human",
                "Evaluate this blog draft.\n\n"
                "Topic: {topic}\nTone: {tone}\nConstraints: {constraints}\n"
                "Iteration: {iteration}\n\n"
                "Draft:\n{draft}",
            ),
        ]
    )

    messages = prompt.format_messages(
        topic=topic,
        tone=tone,
        constraints=constraints,
        iteration=iteration,
        draft=draft,
    )

    structured_llm = critic_llm.with_structured_output(CriticScore)
    result: CriticScore = structured_llm.invoke(messages)

    # Store current score
    last_score = float(result.overall_score)
    state["last_score"] = last_score
    state["critic_feedback"] = result.short_feedback

    # Track score history
    scores_list = state.get("confidence_scores", [])
    scores_list.append(last_score)
    state["confidence_scores"] = scores_list

    # Update best draft if this is better
    best_score = state.get("best_score", 0.0)
    if last_score > best_score:
        state["best_score"] = last_score
        state["best_draft"] = draft

    # Accumulate feedback for future iterations
    mistake_memory = state.get("mistake_memory", [])
    mistake_memory.append(result.short_feedback)
    state["mistake_memory"] = mistake_memory

    return state