# agents/guardrails.py
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from typing import List, Optional

from config.settings import get_critic_llm
from utils.prompt_loader import load_prompt
from graph.state import BlogState

# deterministic guard model
guard_llm = get_critic_llm(temperature=0.0)


class GuardrailsOutput(BaseModel):
    valid: bool = Field(description="Whether the inputs pass guardrails (True/False).")
    topic: str = Field(description="Sanitized / possibly rewritten topic.")
    constraints: Optional[str] = Field(default="", description="Sanitized / rewritten constraints.")
    issues: List[str] = Field(default_factory=list, description="List of detected issues (if any).")
    corrective_action: Optional[str] = Field(
        default="",
        description="Instruction for user or next steps (e.g., 'Please rephrase', 'Removed personal data', etc.)"
    )


def guardrails_node(state: BlogState) -> BlogState:
    """
    Validate and sanitize user inputs (topic, constraints) using the guard LLM.
    If invalid, set state['route'] = 'done' so graph can exit gracefully.
    Writes back:
      - state['topic'] (possibly modified),
      - state['constraints'] (possibly modified),
      - state['guardrails_issues'] (list),
      - state['guardrails_valid'] (bool),
      - state['guardrails_action'] (string)
    """
    raw_topic = state.get("topic", "")
    raw_constraints = state.get("constraints", "")

    system_prompt = load_prompt("guardrails_system")

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "human",
                "Validate and sanitize the following user inputs. Respond with structured JSON matching the GuardrailsOutput schema.\n\n"
                "Topic: {topic}\n\n"
                "Constraints: {constraints}\n\n"
                "Return only JSON."
            ),
        ]
    )

    # Format messages (safe now because guardrails_system.txt uses double braces)
    try:
        messages = prompt.format_messages(topic=raw_topic, constraints=raw_constraints)
    except Exception as e:
        # formatting failed — conservative fallback
        state["guardrails_valid"] = False
        state["guardrails_issues"] = [f"prompt formatting failed: {e}"]
        state["guardrails_action"] = "Please rephrase the topic or constraints."
        # halt pipeline
        state["route"] = "done"
        return state

    structured_llm = guard_llm.with_structured_output(GuardrailsOutput)

    try:
        result: GuardrailsOutput = structured_llm.invoke(messages)
    except Exception as e:
        # conservative fallback: mark invalid and request clarification
        state["guardrails_valid"] = False
        state["guardrails_issues"] = [f"guardrails invocation failed: {e}"]
        state["guardrails_action"] = "Please rephrase the topic or constraints."
        state["route"] = "done"
        return state

    # Apply sanitized values back to state
    state["topic"] = result.topic.strip()
    state["constraints"] = (result.constraints or "").strip()
    state["guardrails_issues"] = result.issues
    state["guardrails_valid"] = bool(result.valid)
    state["guardrails_action"] = result.corrective_action or ""

    # If inputs invalid, halt the pipeline by setting route to 'done'
    if not state["guardrails_valid"]:
        # if invalid, ensure the orchestrator will see route 'done' and stop.
        state["route"] = "done"

    return state
