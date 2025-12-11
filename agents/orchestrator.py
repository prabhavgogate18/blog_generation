from graph.state import BlogState

# Confidence threshold for early stopping (0.80 = 80%)
CONFIDENCE_THRESHOLD: float = 0.80


def orchestrator_node(state: BlogState) -> BlogState:
    """
    Orchestrator decides whether to continue another iteration or stop.

    Logic:
    - iteration is incremented on every orchestrator call
    - EARLY STOP when last_score >= CONFIDENCE_THRESHOLD (e.g. 0.80 → 80%)
    - otherwise HARD STOP when iteration >= max_iterations
    - no LLM is used here: this is a pure rule-based controller
    """

    # Scores set by critic
    last_score: float = state.get("last_score", 0.0)      # 0–1
    best_score: float = state.get("best_score", 0.0)      # 0–1

    # Loop control
    iteration: int = state.get("iteration", 0)
    max_iterations: int = state.get("max_iterations", 5)

    # Update iteration counter
    if iteration <= 0:
        new_iteration = 1
    else:
        new_iteration = iteration + 1

    state["iteration"] = new_iteration

    # EARLY STOP: confidence >= threshold (e.g. 80%)
    if last_score >= CONFIDENCE_THRESHOLD:
        state["route"] = "done"
        state["stop_reason"] = (
            f"Early stop: confidence reached {last_score * 100:.1f}% "
            f"(threshold {CONFIDENCE_THRESHOLD * 100:.0f}%)"
        )
        return state

    # HARD STOP: max iterations reached
    if new_iteration >= max_iterations:
        state["route"] = "done"
        state["stop_reason"] = (
            f"Max iterations ({max_iterations}) reached without hitting threshold "
            f"{CONFIDENCE_THRESHOLD * 100:.0f}% "
            f"(best score {best_score * 100:.1f}%)"
        )
        return state

    # OTHERWISE: continue loop
    state["route"] = "continue"
    state["stop_reason"] = (
        f"Below confidence threshold ({last_score * 100:.1f}% < "
        f"{CONFIDENCE_THRESHOLD * 100:.0f}%), iterations remaining"
    )
    return state