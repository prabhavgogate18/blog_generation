from typing import TypedDict, List, Literal
from typing_extensions import NotRequired


class BlogState(TypedDict, total=False):
    # User inputs
    topic: str
    word_count: int
    tone: str
    constraints: str

    # Data produced by agents
    research_notes: str
    draft: str
    critic_feedback: str

    # Loop control
    iteration: int
    max_iterations: int
    route: NotRequired[Literal["continue", "done"]]
    stop_reason: str

    # Scoring (simplified)
    last_score: float  # 0-1 scale, set by critic
    best_score: float  # 0-1 scale, highest score achieved
    best_draft: str    # draft with the best score
    
    # History tracking (for display/debugging)
    confidence_scores: List[float]  # 0-1 scale
    
    # Memory of mistakes across iterations
    mistake_memory: List[str]