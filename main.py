from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from graph.builder import build_blog_graph
from graph.state import BlogState


console = Console()


def get_user_input() -> BlogState:
    console.print("[bold cyan]Blog Generator Agent[/bold cyan]")

    topic = input("Enter blog topic: ").strip()
    word_count_str = input("Target number of words (e.g. 800): ").strip()
    tone = input("Tone of the blog (e.g. friendly, professional, humorous): ").strip()
    constraints = input("Additional constraints (SEO terms, audience, etc.): ").strip()

    try:
        word_count = int(word_count_str)
    except ValueError:
        word_count = 800

    initial_state: BlogState = {
        "topic": topic,
        "word_count": word_count,
        "tone": tone,
        "constraints": constraints,
        "iteration": 0,
        "max_iterations": 5,
        "best_score": 0.0,
        "confidence_scores": [],
    }
    return initial_state


def main():
    load_dotenv()
    state = get_user_input()

    app = build_blog_graph()

    console.print("\n[bold yellow]Running multi-agent blog generator...[/bold yellow]")
    console.print("[dim]Note: Research is performed only once in the first iteration[/dim]")
    console.print("[dim]Subsequent iterations refine the blog based on critic feedback[/dim]\n")

    final_state: BlogState = app.invoke(state)

    best_draft = final_state.get("best_draft", "")
    best_score = final_state.get("best_score", 0.0)
    scores = final_state.get("confidence_scores", [])
    stop_reason = final_state.get("stop_reason", "Completed")

    console.print("\n[bold green]=== BEST BLOG (HIGHEST CONFIDENCE) ===[/bold green]\n")
    console.print(best_draft)

    console.print("\n[bold magenta]=== METRICS ===[/bold magenta]")
    console.print(f"Final confidence: {scores[-1] * 100:.1f}%" if scores else "N/A")
    if scores:
        console.print(
            "Confidence progression: "
            + " → ".join([f"{s * 100:.1f}%" for s in scores])
        )
    else:
        console.print("Confidence progression: [dim]no scores recorded[/dim]")
    console.print(f"Best score: {best_score:.3f} ({best_score * 100:.1f}%)")
    console.print(f"Total iterations: {final_state.get('iteration', 0)}")
    console.print(f"Stop reason: {stop_reason}")
    
    # Show efficiency gain
    iterations_used = final_state.get("iteration", 0)
    if iterations_used < final_state.get("max_iterations", 5):
        console.print(
            f"[green]✓ Stopped early! Saved "
            f"{final_state.get('max_iterations', 5) - iterations_used} iteration(s)[/green]"
        )

    # TABLE: per-iteration confidence
    if scores:
        console.print("\n[bold cyan]📊 Score Progression per Iteration:[/bold cyan]")

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Iteration", style="cyan", justify="center")
        table.add_column("Confidence", justify="center")

        for i, score in enumerate(scores):
            iteration_num = i + 1
            score_pct = score * 100.0
            table.add_row(str(iteration_num), f"{score_pct:.1f}%")

        console.print(table)

    console.print()


if __name__ == "__main__":
    main()