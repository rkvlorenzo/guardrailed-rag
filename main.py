from dotenv import load_dotenv
from rich.align import Align
from rich.panel import Panel
from rich.text import Text

load_dotenv()

def generate_welcome_panel():
    welcome_text = Text()
    welcome_text.append("Welcome to RAGuard!\n", style="bold green")
    welcome_text.append("Your AI-powered retrieval assistant.\n", style="italic cyan")
    welcome_text.append("Type 'exit' or 'quit' to leave.\n", style="yellow")

    panel = Panel(
        Align.center(welcome_text, vertical="middle"),
        title="[bold magenta] RAGuard CLI [/bold magenta]",
        border_style="bright_blue",
        padding=(1, 5),
        expand=False
    )

    return panel

def main():
    from rich.console import Console
    from app.rag_pipeline import rag_pipeline_response
    from app.ingestion import initialize_vector

    console = Console()
    console.print(generate_welcome_panel())

    with console.status(
        "[bold yellow]Loading documents and building vector store, please wait...[/bold yellow]",
        spinner="dots"
    ):
        initialize_vector()

    while True:
        user_input = console.input("\n[bold blue]Enter your question: [/bold blue]")
        if user_input.lower() in ["exit", "quit"]:
            console.print("\n[bold red]Thanks for using RAGuard![/bold red]")
            break

        with console.status(
            "[bold green]Generating answer...[/bold green]",
            spinner="dots"
        ):
            response = rag_pipeline_response(user_input=user_input)
        console.print(f"\n[bold magenta]RAGuard: [/bold magenta] {response}")



if __name__ == "__main__":
    main()