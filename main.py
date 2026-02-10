from dotenv import load_dotenv
load_dotenv()

def main():
    from rich.console import Console
    from app.ingestion import convert_files_to_vector
    from app.rag_pipeline import evaluation_temp

    console = Console()
    console.print("[bold green]Welcome to RAGuard! [/bold green] [italic]Type 'exit' or 'quit' to leave.[/italic]")

    with console.status(f"[bold yellow]Loading documents and building vector store, please wait......[/bold yellow]",
                        spinner="dots"):
        global_vector = convert_files_to_vector()
        retriever = global_vector.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 4}
        )

    while True:
        user_input = console.input("[bold blue]Enter your question: [/bold blue]")
        if user_input.lower() in ["exit", "quit"]:
            console.print("[bold red]Thanks for using RAGuard![/bold red]")
            break

        # response = rag_pipeline_response(user_input=user_input, retriever=retriever)
        x, response = evaluation_temp(user_input=user_input, retriever=retriever)
        console.print(f"[bold magenta]RAGuard: [/bold magenta] {response}")
        console.print(f"[bold magenta]RAGuard: [/bold magenta] {x}")


if __name__ == "__main__":
    main()