import asyncio
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.live import Live
from rich.markdown import Markdown

# Mock client for the TUI to simulate SSE connection to API
class StreamClient:
    async def simulate_stream(self, prompt: str):
        words = ["This ", "is ", "the ", "elite ", "model ", "speaking ", "from ", "the ", "terminal", ".", "\n"]
        for word in words:
            await asyncio.sleep(0.1)
            yield word

async def run_tui():
    console = Console()
    client = StreamClient()

    console.print(Panel("[bold green]FastGPT-Lab Terminal UI[/bold green]\nConnects to FastAPI SSE backends for real-time inference.", style="green"))

    while True:
        user_input = Prompt.ask("\n[bold blue]User[/bold blue]")
        if user_input.lower() in ['exit', 'quit']:
            break

        console.print("[bold yellow]Model:[/bold yellow] ", end="")
        
        response_text = ""
        # In a real scenario, this would use aiohttp to connect to http://localhost:8000/v1/completions/stream
        async for chunk in client.simulate_stream(user_input):
            response_text += chunk
            console.print(chunk, end="", style="bold white")
            
        console.print("\n")

if __name__ == "__main__":
    try:
        asyncio.run(run_tui())
    except KeyboardInterrupt:
        print("\nExiting TUI.")
