import asyncio
import aiohttp
import json
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.live import Live
from rich.markdown import Markdown

async def stream_from_api(prompt: str):
    """Connects to the local FastAPI gateway and streams real tokens."""
    url = "http://127.0.0.1:8000/v1/completions/stream"
    payload = {"prompt": prompt, "max_new_tokens": 150}
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    yield f"[bold red]Error: API returned {response.status}[/bold red]"
                    return
                
                # Iterate over the SSE stream
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    if line.startswith("data: "):
                        token = line[len("data: "):]
                        yield token
    except Exception as e:
        yield f"[bold red]Error: Could not connect to API. (Make sure 'uvicorn deploy.api:app' is running in another terminal)[/bold red]"

async def run_tui():
    console = Console()
    console.print(Panel("[bold green]FastGPT-Lab Terminal UI (v1.0.0)[/bold green]\n[dim]Connected to local Inference Gateway[/dim]", style="green"))

    while True:
        user_input = Prompt.ask("\n[bold blue]User[/bold blue]")
        if user_input.lower() in ['exit', 'quit']:
            break

        console.print("[bold yellow]Model:[/bold yellow] ", end="")
        
        # Real-time streaming output
        async for chunk in stream_from_api(user_input):
            console.print(chunk, end="", style="bold white")
            
        console.print("\n")

if __name__ == "__main__":
    try:
        asyncio.run(run_tui())
    except KeyboardInterrupt:
        print("\nExiting TUI.")
