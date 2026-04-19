# --- setup.ps1 for Windows ---
Write-Host "🚀 Setting up fast-gpt-lab environment..." -ForegroundColor Cyan

# Check if uv is installed
if (!(Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Error "uv not found. Please install it from https://github.com/astral-sh/uv"
    exit 1
}

# Create venv and sync dependencies
Write-Host "📦 Creating virtual environment and syncing dependencies..."
uv venv
uv sync

Write-Host "✅ Setup complete! Use 'uv run python' to execute scripts." -ForegroundColor Green
