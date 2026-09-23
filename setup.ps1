$ErrorActionPreference = "Stop"

$venv = ".venv"
if (-not (Test-Path $venv)) {
    python -m venv $venv
}
& "$venv/Scripts/pip" install -U pip
& "$venv/Scripts/pip" install -r requirements.lock
& "$venv/Scripts/pip" install -e ".[dev]"

Write-Host "Setup complete. Activate with: .venv\Scripts\Activate.ps1"
