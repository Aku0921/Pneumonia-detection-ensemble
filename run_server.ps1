# run_server.ps1 - Start the Pneumonia Detection API server

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Pneumonia Detection API" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if venv exists
if (-not (Test-Path "./venv")) {
    Write-Host "Virtual environment not found. Creating..." -ForegroundColor Yellow
    python -m venv venv
}

# Activate venv
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
. ./venv/Scripts/Activate.ps1

# Install dependencies
Write-Host "Installing dependencies..." -ForegroundColor Yellow
pip install --upgrade pip -q
pip install -r requirements.txt -q
pip install fastapi uvicorn[standard] python-multipart -q

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "✓ Environment ready" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Starting API server..." -ForegroundColor Cyan
Write-Host "🌐 Open your browser: http://localhost:8000" -ForegroundColor Green
Write-Host "📚 API Docs: http://localhost:8000/docs" -ForegroundColor Green
Write-Host ""

# Run the server
python -m uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload
