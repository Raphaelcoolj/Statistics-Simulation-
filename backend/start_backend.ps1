# Start the FastAPI backend server
Write-Host "Starting FastAPI backend on http://127.0.0.1:8000 ..." -ForegroundColor Cyan
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
