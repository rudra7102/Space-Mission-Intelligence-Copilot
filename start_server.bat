@echo off
title SpaceCopilot Server
echo ============================================
echo   SpaceCopilot - Starting Server...
echo ============================================
echo.

:: Kill anything on port 8000
for /f "tokens=5" %%a in ('netstat -aon ^| find ":8000" ^| find "LISTENING"') do (
    echo Stopping existing process on port 8000 (PID: %%a)
    taskkill /f /pid %%a >nul 2>&1
)

echo Starting SpaceCopilot on http://localhost:8000
echo Press Ctrl+C to stop.
echo.

python -m uvicorn serving.serve:app --host 0.0.0.0 --port 8000 --reload

pause
