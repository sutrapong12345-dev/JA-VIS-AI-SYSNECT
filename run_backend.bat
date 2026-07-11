@echo off
echo ===================================
echo   Starting JARVIS AI Backend
echo ===================================

REM --- Ollama ---
tasklist /fi "ImageName eq ollama.exe" 2>NUL | find /i "ollama.exe" >NUL
if "%ERRORLEVEL%"=="1" (
    echo [INFO] Ollama is not running. Starting Ollama...
    if exist "%LOCALAPPDATA%\Programs\Ollama\ollama app.exe" (
        start "" "%LOCALAPPDATA%\Programs\Ollama\ollama app.exe"
    ) else (
        start "" ollama serve
    )
    timeout /t 5 /nobreak >nul
)

REM --- Stop any backend already holding port 8000 (incl. child processes) ---
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":8000" ^| findstr "LISTENING"') do (
    echo [INFO] Port 8000 busy - killing PID %%p and its children
    taskkill /F /T /PID %%p >nul 2>&1
)

REM --- venv ---
if not exist "%~dp0venv\Scripts\python.exe" (
    echo [INFO] Creating Python Virtual Environment...
    python -m venv "%~dp0venv"
    "%~dp0venv\Scripts\python.exe" -m pip install -r "%~dp0backend\requirements.txt"
)

echo [INFO] Starting FastAPI Server (no reload - use restart_backend.bat after code changes)...
cd /d "%~dp0backend"
REM Always use the venv python explicitly. "call activate" + plain python can
REM silently fall back to the system Python and miss installed packages.
"%~dp0venv\Scripts\python.exe" -m uvicorn main:app --host 127.0.0.1 --port 8000
pause
