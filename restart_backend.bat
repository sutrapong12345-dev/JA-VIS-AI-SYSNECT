@echo off
REM ============================================================
REM   Restart JARVIS backend cleanly (loads the latest .env)
REM   Use this whenever you change backend\.env (e.g. ACTIVE_AI)
REM ============================================================
echo [1/3] Stopping any running backend on port 8000...
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":8000" ^| findstr "LISTENING"') do (
    echo   killing PID %%p and its child processes
    REM /T kills the whole tree - uvicorn reload mode spawns a child
    REM worker that otherwise survives and keeps holding port 8000
    taskkill /F /T /PID %%p >nul 2>&1
)
timeout /t 2 /nobreak >nul

echo [2/3] Starting backend and checking Ollama...
tasklist /fi "ImageName eq ollama.exe" 2>NUL | find /i "ollama.exe" >NUL
if "%ERRORLEVEL%"=="1" (
    echo   Ollama is not running. Starting Ollama...
    if exist "%LOCALAPPDATA%\Programs\Ollama\ollama app.exe" (
        start "" "%LOCALAPPDATA%\Programs\Ollama\ollama app.exe"
    ) else (
        start "" ollama serve
    )
)

cd /d "%~dp0backend"
start "JARVIS Backend" "%~dp0venv\Scripts\python.exe" -m uvicorn main:app --host 127.0.0.1 --port 8000

echo [3/3] Done. Backend is starting in a new window.
echo Refresh the web page (Ctrl+F5) in a few seconds.
timeout /t 3 /nobreak >nul
