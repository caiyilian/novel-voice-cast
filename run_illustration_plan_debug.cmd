@echo off
chcp 65001 >nul
setlocal

cd /d "%~dp0"

if /I "%~1"=="help" goto :help
if /I "%~1"=="--help" goto :help

echo [START] Novel Voice Cast illustration planner
echo [CWD] %CD%

set "PYTHONUTF8=1"
set "AGNES_PROXY_URL=http://127.0.0.1:7890"
set "PYTHON_EXE=%CD%\.venv\Scripts\python.exe"
set "EXIT_CODE=1"

if not exist "%PYTHON_EXE%" (
    echo [ERROR] Python not found: %PYTHON_EXE%
    goto :end
)

if not exist "logs" mkdir logs

if /I "%~1"=="resume" (
    set "RUN_MODE=--resume"
    set "LOG_FILE=logs\illustration_plan_resume_latest.log"
    echo [MODE] resume from checkpoint
) else (
    set "RUN_MODE=--fresh"
    set "LOG_FILE=logs\illustration_plan_latest.log"
    echo [MODE] fresh run, old checkpoint will be deleted
)

echo [PYTHON] %PYTHON_EXE%
echo [LOG] %LOG_FILE%
echo [VISUAL_MEMORY] output\character_visual_memory.json
echo [PROXY] %AGNES_PROXY_URL%
echo.

"%PYTHON_EXE%" -u scripts\run_illustration_plan.py ^
    --novel novels\novel.txt ^
    --labels novels\labels.txt ^
    --output output\illustration_plan.json ^
    %RUN_MODE% ^
    --debug ^
    --visual-memory-output output\character_visual_memory.json ^
    --log-file "%LOG_FILE%"

set "EXIT_CODE=%ERRORLEVEL%"
echo.
echo [DONE] Exit code: %EXIT_CODE%
goto :end

:help
echo Usage:
echo   run_illustration_plan_debug.cmd          Fresh run, deletes old checkpoint
echo   run_illustration_plan_debug.cmd resume   Resume from checkpoint
echo.
goto :eof

:end
echo.
echo Press any key to close this window...
pause >nul
exit /b %EXIT_CODE%
