@echo off
REM 手动启动 novel-voice-cast 流水线只读进度面板
REM
REM 背景：该面板原本通过 Windows 计划任务 NovelVoiceCast-ProgressMonitor
REM 在每次登录时自动启动（会弹出控制台窗口），已于 2026-09-29 禁用自启动。
REM 需要时用本脚本手动启动，用完关掉窗口即可。
REM
REM 面板地址：http://127.0.0.1:8765
REM 只读设计：不与流水线进程通信，只读 manifest/checkpoint，启停不影响续跑。

cd /d "%~dp0"
title novel-voice-cast 进度面板 (Ctrl+C 退出)

echo.
echo   进度面板启动中...
echo   浏览器打开: http://127.0.0.1:8765
echo   按 Ctrl+C 关闭面板
echo.

.venv\Scripts\python.exe -u scripts\progress_monitor.py --config config\config.yaml --host 127.0.0.1 --port 8765

echo.
echo   面板已关闭。
pause
