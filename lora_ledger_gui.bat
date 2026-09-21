@echo off
setlocal
cd /d "%~dp0"
if exist "venv\Scripts\python.exe" (
    "venv\Scripts\python.exe" lora_ledger_gui.py %*
) else (
    python lora_ledger_gui.py %*
)
if errorlevel 1 pause
endlocal
