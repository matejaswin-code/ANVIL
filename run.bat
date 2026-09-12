@echo off
REM ANVIL - start the local server.
REM Uses the project-local venv; never touches a global Python.

setlocal
set "PROJECT_ROOT=%~dp0"
set "PROJECT_ROOT=%PROJECT_ROOT:~0,-1%"
set "VENV_PY=%PROJECT_ROOT%\essential_tools\venv\Scripts\python.exe"

if not exist "%VENV_PY%" (
    echo Local environment not found. Run this first:
    echo   essential_tools\setup.bat
    pause
    exit /b 1
)

cd /d "%PROJECT_ROOT%"

REM Keep downloaded model weights inside the project instead of the global
REM HuggingFace cache in %%USERPROFILE%%\.cache.
set "HF_HOME=%PROJECT_ROOT%\models\hf"

echo.
echo   ANVIL starting on http://127.0.0.1:8765
echo   Ctrl-C to stop.
echo.

start "" http://127.0.0.1:8765
"%VENV_PY%" -m server.app %*

if %errorlevel% neq 0 (
    echo.
    echo   ANVIL exited with an error - see above.
    pause
)
endlocal
