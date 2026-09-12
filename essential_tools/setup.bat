@echo off
REM ============================================================================
REM  ANVIL - local environment setup (Windows)
REM
REM  Creates a project-local Python venv and a project-local node_modules.
REM  Nothing is installed globally. The only things you need already on your
REM  machine are Python and Node.js themselves.
REM
REM    Usage:  essential_tools\setup.bat
REM ============================================================================

setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
pushd "%SCRIPT_DIR%\.." >nul
set "PROJECT_ROOT=%CD%"
popd >nul
set "VENV_DIR=%SCRIPT_DIR%\venv"

echo.
echo    ###   #   # #   # ### #
echo   #   #  ##  # #   #  #  #
echo   #####  # # # #   #  #  #
echo   #   #  #  ##  # #   #  #
echo   #   #  #   #   #   ### #####
echo   local-first AI 3D asset pipeline
echo.

REM ---------------------------------------------------------------------------
echo ==^> Checking prerequisites
REM ---------------------------------------------------------------------------

set "PYTHON_BIN="
for %%P in (python py python3) do (
    if not defined PYTHON_BIN (
        where %%P >nul 2>&1
        if !errorlevel! equ 0 (
            %%P -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)" >nul 2>&1
            if !errorlevel! equ 0 set "PYTHON_BIN=%%P"
        )
    )
)

if not defined PYTHON_BIN (
    echo.
    echo   [X] Python 3.10 or newer is required but wasn't found.
    echo.
    echo       Install it from https://www.python.org/downloads/
    echo       IMPORTANT: tick "Add Python to PATH" during installation.
    echo.
    echo       This is the one dependency this script can't install for you,
    echo       since it's the thing that runs everything else.
    echo.
    exit /b 1
)

for /f "delims=" %%V in ('%PYTHON_BIN% --version 2^>^&1') do set "PYVER=%%V"
echo   [+] Python: !PYVER!

set "NODE_OK=0"
where node >nul 2>&1
if %errorlevel% equ 0 (
    for /f "delims=" %%V in ('node -p "process.versions.node.split('.')[0]" 2^>nul') do set "NODE_MAJOR=%%V"
    if !NODE_MAJOR! geq 18 (
        set "NODE_OK=1"
        for /f "delims=" %%V in ('node --version') do echo   [+] Node.js: %%V
    ) else (
        echo   [!] Node.js found but 18+ is needed for the 3D preview. Skipping it.
    )
) else (
    echo   [!] Node.js not found - the browser 3D preview will be disabled.
    echo   [!] Everything else works. Install Node 18+ and re-run to enable it.
)

REM ---------------------------------------------------------------------------
echo.
echo ==^> Creating the local Python environment
REM ---------------------------------------------------------------------------

if exist "%VENV_DIR%\Scripts\python.exe" (
    echo   [+] Reusing existing venv at essential_tools\venv
) else (
    %PYTHON_BIN% -m venv "%VENV_DIR%"
    if !errorlevel! neq 0 (
        echo   [X] Failed to create the virtual environment.
        exit /b 1
    )
    echo   [+] Created essential_tools\venv
)

set "VENV_PY=%VENV_DIR%\Scripts\python.exe"
if not exist "%VENV_PY%" (
    echo   [X] venv created but %VENV_PY% is missing.
    echo       Delete essential_tools\venv and re-run.
    exit /b 1
)

"%VENV_PY%" -m pip install --quiet --upgrade pip setuptools wheel
echo   [+] pip upgraded inside the venv

REM ---------------------------------------------------------------------------
echo.
echo ==^> Installing Python dependencies (local to the venv)
REM ---------------------------------------------------------------------------

cd /d "%PROJECT_ROOT%"
"%VENV_PY%" -m pip install --quiet -r requirements.txt
if %errorlevel% neq 0 (
    echo   [X] Dependency installation failed. Scroll up for the error.
    exit /b 1
)
echo   [+] Installed from requirements.txt
echo       Nothing was installed outside essential_tools\venv.

REM ---------------------------------------------------------------------------
echo.
echo ==^> Installing frontend vendor files (local to node_modules)
REM ---------------------------------------------------------------------------

if "%NODE_OK%"=="1" (
    call npm install --prefix "%PROJECT_ROOT%" --silent --no-audit --no-fund
    node "%SCRIPT_DIR%\vendor.js"
    echo   [+] three.js vendored into frontend\vendor\
) else (
    if not exist "%PROJECT_ROOT%\frontend\vendor" mkdir "%PROJECT_ROOT%\frontend\vendor"
    echo   [!] Skipped - the app runs fine, you just won't get the mesh preview.
)

REM ---------------------------------------------------------------------------
echo.
echo ==^> Preparing configuration
REM ---------------------------------------------------------------------------

if exist "%PROJECT_ROOT%\.env" (
    echo   [+] .env already exists - leaving it alone
) else (
    copy /y "%PROJECT_ROOT%\.env.example" "%PROJECT_ROOT%\.env" >nul
    echo   [+] Created .env from .env.example
    echo   [!] Open .env and fill in your model paths / API keys before real generation.
)

if not exist "%PROJECT_ROOT%\sessions" mkdir "%PROJECT_ROOT%\sessions"
if not exist "%PROJECT_ROOT%\assets"   mkdir "%PROJECT_ROOT%\assets"
if not exist "%PROJECT_ROOT%\logs"     mkdir "%PROJECT_ROOT%\logs"
echo   [+] Created sessions\, assets\, logs\

REM ---------------------------------------------------------------------------
echo.
echo ==^> Verifying the install
REM ---------------------------------------------------------------------------

"%VENV_PY%" "%SCRIPT_DIR%\verify.py"
if %errorlevel% neq 0 (
    echo   [!] Verification reported problems - see above. The install itself completed.
)

echo.
echo ============================================================
echo   Setup complete.
echo.
echo     Start ANVIL:   essential_tools\run.bat
echo     Then open:     http://127.0.0.1:8765
echo.
echo   Everything lives inside this project directory:
echo     essential_tools\venv\   Python environment
echo     node_modules\           frontend packages
echo     sessions\               generated assets + checkpoints
echo.
echo   Out of the box ANVIL runs on MOCK BACKENDS - the full 10-step
echo   pipeline works on CPU with no model weights, so you can confirm
echo   everything is wired up before downloading anything. Switch to
echo   real backends in .env or the Settings panel when you're ready.
echo ============================================================
echo.

endlocal
