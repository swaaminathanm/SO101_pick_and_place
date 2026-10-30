@echo off
setlocal

:: Path to Isaac Sim Python
set ISAAC_PYTHON=C:\isaacsim\python.bat

:: Verify Isaac Sim exists
if not exist "%ISAAC_PYTHON%" (
    echo.
    echo [ERROR] Could not find Isaac Sim at "%ISAAC_PYTHON%"!
    exit /b 1
)

:: Check if a script was provided
if "%~1"=="" (
    echo.
    echo ========================================================
    echo  ERROR: No Python script specified!
    echo ========================================================
    echo  Usage:   run.bat ^<path_to_script.py^> [arguments]
    echo  Example: run.bat scripts\run_env.py
    echo           run.bat scripts\teleop.py --headless
    echo ========================================================
    echo.
    exit /b 1
)

:: Forward all arguments to Isaac Sim's python.bat
call "%ISAAC_PYTHON%" %*

endlocal