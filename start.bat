@echo off
REM One-click setup + launch for Conjure on Windows.
REM Double-click this file, or run it from PowerShell / cmd: start.bat
REM It just finds your Python and hands off to run.py, which does everything.
setlocal
cd /d "%~dp0"

REM Prefer the "py" launcher — it picks a suitable installed Python version.
where py >nul 2>nul
if %errorlevel%==0 (
    py -3 run.py %*
    goto :end
)

where python >nul 2>nul
if %errorlevel%==0 (
    python run.py %*
    goto :end
)

echo.
echo Python was not found on this PC.
echo Install Python 3.10 or newer from https://python.org
echo and tick "Add python.exe to PATH" during setup, then run this again.
echo.
pause
goto :eof

:end
if not "%errorlevel%"=="0" (
    echo.
    echo Something went wrong above. Read the message, fix it, then run this again.
    pause
)
endlocal
