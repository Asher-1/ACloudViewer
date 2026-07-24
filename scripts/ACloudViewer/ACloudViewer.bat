@echo off
set SCRIPT_PATH=%~dp0
cd /d "%SCRIPT_PATH%"
set PATH=%SCRIPT_PATH%;%SCRIPT_PATH%lib;%PATH%
if exist "%SCRIPT_PATH%lib\cuda-runtime" set PATH=%SCRIPT_PATH%lib\cuda-runtime;%PATH%

rem Check if any argument requires CLI/headless mode (foreground execution).
set _CLI_MODE=0
for %%A in (%*) do (
    if /I "%%~A"=="-SILENT" set _CLI_MODE=1
    if /I "%%~A"=="--version" set _CLI_MODE=1
    if /I "%%~A"=="-v" set _CLI_MODE=1
    if /I "%%~A"=="--help" set _CLI_MODE=1
    if /I "%%~A"=="-h" set _CLI_MODE=1
)

if %_CLI_MODE%==1 (
    "%SCRIPT_PATH%ACloudViewer.exe" %*
    exit /b %ERRORLEVEL%
) else (
    start /b "" "%SCRIPT_PATH%ACloudViewer.exe" %* >nul 2>nul
    exit
)
