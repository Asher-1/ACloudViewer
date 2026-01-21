@echo off
set SCRIPT_PATH=%~dp0
cd /d "%SCRIPT_PATH%"
set PATH=%SCRIPT_PATH%;%SCRIPT_PATH%lib;%PATH%
start /b "" "%SCRIPT_PATH%ACloudViewer.exe" >nul 2>nul
exit