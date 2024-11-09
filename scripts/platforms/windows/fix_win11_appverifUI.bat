@echo off
setlocal

set "COMP=HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Windows\CurrentVersion\Installer\UserData\S-1-5-18\Components"
set "OUT_DIR=C:\Windows\SysWOW64"
set "VAL=0E4DCE7FFF04CE369632E0E2BED6EBFC"
goto :Process

:Fix
set "KEY=%COMP%\%1"
set "OLD_FILE=%2"
set "NEW_FILE=%3"

if exist %OLD_FILE% (
    move %OLD_FILE% %NEW_FILE% && echo.%OLD_FILE% successfully moved to %NEW_FILE%

    for /f "tokens=2*" %%A in ('reg query "%KEY%" /v "%VAL%" 2^>nul') do (
        if "%%A"=="REG_SZ" if "%%B"=="%OLD_FILE%" (
            reg add "%KEY%" /v "%VAL%" /t REG_SZ /d "%NEW_FILE%" /f && echo.Registry component for %KEY%\%VAL% fixed
        )
    )
) else (
    echo.%OLD_FILE% did not exist.
)
goto :EOF

:Process
call :Fix "0AF818DE4685190F5347FAF54BD80C82" "C:\appverifUI.dll" "%OUT_DIR%\appverifUI.dll"
call :Fix "E0C37F311948CF28AE087B694A681271" "C:\vfcompat.dll" "%OUT_DIR%\vfcompat.dll"

endlocal
