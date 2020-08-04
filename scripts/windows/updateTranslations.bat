set QT_DIR=D:\develop\software\Qt\Qt5.13.2\5.13.2\msvc2017_64\bin

set SRC_DIRS=common eCV plugins/core libs/PCLEngine libs/cCV_db libs/OpenGLEngine libs/eCV_io

echo "Updating translation files"

%call lupdate
cd ..\..
%QT_DIR%\lupdate.exe %SRC_DIRS% -no-obsolete -ts eCV\translations\ErowCloudViewer_zh.ts
cd scripts\windows
