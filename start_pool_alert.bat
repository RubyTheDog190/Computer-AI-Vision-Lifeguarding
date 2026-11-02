@echo off
setlocal EnableExtensions
set "APP_DIR=%~dp0"
set "ENV_DIR=%USERPROFILE%\.conda\envs\env_yolov5"
set "PYTHON_EXE=%ENV_DIR%\python.exe"
set "PYTHONW_EXE=%ENV_DIR%\pythonw.exe"
set "LOG_FILE=%APP_DIR%pool_risk_gui_startup.log"

if not exist "%PYTHON_EXE%" (
    set "ENV_DIR=%USERPROFILE%\anaconda3\envs\env_yolov5"
    set "PYTHON_EXE=%USERPROFILE%\anaconda3\envs\env_yolov5\python.exe"
    set "PYTHONW_EXE=%USERPROFILE%\anaconda3\envs\env_yolov5\pythonw.exe"
)

if not exist "%PYTHON_EXE%" (
    echo ERROR: env_yolov5 Python was not found.>"%LOG_FILE%"
    echo Expected under: %USERPROFILE%\.conda\envs\env_yolov5>>"%LOG_FILE%"
    type "%LOG_FILE%"
    pause
    exit /b 1
)

set "PATH=%ENV_DIR%;%ENV_DIR%\Scripts;%ENV_DIR%\Library\bin;%ENV_DIR%\Library\usr\bin;%PATH%"
set "QT_PLUGIN_PATH=%ENV_DIR%\Library\plugins"
set "QT_QPA_PLATFORM_PLUGIN_PATH=%ENV_DIR%\Library\plugins\platforms"
set "PYTHONUTF8=1"

cd /d "%APP_DIR%"
if not exist "%PYTHONW_EXE%" set "PYTHONW_EXE=%PYTHON_EXE%"
start "Pool Risk Alert" "%PYTHONW_EXE%" "%APP_DIR%pool_risk_gui.py"
if errorlevel 1 (
    echo ERROR: Windows could not launch the application.>"%LOG_FILE%"
    pause
    exit /b 1
)
exit /b 0
