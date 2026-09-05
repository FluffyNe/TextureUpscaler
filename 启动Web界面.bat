@echo off
chcp 65001 >nul
title TextureUpscaler Web
cd /d "%~dp0"

rem 优先使用自带 Python 环境（python_embed），无需另装 Python
set "PY=%~dp0python_embed\python.exe"
if not exist "%PY%" (
    set "PY=python"
    where python >nul 2>nul
    if errorlevel 1 (
        echo [错误] 未找到自带 Python，也未检测到系统 Python
        pause
        exit /b 1
    )
)

rem 校验依赖，缺失时自动安装（自带环境已预装，通常无需联网）
"%PY%" -c "import flask, PIL, numpy" >nul 2>nul
if errorlevel 1 (
    echo 首次运行，正在安装依赖 flask / pillow / numpy ...
    "%PY%" -m pip install -r "%~dp0web\requirements.txt"
    if errorlevel 1 (
        echo [错误] 依赖安装失败，请检查网络
        pause
        exit /b 1
    )
)

rem 延迟 2 秒自动打开浏览器（等服务起来）
start "" /b cmd /c "timeout /t 2 /nobreak >nul & start http://127.0.0.1:5000"

cd /d "%~dp0web"
"%PY%" app.py
pause
