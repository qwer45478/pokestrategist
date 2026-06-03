@echo off
REM ============================================================
REM  pokestrategist Showdown 本地服务启动器
REM  双击此文件即可打开 GUI 配置界面
REM ============================================================

cd /d "%~dp0..\.."

REM 尝试激活 pokestrategist conda 环境
set CONDA_ENV=pokestrategist
where conda >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    call conda activate %CONDA_ENV% 2>nul
    if %ERRORLEVEL% EQU 0 (
        echo [INFO] Conda 环境 %CONDA_ENV% 已激活
    ) else (
        echo [WARN] 无法激活 conda 环境 %CONDA_ENV%，使用系统默认 Python
    )
)

REM 启动 GUI 启动器
python "%~dp0launch_gui.py"
pause
