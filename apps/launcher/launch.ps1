# ============================================================
#  pokestrategist Showdown 本地服务启动器
#  在 PowerShell 中运行此脚本以打开 GUI 配置界面
# ============================================================

param(
    [switch]$SkipConda
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path "$ScriptDir\..\.."

Push-Location $ProjectRoot

# 尝试激活 pokestrategist conda 环境
if (-not $SkipConda) {
    $condaHook = "$env:USERPROFILE\.conda\Scripts\conda.exe"
    if (Test-Path $condaHook) {
        try {
            conda activate pokestrategist 2>$null
            Write-Host "[INFO] Conda 环境 pokestrategist 已激活" -ForegroundColor Green
        } catch {
            Write-Host "[WARN] 无法激活 conda 环境 pokestrategist，使用系统默认 Python" -ForegroundColor Yellow
        }
    } else {
        # 尝试 Anaconda 路径
        $anacondaHook = "D:\anacoda3\shell\condabin\conda-hook.ps1"
        if (Test-Path $anacondaHook) {
            . $anacondaHook
            conda activate pokestrategist
            Write-Host "[INFO] Conda 环境 pokestrategist 已激活" -ForegroundColor Green
        } else {
            Write-Host "[WARN] 未找到 conda，使用系统默认 Python" -ForegroundColor Yellow
        }
    }
}

# 启动 GUI 启动器
python "$ScriptDir\launch_gui.py"

Pop-Location
