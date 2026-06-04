# pokestrategist 本地服务启动器

一键启动 pokestrategist Showdown 本地推理服务的图形界面工具。

## 功能

- 🔍 **自动发现 Checkpoint** — 扫描 `models/` 目录，自动列出所有可用模型
- 🎛️ **可视化配置** — 图形界面管理所有服务参数，无需记忆命令行
- 📋 **配置持久化** — 自动保存上次配置，下次启动无需重新填写
- 📡 **实时日志** — 内置终端风格的日志窗口，实时显示服务输出
- 🩺 **健康检查** — 一键检测服务是否正常运行
- 🟢 **状态指示** — 清晰显示服务运行/停止/异常状态

## 如何使用

### 方式一：双击启动（推荐）

直接双击 `launch.bat`。

### 方式二：PowerShell

```powershell
cd apps/launcher
.\launch.ps1
```

### 方式三：命令行

```powershell
cd apps/launcher
python launch_gui.py
```

## 配置项说明

| 配置项 | 说明 | 默认值 |
|---|---|---|
| Python 解释器 | pokestrategist conda 环境的 python.exe | 自动检测 |
| 模型 Checkpoint | 要加载的 v1 决策模型 .pt 文件 | 无（必选） |
| 监听地址 | HTTP 服务绑定的 IP | 127.0.0.1 |
| 端口 | HTTP 服务端口 | 8765 |
| 推理设备 | auto / cuda / cpu | auto |
| 建议数量 TopK | 返回前 K 个建议 | 3 |
| Team Preview 先验 | 是否使用队伍预览先验 | 开启 |
| 使用率先验 | 是否使用宝可梦使用率统计 | 开启 |
| 隐藏候选 TopK | 隐藏招式候选数量 | 留空=默认 |

## 启动后

服务启动后，Chrome 扩展会自动连接到 `http://127.0.0.1:8765`。
在 Pokemon Showdown 对战中即可看到 AI 建议面板。

如果扩展未连接，请在扩展弹窗中确认服务地址与启动器中的一致。
