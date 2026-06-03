# pokestrategist Showdown Companion

这是一个最小可运行的 Chrome 扩展 MVP，用来把 pokestrategist 逐步做成真正可交互的工具。

当前版本已经打通“扩展 + 本地推理服务”的基本闭环，完成了五件事：

1. 在 Pokemon Showdown 页面自动注入一个侧边建议面板。
2. 从页面 DOM 中抓取当前 battle snapshot，包括 turn、可执行 move/switch、可见 active 信息和最近日志。
3. 将 snapshot 发送给一个本地 HTTP 接口，请求真实的 pokestrategist 模型建议。
4. 本地服务会加载现有 v1 checkpoint，把 snapshot 适配成模型输入，执行真实前向推理，并返回排序后的建议。
5. 如果本地服务未连通，则扩展会退回到一个明确标注为 fallback 的占位建议模式。

## 目录结构

- `manifest.json`: Chrome Manifest V3 配置。
- `background.js`: 配置存储与本地建议接口请求代理。
- `content.js`: Showdown 页面抓取、状态构造和注入面板。
- `content.css`: 面板样式。
- `popup.html` / `popup.js`: 扩展弹窗，用于配置本地接口地址和刷新间隔。

后端对应入口：

- `src/pokestrategist/serving/showdown.py`: snapshot 到模型输入的适配、checkpoint 加载和建议生成。
- `src/pokestrategist/cli/serve_showdown.py`: 本地 HTTP 服务入口。

## 如何加载

1. 打开 Chrome 的扩展管理页：`chrome://extensions/`
2. 开启开发者模式。
3. 选择“加载已解压的扩展程序”。
4. 选择当前目录：`apps/showdown-companion-extension`

## 如何启动完整链路

推荐按下面顺序启动：

1. 进入仓库根目录。
2. **推荐方式**：使用 GUI 启动器（`apps/launcher/launch.bat`），可视化配置所有参数后一键启动。
3. 或者用命令行启动：

在 PowerShell 中设置 `PYTHONPATH=src`，然后启动本地 pokestrategist suggestion server：

```powershell
$env:PYTHONPATH = "src"
python -m pokestrategist.cli.serve_showdown --checkpoint runs/pokestrategist_v1_metamon_recent10000_honestv4_fixall_v1/best_model.pt --device auto
```

如果你是通过包脚本启动，也可以用：

```powershell
$env:PYTHONPATH = "src"
pokestrategist-serve-showdown --checkpoint runs/pokestrategist_v1_metamon_recent10000_honestv4_fixall_v1/best_model.pt --device auto
```

默认服务地址：

- 健康检查：`http://127.0.0.1:8765/health`
- 建议接口：`http://127.0.0.1:8765/api/suggest`

扩展默认就会请求这个地址，所以本地服务启动后，通常只需要刷新 Showdown 页面即可。

## 当前本地接口约定

扩展会向如下地址发送 POST 请求：

- 默认：`http://127.0.0.1:8765/api/suggest`

请求体格式：

```json
{
  "snapshot": {
    "source": "showdown-dom",
    "pageTitle": "[Gen 9] OU battle",
    "turn": "Turn 12",
    "forcedSwitch": false,
    "self": { "name": "Great Tusk", "hp": "74%" },
    "opponent": { "name": "Dragapult", "hp": "63%" },
    "legalMoves": [
      { "label": "Headlong Rush", "disabled": false, "tooltip": "" }
    ],
    "legalSwitches": [
      { "label": "Gholdengo", "disabled": false, "tooltip": "" }
    ],
    "recentLog": [
      "Turn 12",
      "The opposing Dragapult used Draco Meteor!"
    ],
    "observedAt": "2026-05-22T08:00:00.000Z",
    "url": "https://play.pokemonshowdown.com/..."
  }
}
```

当前服务返回格式：

```json
{
  "source": "pokestrategist-local",
  "metadata": {
    "plan_label": "pressure",
    "phase_label": "tempo",
    "line_label": "tempo",
    "response_label": "hold",
    "frontier_size": 5,
    "legal_action_count": 7,
    "move_head_prob": 0.82,
    "switch_head_prob": 0.18,
    "turn": 12,
    "battle_format": "gen9ou"
  },
  "suggestions": [
    {
      "label": "Headlong Rush",
      "confidence": 0.71,
      "reason": "模型当前更偏向出招，高层判断为 plan=pressure、phase=tempo、line=tempo，该候选所属 family=attack。",
      "action": {
        "key": "move|Headlong Rush|attack|0||",
        "head": "move",
        "move_token": "Headlong Rush",
        "move_family": "attack",
        "tera": false,
        "switch_slot": null,
        "switch_species": null,
        "candidate_source": "revealed",
        "candidate_prior_prob": 1.0,
        "candidate_support_score": 1.0
      }
    },
    {
      "label": "Switch -> Gholdengo",
      "confidence": 0.18,
      "reason": "模型当前更偏向换人，高层判断为 plan=pressure、phase=tempo、line=tempo，该候选所属 family=switch。"
    }
  ]
}
```

说明：

1. `label` 和 `confidence` 直接用于扩展面板展示。
2. `reason` 是当前的简化解释文本，主要反映模型高层路由与候选 family，并不是完整可解释推理链。
3. `action` 字段保留了原始候选动作信息，便于后续把扩展面板做得更细。

## 当前服务实现状态

本地 suggestion server 已经实现，当前流程是：

1. 扩展抓取 Showdown 页面上的可见 battle state。
2. 服务端把 snapshot 适配成 `DecisionSample` 风格输入。
3. 服务端复用现有 checkpoint、特征化逻辑和 v1 决策模型做真实推理。
4. 服务端返回 top-k 建议给扩展面板。

这意味着现在已经不是纯前端占位框架，而是“前端抓取 + 本地模型服务 + 真实建议返回”的工作链条已经接通。

## 现实边界

这条产品路径现在已经可运行，但离稳定实战工具还有几个明显边界：

1. 更稳健的 Showdown DOM 适配层。现在使用的是一组通用选择器，足够做 MVP，但还需要针对不同 battle room 布局做加固。
2. snapshot 适配层仍然是 MVP。当前只把扩展抓到的可见状态映射到模型输入的近似形式，还没有完全恢复训练时最丰富的协议级信息。
3. 当前解释文本是简化版，不是完整的可解释推理轨迹。
4. 最重要的一点：代码闭环已打通，但还没有完成系统性的实战验证，所以当前建议质量仍需要继续联调和评估。

## 下一步建议

后续最自然的两步是：

1. 继续增强扩展端 DOM 抓取，补齐天气、场地、异常状态、更多队伍信息和更稳的 battle room 选择器。
2. 继续增强服务端 snapshot 适配，把当前近似输入逐步对齐到 pokestrategist 的真实 inference schema，再做实战验证与误差分析。