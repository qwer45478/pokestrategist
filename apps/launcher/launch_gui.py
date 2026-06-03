"""
PokeStrategist Showdown 本地服务启动器 (GUI)
========================================
提供一个简单的图形界面来配置和启动 pokestrategist 本地推理服务。

用法:
    cd apps/launcher
    python launch_gui.py

或直接双击 launch_gui.py（如果 .py 关联到 PokePilot conda 环境的 python）。
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from tkinter import (
    BOTH,
    DISABLED,
    END,
    HORIZONTAL,
    LEFT,
    NORMAL,
    RIGHT,
    VERTICAL,
    BooleanVar,
    Entry,
    Frame,
    IntVar,
    Label,
    LabelFrame,
    Menu,
    OptionMenu,
    Scale,
    StringVar,
    Text,
    Tk,
    messagebox,
    ttk,
)
from typing import Any

# ── 项目根目录 ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = PROJECT_ROOT / "runs"
SRC_DIR = PROJECT_ROOT / "src"

# ── 默认配置 ────────────────────────────────────────────────
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_DEVICE = "auto"
DEFAULT_TOPK = 3
DEFAULT_HIDDEN_TOPK = 4  # None in CLI, but typically 4


def discover_checkpoints() -> list[tuple[str, str]]:
    """扫描 runs/ 目录，返回 [(显示名, 路径), ...] 列表。"""
    checkpoints: list[tuple[str, str]] = []
    if not RUNS_DIR.exists():
        return checkpoints

    for run_dir in sorted(RUNS_DIR.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue
        pt_file = run_dir / "best_model.pt"
        ranking_pt = run_dir / "best_ranking_model.pt"
        if pt_file.exists():
            checkpoints.append((f"{run_dir.name}  (composite)", str(pt_file)))
        if ranking_pt.exists() and ranking_pt != pt_file:
            checkpoints.append((f"{run_dir.name}  (ranking)", str(ranking_pt)))

    return checkpoints


def find_conda_python() -> str | None:
    """尝试找到 PokePilot conda 环境中的 python 路径。"""
    candidates = [
        # 常见的 conda env 路径
        Path.home() / ".conda" / "envs" / "PokePilot" / "python.exe",
        Path.home() / "anaconda3" / "envs" / "PokePilot" / "python.exe",
        Path(os.environ.get("CONDA_PREFIX", "")) / "python.exe" if os.environ.get("CONDA_PREFIX") else None,
    ]
    for candidate in candidates:
        if candidate and candidate.exists():
            return str(candidate)
    return sys.executable  # 回退到当前解释器


def load_config() -> dict[str, Any]:
    """从 config.json 加载上次保存的配置。"""
    config_path = Path(__file__).with_suffix(".config.json")
    if config_path.exists():
        try:
            return json.loads(config_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def save_config(config: dict[str, Any]) -> None:
    """保存当前配置到 config.json。"""
    config_path = Path(__file__).with_suffix(".config.json")
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")


class PokeStrategistLauncher:
    """启动器主窗口。"""

    def __init__(self, root: Tk) -> None:
        self.root = root
        self.root.title("PokeStrategist Showdown 本地服务启动器")
        self.root.geometry("780x680")
        self.root.minsize(680, 580)
        self.root.resizable(True, True)

        # 服务进程
        self._process: subprocess.Popen[bytes] | None = None
        self._monitor_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        # 可用 checkpoint 列表
        self._checkpoints = discover_checkpoints()

        # 上次配置
        self._saved = load_config()

        # ── 变量 ──
        self._checkpoint_var = StringVar(value=self._saved.get("checkpoint", ""))
        self._host_var = StringVar(value=self._saved.get("host", DEFAULT_HOST))
        self._port_var = IntVar(value=self._saved.get("port", DEFAULT_PORT))
        self._device_var = StringVar(value=self._saved.get("device", DEFAULT_DEVICE))
        self._topk_var = IntVar(value=self._saved.get("topk", DEFAULT_TOPK))
        saved_hidden = self._saved.get("hidden_candidate_topk")
        self._hidden_topk_var = StringVar(value=str(saved_hidden) if saved_hidden is not None else "")
        self._team_preview_var = BooleanVar(value=self._saved.get("use_team_preview_prior", True))
        self._usage_priors_var = BooleanVar(value=self._saved.get("use_usage_priors", True))
        self._usage_data_dir_var = StringVar(value=self._saved.get("usage_data_dir", ""))
        self._python_var = StringVar(value=self._saved.get("python_path", find_conda_python() or sys.executable))

        self._running = False

        self._build_ui()
        self._update_start_button()

    # ── UI 构建 ─────────────────────────────────────────────

    def _build_ui(self) -> None:
        # 顶部菜单
        menubar = Menu(self.root)
        self.root.config(menu=menubar)
        help_menu = Menu(menubar, tearoff=0)
        help_menu.add_command(label="关于 PokeStrategist 启动器", command=self._show_about)
        menubar.add_cascade(label="帮助", menu=help_menu)

        # 主容器
        main_frame = Frame(self.root, padx=10, pady=10)
        main_frame.pack(fill=BOTH, expand=True)

        # ── 上半部分：配置面板 ──
        config_frame = LabelFrame(main_frame, text="服务配置", padx=8, pady=8)
        config_frame.pack(fill=BOTH, expand=False)

        row = 0

        # Python 解释器
        Label(config_frame, text="Python 解释器:").grid(row=row, column=0, sticky="w", pady=3)
        python_frame = Frame(config_frame)
        python_frame.grid(row=row, column=1, sticky="ew", pady=3)
        Entry(python_frame, textvariable=self._python_var, width=50).pack(side=LEFT, fill="x", expand=True)
        ttk.Button(python_frame, text="自动检测", command=self._auto_detect_python).pack(side=RIGHT, padx=(4, 0))
        row += 1

        # Checkpoint
        Label(config_frame, text="模型 Checkpoint:").grid(row=row, column=0, sticky="w", pady=3)
        ckpt_frame = Frame(config_frame)
        ckpt_frame.grid(row=row, column=1, sticky="ew", pady=3)
        if self._checkpoints:
            display_names = [name for name, _ in self._checkpoints]
            self._checkpoint_combo = ttk.Combobox(
                ckpt_frame, textvariable=self._checkpoint_var, values=display_names, width=47, state="readonly"
            )
            self._checkpoint_combo.pack(side=LEFT, fill="x", expand=True)
            self._checkpoint_combo.bind("<<ComboboxSelected>>", self._on_checkpoint_selected)
            # 尝试恢复上次选择
            saved_path = self._saved.get("checkpoint", "")
            for i, (_, path) in enumerate(self._checkpoints):
                if path == saved_path:
                    self._checkpoint_combo.current(i)
                    break
        else:
            Entry(ckpt_frame, textvariable=self._checkpoint_var, width=50).pack(side=LEFT, fill="x", expand=True)
        ttk.Button(ckpt_frame, text="浏览...", command=self._browse_checkpoint).pack(side=RIGHT, padx=(4, 0))
        row += 1

        # 网络
        Label(config_frame, text="监听地址:").grid(row=row, column=0, sticky="w", pady=3)
        net_frame = Frame(config_frame)
        net_frame.grid(row=row, column=1, sticky="ew", pady=3)
        Entry(net_frame, textvariable=self._host_var, width=20).pack(side=LEFT)
        Label(net_frame, text="  端口:").pack(side=LEFT)
        Entry(net_frame, textvariable=self._port_var, width=8).pack(side=LEFT, padx=(4, 0))
        row += 1

        # 设备
        Label(config_frame, text="推理设备:").grid(row=row, column=0, sticky="w", pady=3)
        device_frame = Frame(config_frame)
        device_frame.grid(row=row, column=1, sticky="ew", pady=3)
        OptionMenu(device_frame, self._device_var, "auto", "cuda", "cpu").pack(side=LEFT)
        Label(device_frame, text="  (auto = 有 GPU 则用 CUDA，否则用 CPU)", fg="gray").pack(side=LEFT, padx=(8, 0))
        row += 1

        # TopK
        Label(config_frame, text="建议数量 TopK:").grid(row=row, column=0, sticky="w", pady=3)
        topk_frame = Frame(config_frame)
        topk_frame.grid(row=row, column=1, sticky="ew", pady=3)
        Scale(topk_frame, from_=1, to=10, orient=HORIZONTAL, variable=self._topk_var, length=200).pack(side=LEFT)
        Label(topk_frame, textvariable=self._topk_var, width=3).pack(side=LEFT, padx=(8, 0))
        row += 1

        # ── 高级选项 ──
        advanced_frame = LabelFrame(config_frame, text="高级选项", padx=6, pady=6)
        advanced_frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(8, 2))
        arow = 0

        ttk.Checkbutton(
            advanced_frame, text="启用 Team Preview 先验", variable=self._team_preview_var
        ).grid(row=arow, column=0, sticky="w", pady=2)
        arow += 1

        ttk.Checkbutton(
            advanced_frame, text="启用使用率先验 (Usage Priors)", variable=self._usage_priors_var
        ).grid(row=arow, column=0, sticky="w", pady=2)
        arow += 1

        Label(advanced_frame, text="隐藏候选 TopK:").grid(row=arow, column=0, sticky="w", pady=2)
        Entry(advanced_frame, textvariable=self._hidden_topk_var, width=8).grid(row=arow, column=1, sticky="w", pady=2)
        Label(advanced_frame, text="(留空 = 使用 checkpoint 默认值)", fg="gray").grid(row=arow, column=2, sticky="w", pady=2)
        arow += 1

        Label(advanced_frame, text="使用率数据目录:").grid(row=arow, column=0, sticky="w", pady=2)
        Entry(advanced_frame, textvariable=self._usage_data_dir_var, width=40).grid(row=arow, column=1, columnspan=2, sticky="ew", pady=2)

        config_frame.columnconfigure(1, weight=1)

        # ── 控制按钮 ──
        btn_frame = Frame(main_frame, pady=8)
        btn_frame.pack(fill="x")
        self._start_btn = ttk.Button(btn_frame, text="▶  启动服务", command=self._start_server)
        self._start_btn.pack(side=LEFT, padx=(0, 8))
        self._stop_btn = ttk.Button(btn_frame, text="■  停止服务", command=self._stop_server, state=DISABLED)
        self._stop_btn.pack(side=LEFT, padx=(0, 8))
        self._health_btn = ttk.Button(btn_frame, text="🔍 健康检查", command=self._health_check, state=DISABLED)
        self._health_btn.pack(side=LEFT, padx=(0, 8))
        Label(btn_frame, text="", fg="gray").pack(side=LEFT, fill="x", expand=True)

        # 状态指示器
        self._status_var = StringVar(value="⚫ 未启动")
        self._status_label = Label(btn_frame, textvariable=self._status_var, fg="gray", font=("", 9, "bold"))
        self._status_label.pack(side=RIGHT)

        # ── 日志输出 ──
        log_frame = LabelFrame(main_frame, text="服务日志", padx=6, pady=6)
        log_frame.pack(fill=BOTH, expand=True, pady=(4, 0))
        self._log_text = Text(log_frame, wrap="word", state=DISABLED, font=("Consolas", 9), bg="#1e1e1e", fg="#d4d4d4")
        self._log_text.pack(fill=BOTH, expand=True)

        # 日志滚动条
        scrollbar = ttk.Scrollbar(self._log_text, orient=VERTICAL, command=self._log_text.yview)
        scrollbar.pack(side=RIGHT, fill="y")
        self._log_text.config(yscrollcommand=scrollbar.set)

    # ── 事件处理 ────────────────────────────────────────────

    def _on_checkpoint_selected(self, event: object) -> None:
        idx = self._checkpoint_combo.current()
        if 0 <= idx < len(self._checkpoints):
            self._checkpoint_var.set(self._checkpoints[idx][1])

    def _browse_checkpoint(self) -> None:
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            title="选择模型 Checkpoint",
            initialdir=str(RUNS_DIR) if RUNS_DIR.exists() else str(PROJECT_ROOT),
            filetypes=[("PyTorch Checkpoint", "*.pt"), ("All Files", "*.*")],
        )
        if path:
            self._checkpoint_var.set(path)

    def _auto_detect_python(self) -> None:
        python = find_conda_python()
        if python:
            self._python_var.set(python)
            self._log(f"[INFO] 已检测到 Python: {python}")
        else:
            self._log("[WARN] 未找到 PokePilot conda 环境，使用当前解释器。")

    def _update_start_button(self) -> None:
        if self._running:
            self._start_btn.config(state=DISABLED)
            self._stop_btn.config(state=NORMAL)
            self._health_btn.config(state=NORMAL)
        else:
            self._start_btn.config(state=NORMAL)
            self._stop_btn.config(state=DISABLED)
            self._health_btn.config(state=DISABLED)

    def _set_status(self, text: str, color: str = "gray") -> None:
        self._status_var.set(text)
        self._status_label.config(fg=color)

    def _log(self, message: str) -> None:
        """向日志窗口追加一行。"""
        self._log_text.config(state=NORMAL)
        timestamp = time.strftime("%H:%M:%S")
        self._log_text.insert(END, f"[{timestamp}] {message}\n")
        self._log_text.see(END)
        self._log_text.config(state=DISABLED)

    def _clear_log(self) -> None:
        self._log_text.config(state=NORMAL)
        self._log_text.delete("1.0", END)
        self._log_text.config(state=DISABLED)

    # ── 配置保存 ────────────────────────────────────────────

    @staticmethod
    def _safe_int(text: str) -> int | None:
        """安全地将字符串转为 int，无法转换时返回 None。"""
        stripped = text.strip()
        if not stripped or stripped.lower() == "none":
            return None
        try:
            return int(stripped)
        except ValueError:
            return None

    def _gather_config(self) -> dict[str, Any]:
        return {
            "checkpoint": self._checkpoint_var.get(),
            "host": self._host_var.get(),
            "port": self._port_var.get(),
            "device": self._device_var.get(),
            "topk": self._topk_var.get(),
            "hidden_candidate_topk": self._safe_int(self._hidden_topk_var.get()),
            "use_team_preview_prior": self._team_preview_var.get(),
            "use_usage_priors": self._usage_priors_var.get(),
            "usage_data_dir": self._usage_data_dir_var.get().strip() or None,
            "python_path": self._python_var.get(),
        }

    # ── 服务控制 ────────────────────────────────────────────

    def _start_server(self) -> None:
        checkpoint = self._checkpoint_var.get().strip()
        if not checkpoint:
            messagebox.showerror("错误", "请先选择一个模型 Checkpoint。")
            return
        if not Path(checkpoint).exists():
            messagebox.showerror("错误", f"Checkpoint 文件不存在:\n{checkpoint}")
            return

        python = self._python_var.get().strip()
        if not python or not Path(python).exists():
            messagebox.showerror("错误", f"Python 解释器无效:\n{python}")
            return

        # 保存配置
        config = self._gather_config()
        save_config(config)

        # 构建命令
        cmd = [
            python,
            "-m", "pokestrategist.cli.serve_showdown",
            "--checkpoint", checkpoint,
            "--host", self._host_var.get(),
            "--port", str(self._port_var.get()),
            "--device", self._device_var.get(),
            "--topk", str(self._topk_var.get()),
        ]
        hidden = self._hidden_topk_var.get().strip()
        if hidden:
            cmd += ["--hidden-candidate-topk", hidden]
        if not self._team_preview_var.get():
            cmd += ["--disable-team-preview-prior"]
        if not self._usage_priors_var.get():
            cmd += ["--disable-usage-priors"]
        usage_dir = self._usage_data_dir_var.get().strip()
        if usage_dir:
            cmd += ["--usage-data-dir", usage_dir]

        env = os.environ.copy()
        env["PYTHONPATH"] = str(SRC_DIR)

        self._clear_log()
        self._log(f"[CMD] {' '.join(cmd)}")
        self._log(f"[ENV] PYTHONPATH={SRC_DIR}")
        self._log("─" * 60)

        try:
            self._process = subprocess.Popen(
                cmd,
                cwd=str(PROJECT_ROOT),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=False,
                bufsize=1,
            )
        except Exception as e:
            self._log(f"[ERROR] 启动进程失败: {e}")
            messagebox.showerror("启动失败", str(e))
            return

        self._running = True
        self._stop_event.clear()
        self._update_start_button()
        self._set_status("🟢 运行中", "green")

        # 启动输出监控线程
        self._monitor_thread = threading.Thread(target=self._monitor_output, daemon=True)
        self._monitor_thread.start()

    def _stop_server(self) -> None:
        if self._process and self._process.poll() is None:
            self._log("[INFO] 正在停止服务...")
            self._stop_event.set()
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
            self._log("[INFO] 服务已停止。")
        self._running = False
        self._process = None
        self._update_start_button()
        self._set_status("⚫ 已停止", "gray")

    def _monitor_output(self) -> None:
        """在后台线程中读取子进程输出。"""
        assert self._process is not None
        assert self._process.stdout is not None

        for line_bytes in self._process.stdout:
            if self._stop_event.is_set():
                break
            line = line_bytes.decode("utf-8", errors="replace").rstrip("\n\r")
            self.root.after(0, self._log, line)

        # 进程退出后的处理
        exit_code = self._process.wait()
        if not self._stop_event.is_set():
            self.root.after(0, self._on_process_exit, exit_code)

    def _on_process_exit(self, exit_code: int) -> None:
        if exit_code != 0:
            self._log(f"[ERROR] 服务进程异常退出，退出码: {exit_code}")
            self._set_status("🔴 异常退出", "red")
        else:
            self._log("[INFO] 服务进程正常退出。")
            self._set_status("⚫ 已停止", "gray")
        self._running = False
        self._process = None
        self._update_start_button()

    def _health_check(self) -> None:
        """发送健康检查请求。"""
        import urllib.request
        import urllib.error
        url = f"http://{self._host_var.get()}:{self._port_var.get()}/health"
        self._log(f"[HEALTH] GET {url}")
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                data = json.loads(resp.read().decode())
                self._log(f"[HEALTH] ✅ 服务正常: {json.dumps(data, indent=2, ensure_ascii=False)}")
                messagebox.showinfo("健康检查", "✅ 服务运行正常！")
        except urllib.error.URLError as e:
            self._log(f"[HEALTH] ❌ 连接失败: {e}")
            messagebox.showerror("健康检查", f"无法连接到服务:\n{e}")
        except Exception as e:
            self._log(f"[HEALTH] ❌ 检查出错: {e}")
            messagebox.showerror("健康检查", str(e))

    def _show_about(self) -> None:
        messagebox.showinfo(
            "关于 PokeStrategist 启动器",
            "PokeStrategist Showdown 本地服务启动器 v0.1\n\n"
            "用于配置和启动 PokeStrategist 本地推理服务，\n"
            "配合 Chrome 扩展在 Pokemon Showdown 中\n"
            "提供实时 AI 决策建议。\n\n"
            "项目根目录: " + str(PROJECT_ROOT),
        )

    def on_close(self) -> None:
        if self._running:
            if messagebox.askokcancel("退出", "服务正在运行中，确定要退出吗？"):
                self._stop_server()
        self.root.destroy()


def main() -> None:
    root = Tk()
    app = PokeStrategistLauncher(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)

    # 设置窗口图标（可选）
    try:
        root.iconbitmap(default="")
    except Exception:
        pass

    root.mainloop()


if __name__ == "__main__":
    main()
