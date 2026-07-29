# Novel Voice Cast 桌面版

桌面版是现有 `scripts/run_full.py` 的图形控制台。它不会复制或改写 13 阶段逻辑，也不会启动新的 HTTP 后端；Electron 主进程直接运行项目虚拟环境中的 Python，读取同一套配置、checkpoint、日志和输出。

## 1. 使用前准备

桌面安装器只包含 Electron 界面，不包含数十 GB 的模型、Python 虚拟环境、FFmpeg、VoxCPM 或 ACE-Step。首次启动前，项目工作区仍需按 [`docs/全流程使用教程.md`](../docs/全流程使用教程.md) 完成：

- `.venv/Scripts/python.exe` 和 Python 依赖；
- `config/config.yaml`；
- SenseNova API Key、VoxCPM、ACE-Step、文生图服务；
- FFmpeg/FFprobe；
- 小说原文和逐条对话角色标注两份 UTF-8 `.txt` 文件。

从源码目录执行开发版时，应用会自动向上查找项目根。安装版建议先在 CMD 设置工作区，再重新打开应用：

```cmd
setx NOVEL_VOICE_CAST_ROOT "E:\projects\novel-voice-cast"
```

只想临时设置当前 CMD 启动的应用，可以使用：

```cmd
set "NOVEL_VOICE_CAST_ROOT=E:\projects\novel-voice-cast"
"C:\Users\你的用户名\AppData\Local\Programs\novel-voice-cast-desktop\Novel Voice Cast.exe"
```

项目根必须包含 `.venv/Scripts/python.exe`、`scripts/run_full.py` 和 `config/config.yaml`。应用从 YAML 的 `output.dir` 解析实际输出目录，不要求它固定叫 `output`。

## 2. 从源码启动

需要 Node.js 24 或更高版本。在项目根执行：

```cmd
cd /d E:\projects\novel-voice-cast\desktop
npm ci
npm run dev
```

`postinstall`/`dev` 会通过 Electron 镜像准备运行时，适合当前网络环境。开发窗口仍直接使用上一级项目中的 Python 和模型环境。

## 3. 用界面运行完整流程

1. 在“小说原文”卡片点击 `+` 或拖入 `novel.txt`。
2. 在“角色标注”卡片点击 `+` 或拖入 `labels.txt`。
3. 两个文件都通过 `.txt`、普通文件、可读性检查后，“开始完整流程”才会启用。
4. 保持“断点起点”为“自动检查全部缓存（推荐）”，点击开始。
5. 中央区域会显示实际 Python 命令、当前阶段、真实 checkpoint 百分比和当前操作；底部 13 张卡片显示每阶段状态。
6. 日志面板默认自动滚动，内存只保留最近 800 条；“清空显示”不会删除 `logs/desktop-*.log`。
7. 完成后查看总耗时、阶段耗时、manifest 核验结论和产物存在/缺失状态，点击“打开输出目录”。

应用启动的命令等价于：

```text
.venv/Scripts/python.exe -u scripts/run_full.py
  --config config/config.yaml
  --novel <选择的小说>
  --labels <选择的标注>
  --stream-tts
  --desktop-events
  --stop-file <本次运行专属停止文件>
  --log <本次运行专属日志>
```

只有 Python 退出码为 0、manifest 属于本次运行、`run_status` 为 `complete`，且全部所选阶段为 `complete/skipped` 时，界面才显示“流水线已完成”。

## 4. 停止、失败与断点继续

点击“停止并保留断点”后：

- 桌面端只写入本次子进程专属停止文件；
- `run_full.py` 在主线程收到与 Ctrl+C 等效的中断；
- 当前阶段记录 `interrupted`，已完成的 checkpoint 和 WAV 保留；
- streaming TTS 等本次运行的子进程会由 Python 自己收尾；
- 若 20 秒仍未退出，桌面端只终止本次持有的 PID 进程树，不扫描其他 Python 任务。

停止或失败后按钮会变成“继续运行”。通常保持“自动检查全部缓存（推荐）”并再次启动，让 Python 校验所有缓存；也可在“断点起点”中明确选择 `tts`、`illustrations` 等阶段，对应 `--from-stage`。桌面端不会删除、伪造或猜测 checkpoint。

## 5. 快速验收（不调用大模型）

`desktop/fixtures/novel.txt` 是 15 行 UTF-8 小说，`labels.txt` 含 4 个对应角色标签。测试只让真实 Python 跑 `parse -> parse`，不会调用 SenseNova、VoxCPM、ACE-Step 或文生图。

```cmd
cd /d E:\projects\novel-voice-cast\desktop
npm run verify
```

这条命令依次检查样例结构、TypeScript、全部 Vitest（包括真实 parse-only 控制器）和生产构建。

## 6. 生成 Windows 安装包

生成并检查 NSIS 安装器：

```cmd
cd /d E:\projects\novel-voice-cast\desktop
npm run package:verify
```

主要产物：

- `desktop/release/win-unpacked/Novel Voice Cast.exe`；
- `desktop/release/novel-voice-cast-desktop-<版本>-x64.exe`（NSIS 安装器）。

对 unpacked 应用执行 4 秒启动冒烟：

```cmd
npm run smoke:win
```

`release/` 是本地构建产物，不提交 Git。安装后仍须通过 `NOVEL_VOICE_CAST_ROOT` 指向准备好的项目工作区。

## 7. 常见问题

- “找不到项目根目录”：设置 `NOVEL_VOICE_CAST_ROOT`，确认路径中存在脚本、配置和 `.venv`，然后重启应用。
- “项目运行时不完整”：按错误中列出的绝对路径补齐 Python、脚本或配置。
- 两份文件齐了仍不能开始：文件必须以 `.txt` 结尾、是普通文件、可读且非空。
- Python 退出码为 0 但界面显示失败：查看 manifest 诊断；旧 manifest、阶段范围不符或所选阶段未完整记录都不会被误判为成功。
- “输出目录不存在”：先让流水线至少创建配置中的 `output.dir`；按钮不会接受任意路径。
- 日志过多：界面只保留尾部，完整内容在 `logs/desktop-*.log`。
- 模型或网络错误：修复配置/服务后点击“继续运行”，兼容 checkpoint 会自动复用。

桌面端的安全边界是 `contextIsolation: true`、`nodeIntegration: false`、`sandbox: true`。渲染层只能调用 preload 白名单 API，不能访问 Node、文件系统、shell 或任意 IPC。
