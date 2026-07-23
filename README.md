# Novel Voice Cast

> 上传已标注好说话人的小说 → 自动识别角色性别 → 情绪与逐句表演导演 → 音色克隆/TTS → 输出完整有声书

## 项目背景

轻小说的对话通常不标注说话人，只用「」包裹。市面上的 TTS 工具要么只有单一朗读者，要么配置复杂。

整个小说转语音大项目分为三个阶段：

1. **[novel_correct](https://github.com/caiyilian/novel_correct)** — OCR 纠错，统一「」符号
2. **[opencode-novel-loop](https://github.com/caiyilian/opencode-novel-loop)** — 利用 AI 自动标注每句对话的说话人
3. **Novel Voice Cast（本项目）** — 最终阶段：音色克隆/TTS，输出有声书

## 流程

```
config.yaml → scripts/run_full.py → 解析 → 性别识别 → 情感标注 → 表演导演 → TTS合成 → 拼接/BGM/插图/视频
```

## 快速开始

先用 dry-run 检查哪些阶段会命中缓存：

```cmd
set "PYTHONUTF8=1" && .venv\Scripts\python.exe -u scripts\run_full.py --dry-run
```

完整前台运行并将控制台内容同步写入 UTF-8 日志：

```cmd
set "PYTHONUTF8=1" && .venv\Scripts\python.exe -u scripts\run_full.py --log logs\run_full.log 2>&1 | powershell -NoProfile -Command "$input | Tee-Object -FilePath 'logs\run_full_console.log'"
```

可用 `--from-stage` 和 `--to-stage` 恢复指定区间，例如只从 TTS 运行到 BGM 混音：

```cmd
set "PYTHONUTF8=1" && .venv\Scripts\python.exe -u scripts\run_full.py --from-stage tts --to-stage bgm-mix --log logs\run_full_tts_to_bgm.log 2>&1 | powershell -NoProfile -Command "$input | Tee-Object -FilePath 'logs\run_full_tts_to_bgm_console.log'"
```

所有长任务都在当前 CMD 前台执行；关闭窗口或按 `Ctrl+C` 才会终止。各阶段使用原子 checkpoint，重新执行同一命令会从兼容断点继续。

## 表演导演与 VoxCPM

情感标签只用于检索和辅助判断。真正送入 VoxCPM 的是逐句自然语言表演控制，例如 `（呼吸稍急但思路清楚，语速加快，短句推进，关键动作词重读）原文`。参考音频始终负责角色音色，控制词只描述本句的意图、潜台词、呼吸、节奏、停顿、音量和句内变化。

每个目标角色先由两个互相隔离的 Agent 建立稳定表演档案，再交给第三个 Agent 裁决；每句台词也执行同样的双盲分析和最终裁决。所有结论必须引用小说中的精确原文，三阶段输入哈希、逐代理 token、上下文占用和调用记录都会写入结果及 checkpoint。`Ctrl+C` 后可从已经完成的 Agent 或 WAV 继续。

VoxCPM 会复用同一参考音频的 prompt cache，生成文件先写临时 WAV，经坏样本、格式和内容哈希检查后才原子替换正式文件。当前关闭 `normalize`，用于规避 Windows 中文路径下 `kaldifst` 无法读取 FST 的问题。

## 生成带字幕的插图视频

完成 TTS 片段、最终混音、插图计划与插图生成后，直接运行：

```bash
python scripts/generate_video.py
```

脚本会按实际 WAV 时长及拼接间隔生成 SRT，并默认将中文字幕烧录进视频。每行最多 16 字、每条最多 2 行，长句优先在中文标点处拆分，显示格式为 `[说话人] 原文`。需要安装带 `libass` 字幕滤镜的 FFmpeg。

所有输入均可通过命令行覆盖，例如：

```bash
python scripts/generate_video.py \
  --novel novels/novel.txt \
  --labels novels/labels.txt \
  --segments-dir output/segments \
  --plan output/illustration_plan.json \
  --illustrations-dir output/illustrations \
  --audio output/full_volume_bgm.mp3 \
  --output output/illustration_video.mp4
```

如需保留无字幕版本，可增加 `--no-subtitles`。标签格式无法安全自动判断时，可显式指定 `--subtitle-label-mode line|parsed-line|dialogue`。

## 技术栈

| 模块 | 选型 |
|------|------|
| 配置 | YAML |
| TTS | VoxCPM（参考音频 + 自然语言表演控制）+ edge-tts（预设） |
| AI 分析 | SenseNova 6.7 Flash Lite（256K 上下文，多 Agent 严格证据审查） |
| 音频处理 | pydub + soundfile |
| 去噪 | DeepFilterNet3 |

## 项目状态

✅ 核心流程已完成。详见 [`docs/方案.md`](docs/方案.md)。
