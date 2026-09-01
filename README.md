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
config.yaml → scripts/run_full.py → 解析 → 性别识别 → 情感标注 → 表演导演 → TTS合成 → 拼接/BGM/视觉规划 → MiniMax H3 动态视频
```

## 快速开始

从 `novel.txt + labels.txt` 配置模型并生成横竖版带 BGM、字幕、H3 动态镜头视频的完整说明，见 [`docs/全流程使用教程.md`](docs/全流程使用教程.md)。

不想使用命令行时，可使用 Electron + SolidJS 桌面版选择/拖入两份文件，一键运行、停止、断点继续，并查看 13 阶段实时进度、日志、耗时和产物。开发启动、Windows NSIS 安装包和完整操作说明见 [`desktop/README.md`](desktop/README.md)。

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

## 生成带字幕的 H3 动态视频

完成 TTS 片段、最终混音、插图计划和提示词审核后，默认通过远程 MiniMax H3 的 `continuous-chain` 模式生成全时长动态镜头。程序按真实音频把 702 个粗镜头拆成 5～10 秒微镜头，并把对应原文、说话人、逐句表演方向、角色卡及剧情阶段视觉状态写入每条提示词；以有限 I2VA 短链和主动 T2VA 切镜限制漂移。动态覆盖不足会直接失败，不再保持末帧。H3 音轨不会替换现有 VoxCPM+BGM。

```cmd
.venv\Scripts\python.exe -u scripts\run_full.py --config config\config.yaml --from-stage video --to-stage video --log logs\h3_continuous_video.log
```

任务可随时 `Ctrl+C`，同一命令会从远程 `job_id`、已下载视频、质量检查、续帧及本地编码分段继续。每条续写片段还会核对生成首帧与输入锚点，断裂时换随机结果重试。旧 `native-chain` 素材会在不覆盖原文件的前提下接受检查并迁入新断点。架构、服务接口、配置和耗时说明见 [`docs/MiniMax-H3视频集成.md`](docs/MiniMax-H3视频集成.md)。静态插图版仍可通过 `video.h3.enabled: false` 保留使用。

全时长持续动画长片的分镜、关键帧、质量门禁、断点迁移、实测覆盖数据与无静止合成设计见 [`docs/MiniMax-H3全时长动态长片方案.md`](docs/MiniMax-H3全时长动态长片方案.md)。

## 技术栈

| 模块 | 选型 |
|------|------|
| 配置 | YAML |
| TTS | VoxCPM（参考音频 + 自然语言表演控制）+ edge-tts（预设） |
| AI 分析 | SenseNova 6.7 Flash Lite（256K 上下文，多 Agent 严格证据审查） |
| 动态视频 | MiniMax H3（音频锁定微镜头 + T2VA/I2VA/FL2VA + 无静帧补齐） |
| 音频处理 | pydub + soundfile |
| 去噪 | DeepFilterNet3 |

## 项目状态

✅ 核心流程已完成。详见 [`docs/方案.md`](docs/方案.md)。
