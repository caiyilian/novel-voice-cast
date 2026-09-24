# Novel Voice Cast

> 把已标注说话人的轻小说自动转成**带 BGM 的完整有声书**，并可选生成插图与动态视频。

输入 `novel.txt` + `labels.txt`，输出 6 小时级有声书。全程断点续跑，任一阶段可单独重跑。

---

## 这是三个阶段的最后一个

| 阶段 | 项目 | 职责 |
|---|---|---|
| 1 | [novel_correct](https://github.com/caiyilian/novel_correct) | OCR 纠错，统一「」符号 |
| 2 | [opencode-novel-loop](https://github.com/caiyilian/opencode-novel-loop) | AI 自动标注每句对话的说话人 |
| 3 | **Novel Voice Cast（本项目）** | 音色克隆 + TTS + BGM + 混音，输出有声书 |

轻小说对话只用「」包裹、不标说话人，这是前两个阶段要解决的问题。本项目假设说话人已标注完毕。

---

## 12 个阶段

```
parse → gender → performance → tts → splice
                                    ↓
        bgm-segment → bgm-label → bgm-generate → bgm-mix
                                    ↓
                    illustration-plan → illustrations → video
```

| # | 阶段 | 做什么 | 关键产物 |
|---|---|---|---|
| 1 | `parse` | 解析小说与角色标注 | 分段对话列表 |
| 2 | `gender` | 识别每个角色的性别 | `gender_results.json` |
| 3 | `performance` | **双盲多 Agent** 生成逐句表演指导 | `performance_directions.json` |
| 4 | `tts` | CosyVoice 3 零样本音色克隆合成 | `segments/*.wav` |
| 5 | `splice` | 拼接为整段人声 | `full_volume.mp3` |
| 6 | `bgm-segment` | 按剧情切分音乐场景 | `bgm_segments.json` |
| 7 | `bgm-label` | 标注情绪 + 细分织体 | 同上（补充字段） |
| 8 | `bgm-generate` | Stable Audio 3 生成 BGM | `bgm/*.mp3` |
| 9 | `bgm-mix` | 人声 + BGM 混音 | **`full_volume_bgm.mp3`** |
| 10 | `illustration-plan` | 规划插图位置与提示词 | `illustration_plan.json` |
| 11 | `illustrations` | 生成插图 | `illustrations_*/` |
| 12 | `video` | MiniMax H3 动态镜头长片 | `*.mp4` |

**主线只需 1-9**（产出有声书）。10-12 是可选增强。

> 注：历史上曾有 `emotion` 阶段（13 阶段），已在 `b2caded` 中移除——
> 情感标签的职责并入 `bgm-label`，避免重复标注。

---

## 快速开始

```cmd
:: 先看哪些阶段会命中缓存
set "PYTHONUTF8=1" && .venv\Scripts\python.exe -u scripts\run_full.py --dry-run

:: 完整运行（前台，Ctrl+C 可中断，重跑同命令即续跑）
set "PYTHONUTF8=1" && .venv\Scripts\python.exe -u scripts\run_full.py --log logs\run_full.log

:: 只跑某个区间（例：从 TTS 到混音）
set "PYTHONUTF8=1" && .venv\Scripts\python.exe -u scripts\run_full.py --from-stage tts --to-stage bgm-mix --log logs\run_tts_to_bgm.log
```

所有阶段使用**原子 checkpoint**，中断后重跑同命令会从兼容断点继续。

**图形界面**：Electron + SolidJS 桌面版支持拖入两份文件、一键运行/停止/续跑，
并显示 12 阶段实时进度、日志、耗时与产物。见 [`desktop/README.md`](desktop/README.md)。

**详细教程**：[`docs/全流程使用教程.md`](docs/全流程使用教程.md)

---

## 核心设计

### 表演指导：双盲多 Agent

每句台词的表演方向由**两个互相隔离的 Agent 独立分析**，再由第三个 Agent 裁决，
所有结论必须引用小说中的精确原文。

产出 `performance_control` 字段（中文自然语言，如「呼吸稍急但思路清楚，语速加快，
短句推进，关键动作词重读」），作为 `instruct` 传给 TTS 引擎。
**参考音频负责音色，控制词只描述本句的意图、呼吸、节奏、停顿、音量与句内变化。**

三阶段输入哈希、逐代理 token、上下文占用与调用记录都写入结果及 checkpoint。

### BGM：场景级情绪 + 织体

`bgm-segment` 按剧情切分场景，`bgm-label` 为每个场景标注**情绪类型**（suspense / daily /
comedy / sad / romantic / battle / epic / horror）与**细分织体**（29 种，如 `indoor_talk` /
`scheming` / `creeping_dread` / `market_bustle`）。

`bgm-generate` 用 Stable Audio 3 为每个场景生成多条 90 秒纯器乐（跨段轮转乐器编制，
避免相邻段落音色雷同），`bgm-mix` 按时间轴混入人声。

实测规模：**210 场景 × 4 条 = 840 条 BGM**，覆盖 6.2 小时不重复。

### 断点与原子性

- 各阶段独立 checkpoint，可单独重跑
- 音频先写临时文件，通过坏样本/格式/内容哈希检查后才原子替换
- `Ctrl+C` 后从已完成的 Agent 或 WAV 继续

---

## TTS 质量控制闭环

CosyVoice 的 `instruct` 是**概率性**的——同一输入多次生成结果可能不同，偶尔会把控制词念出来。
为此建立检测闭环：

```
ASR 转录 → LLM 判定 → 有问题则改写提示词/重生成 → 复检
```

| 脚本 | 作用 |
|---|---|
| `scripts/tts_quality_loop.py` | 三段式：`transcribe`（ASR）/ `audit`（LLM 判定）/ `repair`（改写+重生成） |
| `scripts/tts_fix_loop.py` | 串联三段的循环驱动器（最多 N 轮） |
| `scripts/tts_leak_detect.py` | 字符级比对（参考用，非判据） |

判定三分类：`OK` / `LEAK`（控制词泄漏）/ `MISMATCH`（内容缺失或大幅不符）。

> **完整的排查过程、5 组实验数据、源码级机制分析与待决事项**：
> [`docs/TTS质量控制排查报告.md`](docs/TTS质量控制排查报告.md)

### 已修复的关键 bug

`backend/cosyvoice_worker.py` 曾只取 `chunks[0]`，而 CosyVoice 前端会把长文本
切成多块逐块 yield → **所有超长句子的后半段被静默丢弃**。修复为 `torch.cat` 拼接全部块后，
长句内容覆盖率从 0.26-0.35 恢复到 1.00。

---

## 技术栈

| 模块 | 选型 |
|------|------|
| 配置 | YAML |
| TTS | **CosyVoice 3**（Fun-CosyVoice3-0.5B，零样本克隆）+ VoxCPM（备选）+ edge-tts / pyttsx3（预设） |
| BGM | **Stable Audio 3**（small-music / medium，纯器乐 44.1kHz 立体声） |
| AI 分析 | SenseNova Pool 代理 → `deepseek-v4-flash`（多 Agent 严格证据审查） |
| 动态视频 | MiniMax H3（音频锁定微镜头 + T2VA/I2VA/FL2VA） |
| 质量检测 | faster-whisper large-v2（ASR）+ OpenCC（繁简归一） |
| 音频处理 | pydub + soundfile + ffmpeg |
| 去噪 | DeepFilterNet3 |
| 桌面端 | Electron + SolidJS |

---

## 环境要求

- **Python**：`.venv`（项目根）
- **CosyVoice 3**：独立仓库 + 独立 venv，路径在 `config/config.yaml` 的 `cosyvoice` 段
- **GPU**：TTS 与 BGM 生成都需要 CUDA。BGM 的 medium 模型峰值显存约 6.4 GB
- **ffmpeg**：混音与格式转换依赖

`config/config.yaml` 中 `cosyvoice.repo_path` / `model_path` / `python` 三项需按本机实际路径配置。

---

## 项目状态

✅ **核心流程（1-9 阶段）已完成**，已产出 6.2 小时带 BGM 有声书。

| 产物 | 说明 |
|---|---|
| `output/full_volume.mp3` | 纯人声，427.6 MB |
| `output/full_volume_bgm.mp3` | **带 BGM 完整版，513.1 MB** |

⚠️ **进行中**：TTS 质量控制闭环的第二轮修复（详见排查报告）。
`illustration-plan` / `illustrations` / `video` 阶段的产物当前未保留。

详细设计见 [`docs/方案.md`](docs/方案.md)。

---

## 文档索引

| 文档 | 内容 |
|---|---|
| [`docs/全流程使用教程.md`](docs/全流程使用教程.md) | 从零到成品的完整操作说明 |
| [`docs/TTS质量控制排查报告.md`](docs/TTS质量控制排查报告.md) | 控制词泄漏的机制分析与修复 |
| [`docs/MiniMax-H3视频集成.md`](docs/MiniMax-H3视频集成.md) | 动态视频架构、接口与配置 |
| [`docs/MiniMax-H3全时长动态长片方案.md`](docs/MiniMax-H3全时长动态长片方案.md) | 分镜、关键帧、质量门禁设计 |
| [`docs/方案.md`](docs/方案.md) | 整体技术方案 |
| [`docs/踩坑记录.md`](docs/踩坑记录.md) | 开发过程中的问题与解决 |
| [`desktop/README.md`](desktop/README.md) | 桌面版开发与打包 |
