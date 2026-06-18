# Novel Voice Cast

> 上传已标注好说话人的小说 → 自动识别角色性别 → 标注情感 → 音色克隆/TTS → 输出完整有声书

## 项目背景

轻小说的对话通常不标注说话人，只用「」包裹。市面上的 TTS 工具要么只有单一朗读者，要么配置复杂。

整个小说转语音大项目分为三个阶段：

1. **[novel_correct](https://github.com/caiyilian/novel_correct)** — OCR 纠错，统一「」符号
2. **[opencode-novel-loop](https://github.com/caiyilian/opencode-novel-loop)** — 利用 AI 自动标注每句对话的说话人
3. **Novel Voice Cast（本项目）** — 最终阶段：音色克隆/TTS，输出有声书

## 流程

```
config.yaml → run_full.py → 解析 → 性别识别 → 情感标注 → TTS合成 → 拼接 → 输出MP3
```

## 快速开始

```bash
# 1. 编辑配置
vim config.yaml

# 2. 运行
python run_full.py
```

## 技术栈

| 模块 | 选型 |
|------|------|
| 配置 | YAML |
| TTS | VoxCPM（克隆）+ edge-tts（预设）+ pyttsx3（离线） |
| 情感标注 | Ollama (qwen3:4b) |
| 音频处理 | pydub + soundfile |
| 去噪 | DeepFilterNet3 |

## 项目状态

✅ 核心流程已完成。详见 [`方案.md`](方案.md)。
