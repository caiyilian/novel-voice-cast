# Novel Voice Cast

> 上传已标注好说话人的小说 → 给每个角色分配音频 → 一键合成带角色演绎的有声书。

## 项目背景

轻小说的对话通常不标注说话人，只用「」包裹。市面上的 TTS 工具要么只有单一朗读者，要么配置复杂。

整个小说转语音大项目分为三个阶段：

1. **[novel_correct](https://github.com/caiyilian/novel_correct)** — OCR 纠错，统一「」符号
2. **[opencode-novel-loop](https://github.com/caiyilian/opencode-novel-loop)** — 利用 AI 自动标注每句对话的说话人
3. **Novel Voice Cast（本项目）** — 最终阶段：Web UI + 音色克隆/TTS，输出有声书

## 流程

```
labeled_novel.txt ──→ 解析提取对话+角色 ──→ Web UI 分配音频 ──→ 音色克隆+TTS ──→ 输出完整有声书
```

## 技术栈

| 层 | 选型 |
|---|---|
| 后端 | FastAPI + SQLite + SQLAlchemy |
| 前端 | Vue 3 + Vite + Naive UI |
| TTS | edge-tts（预设）+ 音色克隆（待接入） |
| 音频处理 | pydub + soundfile |
| 异步任务 | ARQ (Redis) |

## 项目状态

🚧 开发中。详见 [`方案.md`](方案.md)。
