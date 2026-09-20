# Novel Voice Cast 重构调研文档（语音链路 · 至 BGM 生成）

> 范围：从 `parse` 到 `bgm-mix` 的完整语音有声书链路（Stage 1–10）。
> 生图（illustration-plan / illustrations）与视频（video）本次仅标注，不深入。
> 目标：① LLM 统一换成 SenseNova Pool 代理的 DeepSeek V4 Flash；② 每个阶段追求高质量；③ 减少重复造轮子。

---

## 0. 摘要（TL;DR）

1. **LLM 阶段（gender / emotion / performance / bgm-segment / bgm-label）**：现在全部被 `for_flash_lite` 钉死在 `sensenova-6.7-flash-lite`，且直连 `token.sensenova.cn` 用 7 个 key 自管理轮询。**应统一换成你 opencode 里那个本地代理 `http://127.0.0.1:18787/v1`（20 账号轮询）+ `deepseek-v4-flash`**，删掉自研的多 key 轮询/冷却/Agnes fallback 这一整套，代码会大幅简化，速度与质量双升。
2. **TTS（VoxCPM）与 BGM（ACE-Step）引擎本身已经是开源里最合适的选择**，不需要换模型。真正的痛点是**集成方式**（跨进程动态脚本 + 硬编码路径）和**代码架构**，而不是引擎选型。
3. **最大的重复造轮子**：`gender / emotion / performance / bgm` 四个文件各自完整复制了一遍「多 Agent 双盲 + 裁决 + 证据引用 + checkpoint + usage 统计」框架（约 4 × 2000+ 行），应抽成一个通用框架。
4. **`scripts/run_full.py` 4690 行**是巨型编排文件，把编排、TTS 脚本生成、缓存校验、checkpoint 全塞一处，应拆分。
5. 没有专门的开源模型能替代「性别 / 情感 / 表演导演」这种**需要整本小说上下文理解**的任务，这几个阶段应继续用 LLM，靠换更强模型 + 精简 prompt 结构提升质量。

---

## 0.1 设计约束（本次重构的硬性原则）

> 用户明确：token 用不完，不怕消耗。但**只有当增加的 token 带来实质质量提升时才值得**；若纯堆 token 只换来微小提升、却显著拖慢总时长，就不做。

三条铁律：

1. **保留工具调用（tool-calling），让模型主动读证据**：不做「Python 硬塞上下文」替代——大模型「想读哪里就读哪里」才是最好的。每个阶段保留 `search_novel` / `read_lines` / `get_dialogues` 等工具，让模型按需检索原文证据。这是本项目质量的核心，不能砍。

2. **任务克制（单对象聚焦）**：每个 LLM 调用**只处理一个对象**——一个人、一句台词、一个 BGM 场景。**绝不让它「一口气标注一堆人的性别 / 一堆句子的情绪」。** 宁可多次调用、每次专注一件事，保证每个对象的判断质量最高。这样单次输出也短（快），不会触发 64K 输出上限、也不会拖慢。

3. **输入大方、输出精简**：deepseek-v4-flash 的 1M context 可以充分利用（输入并行编码，几乎不增加延迟），但输出是逐 token 串行生成、越短越快，所以结构化输出字段要精简（evidence 压缩到必要长度），单次输出自然就小。

一句话：**单对象聚焦 + 保留工具调用 + 输入大方、输出精简。**

> 推论：**不做批量标注**（不一次让模型标注多句/多场景）。质量和「时间可控」都来自「单对象聚焦 + 精简输出 + 并发调度」，而不是「一次塞一堆任务」。

---

## 1. 现状盘点

### 1.1 项目是什么

小说转有声书的收官阶段：输入「已标注说话人的小说 + 配置」，输出「带 BGM 的完整音频（进一步到带字幕动态视频）」。上游两个仓库分别做 OCR 纠错和说话人标注。

### 1.2 13 阶段清单与模型依赖

| # | Stage | 作用 | 用 LLM？ | 当前模型 | 引擎/模型 |
|---|-------|------|:---:|---------|----------|
| 1 | parse | 解析小说与角色标注 | 否 | — | 正则 `parser.py` |
| 2 | gender | 识别角色性别 | **是** | `sensenova-6.7-flash-lite` | SenseNova |
| 3 | emotion | 标注逐句情绪+语气 | **是** | `sensenova-6.7-flash-lite` | SenseNova |
| 4 | performance | 角色档案 + 逐句表演指导 | **是** | `sensenova-6.7-flash-lite` | SenseNova |
| 5 | tts | 逐句音色克隆合成 | 否 | — | VoxCPM（本地） |
| 6 | splice | 拼接完整语音 | 否 | — | pydub |
| 7 | bgm-segment | 划分 BGM 场景 | **是** | `sensenova-6.7-flash-lite` | SenseNova |
| 8 | bgm-label | 标注 BGM 类型+提示词 | **是** | `sensenova-6.7-flash-lite` | SenseNova |
| 9 | bgm-generate | 生成 BGM 音频 | 否 | — | ACE-Step（本地） |
| 10 | bgm-mix | 语音+BGM 混音 | 否 | — | pydub/ffmpeg |
| 11 | illustration-plan | 规划插图 | **是** | `deepseek-v4-flash`（已是目标） | SenseNova |
| 12 | illustrations | 提示词审核+生图 | **是** | `sensenova-6.7-flash-lite` | 本地文生图 |
| 13 | video | 字幕+动态视频 | 否 | — | MiniMax H3 |

> 关键观察：只有 `illustration-plan`（`illustration_planner.py`）已经在用默认 `LLMClient()` = `deepseek-v4-flash`；其余 LLM 阶段全部走 `for_flash_lite` 被钉在 flash-lite 上。

---

## 2. LLM 现状与统一替换方案

### 2.1 当前 LLM 架构（问题所在）

`backend/app/core/llm_client.py`（683 行）是一个**自研的多账号 OpenAI 兼容客户端**，核心逻辑：

- 直连 `https://token.sensenova.cn/v1`，从 `config/sensenova_apikeys` 读 **7 个 key**；
- 自己实现 round-robin 轮询、每账号冷却（quota/rate-limit/retry 三类冷却，最短 5 小时）；
- 每账号调用窗口计数（1500 次 / 5 小时），持久化到 `logs/sensenova_quota_state.json`；
- 失败时 fallback 到 Agnes（`https://apihub.agnes-ai.com`）；
- 每次调用写一条 telemetry JSONL。

这套东西在**你还没有本地代理之前**是必要的；但既然现在 opencode 里已经有「20 账号轮询」的本地代理，这整块轮询/冷却/fallback 逻辑都可以删掉，变成一个几十行的薄客户端。

### 2.2 opencode 里的 SenseNova Pool 代理（复用目标）

来自 `~/.config/opencode/opencode.jsonc`：

```jsonc
{
  "provider": {
    "sensenova-pool": {
      "npm": "@ai-sdk/openai-compatible",
      "options": {
        "baseURL": "http://127.0.0.1:18787/v1",
        "apiKey": "local-****（已脱敏，实际值见 ~/.config/opencode/opencode.jsonc）"
      },
      "models": {
        "deepseek-v4-flash": { "limit": { "context": 1048576, "output": 65536 } }
      }
    }
  }
}
```

要点：

- 它是一个 **OpenAI-compatible 的本地代理**（`/v1/chat/completions`），20 账号轮询由代理层完成；
- 单固定 `apiKey`，**不需要**再维护多 key、轮询、冷却；
- `deepseek-v4-flash`：**1M context / 64K output**，远大于当前 flash-lite 的 256K context。

### 2.3 统一替换方案

**改 `llm_client.py`（或直接重写一个薄客户端）**：

| 项 | 现状 | 目标 |
|---|---|---|
| base_url | `https://token.sensenova.cn/v1` | `http://127.0.0.1:18787/v1` |
| api_key | 7 个 key 轮询（`config/sensenova_apikeys`） | 单个固定 key |
| 模型 | `sensenova-6.7-flash-lite`（被钉死） | `deepseek-v4-flash` |
| 轮询/冷却 | 自研（约 400 行） | **删除**，交给代理 |
| Agnes fallback | 有 | **删除** |
| context 窗口 | 256K | **1M** |

**具体改动点**：

1. 删除 `LLMClient` 里的多账号轮询、冷却、quota state、Agnes fallback 全套逻辑，只保留 `chat()`（一次 `requests.post`）+ 遥测 + 上下文检查。
2. 把 `for_flash_lite()` 工厂**改为返回 deepseek-v4-flash**（或直接删掉工厂，全项目统一用默认构造）。
3. 删除 `bgm_segmenter.py:36` 的硬断言 `if model_name != SENSENOVA_FLASH_LITE_MODEL: raise ...`。
4. 删除 run_full.py 里所有 `SENSENOVA_FLASH_LITE_MODEL` 的 checkpoint 模型校验（`meta.get("model") != SENSENOVA_FLASH_LITE_MODEL`），改为记录实际模型名。

**⚠️ 副作用（必须知道）**：所有阶段 checkpoint 的 `model` 字段变化后，旧 checkpoint 会判定不兼容而**全部重跑**。这是预期行为——换更强模型本就应该重新生成以获得更高质量，且你有 20 账号轮询，速度不再是瓶颈。

**额外收益（高质量追求）**：1M context 意味着可以：
- `emotion` / `performance` 阶段的 tool-calling 检索范围可以更广（`read_lines` 能读更远的上下文，而不止当前 ±60 行）；
- `performance` 的角色档案可以带更完整的角色生平证据（通过工具按需检索）；
- `bgm-segment` 单 chunk 的分割范围可以适度扩大，减少 chunk 拼接带来的边界不一致。

---

## 3. 逐阶段分析（Stage 1–10）

### 3.1 parse（解析）— `backend/app/core/parser.py`（188 行）

**现状**：纯正则解析。识别章节标题（第X章 / Chapter X / 序章等）、「」对话（日式「」、行首说话人冒号、行内对话、一行多对话）、旁白（>5 字叙述）。

**评价**：不算臃肿，但有几处脆弱：

- 对话提取正则非常 ad-hoc（`DIALOGUE_JP` / `DIALOGUE_CN` / `DIALOGUE_INLINE` 多套并存，边界情况靠猜）；
- 「旁白」判定阈值 `len(text) > 5` 硬编码，短叙述被丢弃；
- 一行多「」对话的说话人回退逻辑（`original_speaker`）容易错位。

**改进建议**：

- 说话人标注本来就在 `labels.txt` 里由上游（opencode-novel-loop）给好了，本项目 parser 只负责对齐「文本 ↔ 说话人」，**不需要重新猜说话人**。可以把「从文本推断说话人」的几条正则路径砍掉，只保留「按 labels 顺序对齐 + 旁白」。
- 章节识别可抽成独立、可测试的小函数（已有 `extract_chapters`，主流程却没复用）。

**是否用 LLM**：不建议。解析是确定性任务，正则/状态机最可靠、最快、可测试。用 LLM 反而引入不稳定。**无需 LLM，无需替换引擎。**

### 3.2 gender（性别识别）— `backend/app/core/gender_identifier.py`（590 行）

**现状**：多 Agent 证据审查。主 Agent 用 tool-calling（`search_novel` / `read_lines` / `get_dialogues`）主动检索证据 → `submit_gender`；再独立 reviewer；分歧时 final adjudicator。证据必须引用行号，不能猜（unknown 是合法结果）。

**评价**：设计质量高（证据驱动、防锚定、可审计），但**过度工程**：

- `NovelIndex.evidence_packet()` 已经预构建了角色证据包（12 处上下文 + 12 条台词），主 Agent 却还要 tool-call 再次全文检索——**同一份证据被反复检索**，白白多烧 token 和调用；
- 每个角色平均触发 2–4 次 LLM 调用（primary 多步 tool-call + review + 可能的 adjudicator），一本 20 角色的书就是 60–80 次调用，且都用 flash-lite（质量差）。

**改进建议**：

- 换 `deepseek-v4-flash` 后，**保留 tool-calling**（`search_novel` / `read_lines` / `get_dialogues` 让模型主动检索证据），单角色一次调用，维持「主分析 → 双盲 review → 分歧裁决」的多 Agent 结构。
- `NovelIndex.evidence_packet()` 预构建的证据包可保留作为「起点线索」，但不强制塞满，让模型自己决定还要不要继续检索。

**是否可复用开源**：没有专门开源模型。性别判断依赖「名字 + 原文上下文 + 人称代词」，本质是 LLM 阅读理解任务。保持 LLM，靠换模型 + 精简流程提效。

### 3.3 emotion（情绪+语气标注）— `backend/app/core/emotion_labeler.py`（613 行）

**现状**：与 gender 同构的多 Agent 框架。7 情绪 × 7 语气，逐句标注，`submit_emotion` 强制包含 `evidence_lines`（含目标行）。双盲 review + 分歧裁决。

**评价**：标注质量设计好，但**成本结构糟糕**：

- **逐句**独立 LLM 调用（一本书几千句 → 几千次调用），每次还带 review，翻倍；
- `context()` 只带 ±60 行上下文，对长程情绪弧线（角色情绪变化、场景铺垫）理解不足；
- `memory` 参数被 `del memory` 丢弃（代码里明确写了不启用），连续性全靠局部上下文。

**改进建议**：

- 换 deepseek-v4-flash 后，**保持逐句标注（单句一次调用）+ 保留 tool-calling**（`read_lines` / `search_novel`），不做批量——每次只判断一句的情绪，保证专注度。
- 修复 `memory` 参数被 `del memory` 丢弃的问题，把**前几句的情绪结论作为连续性记忆**注入，改善长程情绪一致（这是目前质量损失点）；
- 旁白当前是跳过的（`label_all_emotions` 里 `speaker in {旁白, narrator}` 直接 continue）——**旁白也需要语气**（冷静/紧张/舒缓），建议为旁白单独建立情绪判定。

**是否可复用开源**：对话情感分类有开源模型（如 emotion 分类器），但都是**短文本单句**，无法理解小说上下文与潜台词。保持 LLM 更合适。

### 3.4 performance（表演导演）— `backend/app/core/performance_director.py`（2716 行，最大）

**现状**：项目最核心、最复杂的阶段。两个子流程：

1. **角色档案（profile）**：为每个角色建立表演档案（语速/音量/动作 cue/台词风格），三阶段决策（primary → review → final）。
2. **逐句表演指导（direction）**：每句台词生成自然语言表演控制（如「呼吸稍急，语速加快，重读」），三阶段决策，带 continuity（`_recent_continuity` / `_continuity_hash` 传递前文状态）。

输出字段丰富（pace 7 档、volume 7 档、scene_relation、action cues 等），校验极严格（`_ProfileValidator` / `_PerformanceValidator`）。

**评价**：这是项目的护城河，理念先进（多 Agent 双盲 + 裁决 + 证据引用），但代码是**四个文件里最臃肿的**：

- 2716 行里，**一半以上是样板**：schema 定义、validator、checkpoint 读写、usage 统计、evidence 校验，真正的「导演逻辑」只占一小部分；
- 每句台词都要 primary + review + 可能 adjudicator，一本几千句台词 → **上万次 LLM 调用**，且全用 flash-lite（这正是你感到慢的根因）；
- 控制词需要被压缩到 32 字符（`control_max_chars`），因为 VoxCPM 的输入限制，导致表演指导信息被大幅截断。

**改进建议**：

- 换 deepseek-v4-flash（质量更高），**保持逐句 + 保留 tool-calling + 固定三阶段**（primary → 盲审 review → 终审裁决）的结构不变（这是护城河，不砍；注意它与 gender/emotion 的「可选 review」不同，是无条件三阶段）；
- 优化点是**输入大方**：用 1M context 让 primary/review 通过 tool-calling 读取更远原文，提升每句的上下文理解，而非削减 review 次数；
- 角色档案**单角色一次生成**（保留 tool-calling 检索证据），输出单个角色的完整档案；
- 若 TTS 换成支持 instruct 的 CosyVoice 3（见 §5.1），**控制词不再需要压缩到 32 字符**，可以直接把完整表演指导喂给 TTS，质量上限显著提升。

**是否可复用开源**：没有现成开源方案。这是纯 LLM 编排任务，价值就在 prompt 工程 + 多 Agent 裁决设计。保持自研，但要**抽象通用框架**（见 §4）。

### 3.5 tts（VoxCPM 合成）— `run_full.py` + `backend/voxcpm_batch.py`

**现状**：`run_full.py` 里 `step_tts` / `run_voxcpm_tasks` 等约 1000 行，核心是：

- 用 `create_voxcpm_script()` **动态生成一段 Python 脚本字符串**（约 200 行的 f-string/heredoc），写到临时文件；
- 用另一个 Python 解释器（`E:\projects\音色克隆\VoxCPM\.venv\python.exe`）子进程执行，加载 `backend/models/VoxCPM2` 权重；
- 通过文件 + JSON 结果回传，主进程轮询 checkpoint 文件。

**评价**：这是**全项目最该重构的一处**。问题：

- 跨进程 + 动态生成脚本 + 硬编码绝对路径（`E:\projects\音色克隆\VoxCPM`），极其脆弱，换个机器/路径就崩；
- 引擎本身（VoxCPM）已经很好（tokenizer-free、RTF 0.15、1.8M 小时训练），但集成方式把它拖垮了；
- `normalize=false` 是为了绕开 Windows 中文路径下 `kaldifst` 读不了 FST 的坑——这是引擎依赖问题，不是业务问题。

**改进建议**：

- **把 VoxCPM 独立成一个常驻 HTTP 服务**（官方仓库自带 FastAPI server + `api_concurrent.py`），主流程通过 HTTP 调用，彻底去掉「动态生成脚本 + 跨进程 + 硬编码路径」；
- 或至少把 VoxCPM 的代码/环境并进本项目（`backend/models/VoxCPM2` 权重已在本地），用一个统一的 `TTSProvider` 抽象（当前 `app/tts/base.py` 已有雏形）收敛 `voxcpm` / `edge-tts` 两套实现；
- 控制词压缩逻辑（`_compact_oversized_control`）可以随 TTS 升级而放宽。

### 3.6 splice（拼接）— `backend/app/core/splicer.py`（394 行）

**现状**：`AudioSplicer` 用 pydub 按顺序拼接各句 WAV，处理句间停顿。

**评价**：中规中矩，pydub 对长音频（全书几小时）内存占用高（一次性 decode），但不至于重写。

**改进建议**：改为**流式/分段拼接**或用 `ffmpeg` concat（对超长音频更稳），减少内存峰值。非关键路径，优先级低。

### 3.7 bgm-segment（BGM 场景分割）— `bgm_segmenter.py` 第一部分

**现状**：把整本小说分成若干 BGM 场景。用 chunk 方式（`segment_novel_chunked` 把小说切 4 块 + overlap），每块 LLM 分割，再做边界 review（`_review_boundary`）。

**评价**：chunk + 边界 review 的设计是对的（长小说无法一次喂），但：

- chunk 数硬编码 4、overlap 10 行，没有按 context 自适应；
- 边界 review 是另一次 LLM 调用，成本和边界不一致风险并存。

**改进建议**：换 deepseek-v4-flash 的 1M context 后，**单 chunk 分割范围适度扩大**（chunk 数动态化），减少拼接边界问题；边界 review 保留但只在 chunk 交界处做。

### 3.8 bgm-label（BGM 类型+提示词标注）— `bgm_segmenter.py` 第二部分

**现状**：每个场景标注 8 类之一（daily/suspense/battle/sad/romantic/epic/comedy/horror）+ 音乐提示词（tempo/调式/情绪弧线）+ 负向提示（无 vocal/无 SFX 等），`label_bgm_types` + `_request_bgm_decision`（双盲 review）。

**评价**：设计完整（把「避免 vocal/SFX」等约束写进提示词，很有价值）。但这里**又出现了硬断言**（`_flash_lite_model` 强制 flash-lite），且逐场景 LLM 调用 + review，成本高。

**改进建议**：换 deepseek-v4-flash，**保持逐场景标注 + 保留 tool-calling**（单场景一次调用），去掉 `_flash_lite_model` 的硬断言。

### 3.9 bgm-generate（ACE-Step 生成）— `bgm_generator.py`（227 行）+ `run_full.py`

**现状**：`bgm_generator.py` 做「BGM 类型 → ACE-Step 英文 caption」映射 + 场景提示词拼接 + manifest。调用 ACE-Step SDK（`acestep-v15-turbo` + `acestep-5Hz-lm-1.7B`）本地生成 60 秒片段 × 3 变奏。

**评价**：**引擎选型已经是最优**（ACE-Step 1.5 是开源音乐生成里 Apache 2.0 + 商业可用 + 4 分钟 + 8GB VRAM 的最佳选择）。这部分不需要换引擎。

**改进建议**：

- caption 映射（`BGM_PROMPTS` 8 类）是写死的英文模板，可**让 LLM 直接产出每场景的 ACE-Step caption**（bgm-label 已经产出了 `bgm_music_prompt`，`build_segment_bgm_prompt` 也在用），进一步减少「类型 → 模板」的中间损耗，让音乐更贴合场景；
- `thinking=false`（禁用语义音频码路径，只用纯 DiT）是当年踩坑后的选择，升级 ACE-Step 版本后可重新评估。

### 3.10 bgm-mix（混音）— `bgm_mixer.py`（519 行）

**现状**：pydub 把语音 + BGM 混音，BGM 音量 `-8dB`，处理淡入淡出。

**评价**：常规混音，功能 OK。非瓶颈，优先级低。

---

## 4. 架构层面的「重复造轮子」

这是你「感觉代码有点差」的根因，按严重程度排序：

### 4.1 多 Agent 审查框架重复实现了 4 遍（最严重）

`gender_identifier.py` / `emotion_labeler.py` / `performance_director.py` / `bgm_segmenter.py` 各自完整复制了：

- `NovelIndex`（全文检索 / read_lines / 证据包）
- tool specs 定义（`_schema` / `TOOL_SPECS` / `REVIEW_TOOL`）
- `_run_primary` / `_run_review`（多步 tool-call 循环）
- `_validate` / validator 类
- checkpoint 读写（`_atomic_write_json`、`source_hash`、`pipeline_version`、`model` 校验）
- usage 统计（`_normalise_usage` / `_usage_delta` / `_merge_usage`）

四个文件加起来 **约 5400 行，其中可能一半是重复样板**。应抽成一个通用框架，例如：

```
backend/app/core/review_orchestrator.py
  ├── NovelIndex（统一证据检索）
  ├── AgentStage（primary / review / adjudicator 三阶段通用驱动）
  ├── CheckpointManager（统一 source_hash + version + model 校验 + 原子写）
  └── UsageTracker（统一 token/call 统计）
```

各阶段只提供「自己的 prompt + 输出 schema + 校验函数」，框架负责其余。这会让每个阶段文件瘦身到几百行，且新增阶段（比如以后做「动作描写导演」）零成本复用。

### 4.2 `run_full.py` 4690 行巨型编排

把「阶段编排、VoxCPM 脚本生成、缓存校验、checkpoint、进度监控、CLI 参数」全塞一个文件。建议拆成：

- `run_full.py` 只保留 CLI + 阶段循环；
- 每个 stage 移到独立模块（部分已在 `backend/app/core/` 有对应实现，run_full 里只是胶水 + 缓存校验，可继续下沉）。

### 4.3 VoxCPM 跨进程动态脚本（见 §3.5）

用 `create_voxcpm_script()` 动态生成代码字符串再跨进程执行，是全项目最脆弱的耦合。应改为常驻 HTTP 服务或统一 TTS Provider。

### 4.4 checkpoint 的 model 字段耦合过深

所有阶段 checkpoint 都把 `model` 作为兼容性判定的一部分，导致「换模型 = 全部重跑」。这本身没错（换模型确实要重跑），但应把「模型名」作为**单一配置源**（而不是散落在 4 个文件的常量里），换模型只需改一处。

---

## 5. 可复用开源方案调研（2026-09 最新）

### 5.1 TTS 音色克隆（VoxCPM vs 替代）

| 模型 | 零样本克隆 | 情感/指令控制 | 中文 | 协议 | 结论 |
|------|:---:|------|:---:|------|------|
| **VoxCPM 1.5**（现用） | 3–10s | 上下文自适应 + 中文前缀 | 强（CER 0.93%） | 开源（OpenBMB） | **保留**，升级集成方式 |
| **CosyVoice 3 (Fun-CosyVoice3-0.5B)** | 3s | **Instruct 自然语言控制情感/语速/音量** | 强（9 语言+18 方言） | Apache 2.0 | **最强替代候选** |
| IndexTTS2 | 3s | 7 种情感 + 精确时长 | 强（WER 1.01%） | Apache 2.0 | 情感/时长控制佳 |
| Fish Speech 1.5 | 10–30s | 多说话人对话原生 | 好 | Apache 2.0 | 多角色对话场景可选 |
| GPT-SoVITS | 少样本 | 中等 | 社区最活跃 | — | 备选 |

**核心判断**：

- **VoxCPM 不需要换**。它已经是当前开源里 CER 最低（0.93%）、RTF 最快（0.15）的音色克隆引擎之一。要修的是集成方式（§3.5），不是引擎。
- **CosyVoice 3 的 Instruct 模式**与你项目「表演导演 → 逐句自然语言表演控制」的理念**天然同构**：现在表演指导要压缩成 32 字符中文前缀喂 VoxCPM，若换 CosyVoice 3，可直接把完整自然语言指导作为 instruct 输入，**表演质量上限大幅提升**。值得作为 A/B 对比验证。
- **IndexTTS2** 的「精确时长控制」对有声书（对齐 BGM/字幕）有独特价值，可关注。

**建议动作**：先不动引擎，把集成方式改成常驻 HTTP 服务；同时拿 CosyVoice 3 做一轮小样本 A/B（选几段复杂情绪台词），对比后再决定是否切换。

### 5.2 BGM 音乐生成

| 模型 | 商业可用 | 人声 | 时长 | 结论 |
|------|:---:|:---:|:---:|------|
| **ACE-Step 1.5**（现用） | Apache 2.0 ✅ | 支持 | ~4min | **保留，已是最优** |
| HeartMuLa | Apache 2.0 ✅ | 全曲 | 长 | 可选升级 |
| YuE | Apache 2.0 ✅ | 全曲 | 长 | 结构控制强，VRAM 高 |
| MusicGen | ❌ CC-BY-NC | 仅器乐 | 30s | 不适用（非商用） |

**结论**：ACE-Step 已经是开源里最合适的（商业可用 + 器乐/人声 + 快），**不需要换**。MusicGen 是非商用协议，务必避开。

### 5.3 性别 / 情感 / 表演导演

**没有现成开源替代**。这些是「整本小说上下文理解」任务，开源情感分类器只处理短文本单句，无法替代。正确做法是：换更强 LLM + 保留多 Agent 架构 + 单对象聚焦（见 §3）。

---

## 6. 重构路线图（建议分 3 步）

> **开发协作方式**：一个阶段一个阶段推进，每完成一个阶段就自测 + 提交推送。复杂子任务或方案讨论可借助 opencode（模型用 `kimi-k3` / `deepseek-v4-pro` / `GLM 5.2`）做协作开发，多讨论可能产生新灵感。

### 第一步（见效最快，1–2 天）：LLM 换血

1. 重写 `llm_client.py` 为薄客户端（本地代理 + deepseek-v4-flash + 单 key），删轮询/冷却/fallback；
2. 统一 `for_flash_lite` → deepseek-v4-flash，删硬断言与散落的模型常量；
3. 改 `config/config.yaml` 的 `llm` 段指向新代理；
4. 跑通 dry-run 验证各阶段 cache 判定，确认旧 checkpoint 会按预期重跑。

### 第二步（进行中）：抽通用框架 + 精简流程（架构不变）

1. 抽 `review_orchestrator.py` 通用框架，让四个 LLM 阶段复用（**保留多 Agent + tool-calling 架构**，只消除重复样板代码）；
2. ✅ 修复 `emotion` 的连续性记忆（`memory` 参数被 `del memory` 丢弃的问题）—— 已完成并提交；
3. ~~`emotion`/`gender` 低置信度才 review~~ **不做**：用户明确质量优先，双盲 review 是质量保证核心，依赖模型 self-confidence 的降级有过度自信风险，保留 `always_verify=True`；
4. 旁白纳入情绪判定范围（需单独设计旁白 taxonomy + 跨 `emotion_labeler` / `run_full` 联动）。

### 第三步（3–5 天）：TTS 集成解耦 + 可选引擎 A/B

1. VoxCPM 改成常驻 HTTP 服务（复用官方 FastAPI server），去掉跨进程动态脚本与硬编码路径；
2. 统一 TTS Provider 抽象，收敛 voxcpm / edge-tts 两套；
3. 可选：CosyVoice 3 Instruct 模式小样本 A/B，评估是否切换。

> 生图（illustration-plan / illustrations）与视频（video）不在本次范围，但其 LLM 阶段（illustration_planner 已在用 deepseek-v4-flash，visual_prompt_auditor 用 flash-lite）也应在第一步一并切到统一代理。

---

## 7. 附录：关键文件清单

| 文件 | 行数 | 角色 |
|------|---:|------|
| `scripts/run_full.py` | 4690 | 13 阶段编排（应拆分） |
| `backend/app/core/llm_client.py` | 683 | LLM 客户端（应重写为薄客户端） |
| `backend/app/core/performance_director.py` | 2716 | 表演导演（应抽框架） |
| `backend/app/core/bgm_segmenter.py` | 1495 | BGM 分割+标注（含硬断言） |
| `backend/app/core/emotion_labeler.py` | 613 | 情感标注 |
| `backend/app/core/gender_identifier.py` | 590 | 性别识别 |
| `backend/app/core/bgm_generator.py` | 227 | ACE-Step caption + manifest |
| `backend/app/core/bgm_mixer.py` | 519 | 混音 |
| `backend/app/core/splicer.py` | 394 | 拼接 |
| `backend/app/core/parser.py` | 188 | 解析 |
| `backend/app/core/ollama_client.py` | 204 | 旧 ollama 客户端（已被 _archive 弃用） |
| `backend/voxcpm_batch.py` | — | 旧文件中转方案（建议废弃） |
| `config/config.yaml` | — | 全局配置 |
| `config/sensenova_apikeys` | 7 key | 旧多 key（新方案后废弃） |
| `~/.config/opencode/opencode.jsonc` | — | SenseNova Pool 代理（复用） |

---

## 8. 实施前需自查的技术项（非用户决策项）

> 以下三项是我在动手前需要自行验证的技术事实，不需要用户逐一确认。

1. **`deepseek-v4-flash` 是否支持 tool-calling（function calling）？** ✅ **已实测验证（2026-09-20）**：`finish_reason=tool_calls`，`arguments` 为标准 JSON 字符串，多轮 tool 回传正常。多 Agent 架构可无缝切换。注意：它是**推理模型**（usage 里有 `reasoning_tokens`），单次简单调用约 1.6–1.9s。
2. **代理的并发上限**：单对象聚焦意味着请求次数多（逐句/逐人/逐场景），需要**并发调度**来压时间。需确认 20 账号轮询能扛多大并发（同时几个请求）、本地代理进程是否限流。
3. **VoxCPM2 权重对应哪个官方版本**（0.5B 还是 1.5）？决定是否值得升级到 VoxCPM 1.5（44.1kHz、patch-size=4、质量更高）。
