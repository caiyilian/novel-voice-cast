# MiniMax H3 全时长动态视频集成

本项目的 `video` 阶段可以调用独立服务器上的 MiniMax H3 服务，把已经完成的 VoxCPM 人声、ACE-Step BGM、中文字幕和小说时间轴合成为横、竖两个动态长片。当前默认模式是 `continuous-chain`：每一帧都必须由真实动态视频覆盖，合成器不会延长末帧、循环短视频、慢放视频或改变人声速度。

## 默认方案：continuous-chain

全书继续使用 702 个已审核的叙事节拍作为粗镜头，但每个粗镜头会按真实音频时长进一步拆成若干条 5～10 秒微镜头：

1. 微镜头数量由 WAV 时间轴和 H3 的 `17k+5` 帧规则自动计算，不硬编码数量；
2. 每条记录保存粗镜头、微镜头、覆盖起止时间、动作阶段、镜头方向和画幅构图；
3. 每条微镜头都绑定其准确时间段内的小说原文、说话人和 `performance_control`，让人物动作与最终 VoxCPM 音频节奏一致；
4. 提示词同时注入 `docs/角色卡.md` 和 `output/character_visual_memory.json` 中该角色、该剧情行范围的外貌、服装、位置与状态；
5. 同一局部动作可用上一条末帧做 I2VA，但链长默认不超过 3 条，之后主动 T2VA 切镜重锚定，避免人物和场景误差无限积累；
6. 若提供目标尾帧，生成器也支持 L2VA 或 FL2VA；
7. 下载后会真实解码并检查分辨率、时长、长时间近乎静止、黑屏及首尾锚点相似度，不合格时保存指标并更换随机生成结果重试；
8. 合成前逐帧核对动态覆盖。任意微镜头少一帧都会中止并指出缺口，不会以静帧补齐；
9. H3 自带音轨全部丢弃，最终音频仍是已经验收的 VoxCPM+BGM 母音频。

以当前第一卷 21,665.77 秒音频实测，702 个粗镜头会生成每画幅 2,487 条微镜头，单个粗镜头包含 1～16 条。竖屏和横屏共享语义分镜，但分别使用 7:9 与 16:9 构图指令，并使用独立断点。程序按“竖屏生成及成片 → 横屏生成及成片”串行执行，因此竖屏完成后无需等待横屏即可先得到竖屏成片。

## 旧素材迁移

新模式写入 `output/h3_video_continuous/`，不会删除或覆盖旧 `output/h3_video/`：

- 旧断点中每个粗镜头的第一条视频会重新执行解码和质量检查；
- 合格视频直接成为新微镜头的候选素材，并在新切点重新提取续写帧；
- 旧断点里已经提交的 `queued/running job_id` 会转移到新断点继续查询；
- 服务器若仍保留任务就下载复用；明确返回 404 时才按新提示词重提。

当前旧竖屏 702 条均可进入候选检查；旧横屏已有 72 条本地成功视频，第 73 条任务号也会尝试恢复。旧文件始终保持原位。

## 配置

`config/config.yaml`：

```yaml
video:
  output_path: "output/h3_continuous_portrait_7x9_subtitled.mp4"
  h3:
    enabled: true
    mode: "continuous-chain"
    endpoint: "http://172.31.102.189:8189"
    output_dir: "output/h3_video_continuous"
    reuse_output_dir: "output/h3_video"
    shot_plan_path: "backend/data/h3_shot_plan.json"
    visual_memory_path: "output/character_visual_memory.json"
    minimum_duration: 5
    maximum_duration: 10
    max_chain_length: 3
    max_freeze_ratio: 0.65
    max_black_ratio: 0.20
    min_anchor_similarity: 0.90
    request_timeout: 60
    poll_seconds: 15
    job_timeout: 14400
    max_attempts: 3
    generation_timeout: 15552000
    render_timeout: 604800
    portrait:
      width: 672
      height: 864
    landscape:
      width: 960
      height: 544
```

宽和高必须为 16 的倍数；H3 请求时长必须在 5～15 秒。`maximum_duration: 10` 是当前质量与单条稳定性的保守选择。`max_freeze_ratio` 判断最长近乎静止段占使用区间的比例，`max_black_ratio` 判断黑屏累计比例。`min_anchor_similarity` 用于核对 I2VA/FL2VA 生成视频是否真正从输入帧开始；旧成片抽样的正常值为 0.978～0.993，默认 0.90 留有压缩和轻微首帧运动余量，同时可拦截明显换景。

服务需要提供：

- `GET /api/health`
- `POST /api/generate`：T2VA 使用 JSON；带首帧或尾帧时使用 multipart
- `GET /api/status/{job_id}`
- `GET /api/download/{job_id}`

运行前健康检查：

```cmd
.venv\Scripts\python.exe -c "import requests; print(requests.get('http://172.31.102.189:8189/api/health', timeout=15).json())"
```

## 运行与断点续跑

现有音频、插图计划和提示词审核完成后，只运行视频阶段：

```cmd
cd /d E:\projects\novel-voice-cast
set "PYTHONUTF8=1"
.venv\Scripts\python.exe -u scripts\run_full.py --config config\config.yaml --from-stage video --to-stage video --log logs\h3_continuous_video.log
```

可以随时按 `Ctrl+C`，再次执行相同命令会：

- 先迁移尚未处理的旧候选片段和远程任务号；
- 复用输入哈希、提示词和依赖帧均匹配的成功微镜头；
- 对服务器仍在排队或运行的任务继续查询原 `job_id`；
- 服务器断电或网络中断期间保留任务号并持续等待；
- 只有服务器恢复后明确返回 404 才重提当前镜头；
- 复用已经编码完成且指纹匹配的本地粗镜头分段；
- 全部动态覆盖完成后才压制字幕和母音频。

主要新断点：

```text
backend/data/h3_shot_plan.json
output/h3_video_continuous/portrait/h3_clips.checkpoint.json
output/h3_video_continuous/portrait/h3_render.checkpoint.json
output/h3_video_continuous/landscape/h3_clips.checkpoint.json
output/h3_video_continuous/landscape/h3_render.checkpoint.json
```

默认新成片：

```text
output/h3_continuous_portrait_7x9_subtitled.mp4
output/h3_continuous_landscape_16x9_subtitled.mp4
```

## 监控

```cmd
.venv\Scripts\python.exe -u scripts\progress_monitor.py --config config\config.yaml --host 0.0.0.0 --port 8765
```

电脑访问 `http://127.0.0.1:8765/`；同一局域网手机访问 `http://电脑局域网IP:8765/`。监控 JSON 和网页会使用断点里的实际微镜头总数，并显示当前粗镜头/微镜头、远程任务号、动态覆盖秒数、本地编码数和 ETA。

## 兼容模式

`native-chain` 是旧版“一条粗镜头对应一条短视频”模式。它能复现旧结果，但长语音区间会停留末帧，不再推荐作为最终长片：

```yaml
video:
  h3:
    mode: "native-chain"
```

`illustration-bridge` 使用两张旧插图作为 FL2VA 首尾帧，要求横、竖插图完整存在：

```yaml
video:
  h3:
    mode: "illustration-bridge"
```

完全关闭 H3 并恢复静态插图视频：

```yaml
video:
  h3:
    enabled: false
```

## 质量与时间预期

当前每画幅规划 2,487 条，旧素材迁移后仍需要生成大量补充镜头。按服务器 5 秒约 7～10 分钟、10 秒约 15～20 分钟估算，横竖屏完整长跑可能持续数周。服务断电、网络中断和本地重启不会自动丢弃已经记录的远程任务。应结合日志、监控页和 checkpoint 中的 `coverage`、`success`、`quality`、`quality_failures` 判断真实进度，不能只按等待时长判断卡死。现有自动门禁能可靠检查文件、运动、黑屏和锚点断裂；人物审美、复杂肢体和剧情表意仍建议在正式发布前抽样人工复核。

更完整的设计原则、验收标准和实测覆盖数据见 [`MiniMax-H3全时长动态长片方案.md`](MiniMax-H3全时长动态长片方案.md)。
