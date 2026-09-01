# MiniMax H3 原生链式视频集成

本项目的 `video` 阶段可以调用独立服务器上的 MiniMax H3 服务，把原来的“静态插图停留”升级为有真实运动的横、竖版视频。默认方案不把 702 张旧插图逐张作为首尾帧，因此旧插图的构图、人物错误和画风波动不会成为 H3 成片的硬约束。

## 默认方案：native-chain

全书仍沿用插图计划的叙事节拍和 BGM 场景划分，但画面由 H3 自己建立和延续：

1. 每个 BGM 粗场景的第一个叙事节拍使用 T2VA，由审核后的视觉提示词建立新画面；
2. 同一粗场景内的后续节拍使用 I2VA，以上一条 H3 视频在真实音频边界处提取的续帧为首帧；
3. 进入下一个粗场景时主动重置为 T2VA，避免错误人物或画面漂移无限传递；
4. 每个节拍开头播放 H3 动画；如果语音时长超过 H3 片段，则停留在 H3 自己的续帧，而不是切回旧插图；
5. H3 自带音轨一律丢弃，最终只使用已经完成的 VoxCPM 人声、ACE-Step BGM 和中文字幕。

当前第一卷共有 702 个叙事节拍和 188 个 BGM 粗场景，因此每种画幅约生成 188 条 T2VA 和 514 条 I2VA。竖屏与横屏使用独立 checkpoint，默认串行执行，避免同时争抢同一台服务器 GPU。

## 配置

`config/config.yaml`：

```yaml
video:
  h3:
    enabled: true
    mode: "native-chain"
    endpoint: "http://172.31.102.189:8189"
    output_dir: "output/h3_video"
    minimum_duration: 5
    maximum_duration: 10
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

宽和高必须是 16 的倍数；H3 请求时长必须在 5–15 秒。`maximum_duration: 10` 是当前质量、运动长度和单条耗时之间的保守选择；若服务器验证 15 秒稳定，可以改为 15，但全书耗时会明显增加。

服务需要提供：

- `GET /api/health`
- `POST /api/generate`：无图片时接收 JSON；I2VA/FL2VA 时接收 multipart 图片
- `GET /api/status/{job_id}`
- `GET /api/download/{job_id}`

运行前健康检查：

```cmd
.venv\Scripts\python.exe -c "import requests; print(requests.get('http://172.31.102.189:8189/api/health', timeout=15).json())"
```

## 运行与断点续跑

现有音频、插图计划和提示词审核已经完成时，只运行视频阶段：

```cmd
cd /d E:\projects\novel-voice-cast
set "PYTHONUTF8=1"
.venv\Scripts\python.exe -u scripts\run_full.py --config config\config.yaml --from-stage video --to-stage video --log logs\h3_video.log
```

可以随时按 `Ctrl+C`。再次执行完全相同的命令会：

- 复用已经下载并通过分辨率、时长校验的 MP4；
- 对仍在服务器排队或运行的任务继续查询原 `job_id`，不会因本地断网立即重复提交；
- 如果服务重启后明确返回 404，才清除旧任务号并使用新随机噪声重提；
- 复用已经提取的续帧和已经编码的本地分段；
- 最后重新拼接字幕和现有音频。

主要断点：

```text
output/h3_video/portrait/h3_clips.checkpoint.json
output/h3_video/portrait/h3_render.checkpoint.json
output/h3_video/landscape/h3_clips.checkpoint.json
output/h3_video/landscape/h3_render.checkpoint.json
```

默认成片：

```text
output/h3_video_local_portrait_7x9_subtitled.mp4
output/h3_video_local_landscape_16x9_subtitled.mp4
```

局域网监控页会分别显示横、竖画幅的 H3 片段数、当前 `job_id` 状态、本地编码分段数和基于最近完成速度估算的 ETA：

```cmd
.venv\Scripts\python.exe -u scripts\progress_monitor.py --config config\config.yaml --host 0.0.0.0 --port 8765
```

电脑访问 `http://127.0.0.1:8765/`；同一局域网手机使用 `http://电脑局域网IP:8765/`。

## 兼容模式

如需复现“旧插图之间做首尾帧过渡”的方式，可改为：

```yaml
video:
  h3:
    mode: "illustration-bridge"
```

该模式才会要求原横、竖插图完整存在。它适合旧插图已经过人工精选、角色和构图都可信的项目；本项目当前默认不使用它。

完全关闭 H3 并恢复静态插图视频：

```yaml
video:
  h3:
    enabled: false
```

## 质量与时间预期

H3 是短视频模型，不能一次生成七小时连贯长片。`native-chain` 的目标是把长片拆成可恢复的局部连续镜头，同时用 BGM 场景边界限制漂移。长语音节拍中，动画结束后会停留在模型自身的稳定续帧；不会拉伸视频，也不会改变人声速度。

按服务器现有实测，5 秒约 7–10 分钟、10 秒约 15–20 分钟。两种画幅共约 1404 条，完整长跑可能持续数周。这是预期行为，不是卡死；请结合日志、桌面版进度或 checkpoint 中的 `success` 数量判断进度。
