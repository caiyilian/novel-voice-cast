# StemDeck — 免费本地人声/乐器分离工具

> **官网**: https://stemdeck.app  
> **GitHub**: https://github.com/stemdeckapp/stemdeck  
> **许可**: Apache-2.0

---

## 一、是什么

StemDeck 是一个**免费、开源、完全本地运行**的音频分轨（stem separation）工具。它能将一首歌曲分离成最多 **6 个独立音轨**：

| 音轨 | 说明 |
|------|------|
| Vocals | 人声 |
| Drums | 鼓 |
| Bass | 贝斯 |
| Guitar | 吉他 |
| Piano | 钢琴 |
| Other | 其他乐器 |

基于 Meta AI 开源的 **Demucs**（`htdemucs_6s`）神经网络模型，音频处理全部在本机完成，**不上传任何数据**。

---

## 二、适用场景

- **练习乐器** — 单独听鼓/贝斯/吉他，跟练
- **扒谱/转录** — 分离出某一声部细致分析
- **Remix / 二次创作** — 导出单轨重新混音
- **教学** — 展示歌曲各声部构成
- **研究** — 分析编曲结构

**对比商业工具（Moises / LALAL.AI）：**

| | StemDeck | 商业工具 |
|--|----------|----------|
| **价格** | 永久免费 | 免费版有限额/需订阅 |
| **运行方式** | 本地运行 | 上传到云端 |
| **账号** | 不需要 | 需要注册 |
| **隐私** | 音频不离机 | 上传到第三方服务器 |
| **网络** | 仅首次下载模型 + YouTube 导入 | 全程需联网 |
| **音轨数** | 6 | 最多 10 |
| **移动端** | ❌ | ✅ iOS/Android |
| **批量处理** | ❌ 一次一个 | 付费版支持 |
| **源码** | 开源 | 闭源 |

---

## 三、安装方式

### 方式一：桌面客户端（推荐）

去 [GitHub Releases](https://github.com/stemdeckapp/stemdeck/releases) 下载对应版本。

#### macOS

| 文件 | 适用 | GPU |
|------|------|-----|
| `StemDeck-macOS-arm64.dmg` | Apple Silicon (M1 及以上) | MPS 加速 |
| `StemDeck-macOS-x64.dmg` | Intel Mac | CPU only |

安装：打开 DMG → 拖入 Applications 文件夹。首次启动自动下载 Python 运行时 (~500 MB) + FFmpeg + Demucs 模型 (~170 MB)。

> ⚠️ macOS 可能弹出 Gatekeeper 拦截，**右键应用 → 打开** 即可绕过。

#### Windows

| 文件 | 适用 | 大小 |
|------|------|------|
| `StemDeck-Windows-x64.zip` | 通用 (CPU) | ~700 MB |
| `StemDeck-Windows-x64.NVIDIA.zip` | 有 NVIDIA 显卡 (CUDA) | ~1.6 GB |

安装：解压任意目录 → 运行 `StemDeck.exe`。首次启动自动下载 FFmpeg 和模型。

---

### 方式二：自托管 Web 服务器（Python）

**前提**：Python 3.12+、ffmpeg、[uv](https://github.com/astral-sh/uv)

```bash
git clone https://github.com/stemdeckapp/stemdeck.git && cd stemdeck
uv sync
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
```

打开 **http://localhost:8000**。

#### 启用 NVIDIA CUDA 加速（Windows）

```bash
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
$env:STEMDECK_DEMUCS_DEVICE = "cuda"
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
```

##### macOS/Linux 一键脚本

```bash
git clone https://github.com/stemdeckapp/stemdeck.git && cd stemdeck
./run.sh setup    # 安装 ffmpeg + uv，执行 uv sync
./run.sh start    # 启动服务器
```

控制命令：

| 命令 | 作用 |
|------|------|
| `./run.sh setup` | 初始化环境 |
| `./run.sh start` | 后台启动 |
| `./run.sh stop` | 停止 |
| `./run.sh restart` | 重启 |
| `./run.sh status` | 查看运行状态 |

---

### 方式三：Docker

```bash
docker compose -f build/docker-compose.yml up --build
```

输出文件存储在宿主机的 `./jobs/` 目录。macOS Docker 不支持 GPU 直通。

---

## 四、使用方法

### 基本流程

1. **导入音频**
   - 拖拽 MP3 / WAV / FLAC 文件到导入栏
   - 或粘贴 YouTube 链接
2. **选择要提取的音轨**（可选）— 点击音轨芯片选择子集
3. **点击 Process** — 依次经历 `Uploading/Downloading → Analyzing → Separating → Mixing tracks`
4. **在混音台操作**
   - ▶ **Play / Pause / Stop** — 主播放控制
   - **M** — 静音该轨
   - **S** — 独奏该轨（可叠加多个）
   - **Monitor** — 仅独奏该轨，清除其他
   - **音量推子** — 拖动调节，双击重置 0 dB
   - **滚轮** — 微调音量；**Shift+滚轮** — 粗调
   - 工具栏 **Reset / Mute / Solo** — 作用于全部音轨
5. **放大波形** — `+` / `-` / `Fit` 按钮 或 `Ctrl/Cmd+滚轮`
6. **循环播放** — 在标尺上拖选区域 → 点击 **Loop**
7. **导出** — 点击 **Download Mix** 下载选中的混音 WAV

### 键盘快捷键

| 按键 | 功能 |
|------|------|
| `Space` | 播放/暂停 |
| `[` | 后退 5 秒 |
| `]` | 前进 5 秒 |
| `L` | 循环开关 |
| `I` | 设置循环入点 |
| `O` | 设置循环出点 |

---

## 五、配置选项（环境变量）

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `STEMDECK_DEMUCS_DEVICE` | auto | 强制指定设备：`cuda` / `mps` / `cpu` |
| `STEMDECK_DEMUCS_MODEL` | `htdemucs_6s` | Demucs 模型名称 |
| `STEMDECK_JOBS_DIR` | `./jobs` | 任务输出目录 |
| `STEMDECK_DATA_DIR` | (无) | 便携模式根目录 |
| `STEMDECK_CACHE_DIR` | `<data>/cache` | 模型缓存目录 |
| `STEMDECK_DOWNLOADS_DIR` | `<data>/downloads` | yt-dlp 下载暂存 |
| `STEMDECK_MAX_DURATION_SEC` | `1200` | 音频最大时长（秒） |
| `STEMDECK_JOB_TTL_SECONDS` | `86400` | 任务文件保留时间（秒，24h） |
| `STEMDECK_MAX_PENDING_JOBS` | `3` | 最大排队任务数 |
| `STEMDECK_TIMEOUT_DEMUCS_STALL` | `1800` | Demucs 无输出超时（秒） |

---

## 六、输出目录结构

```
jobs/<job_id>/
└── stems/
    ├── vocals.wav
    ├── drums.wav
    ├── bass.wav
    ├── guitar.wav
    ├── piano.wav
    ├── other.wav
    ├── original.wav    # 子集模式下，未被选中音轨的补集
    └── mix.wav         # 选中音轨的混音
```

---

## 七、注意事项

1. **首次分离很慢** — Demucs 需下载 ~170 MB 模型权重，之后缓存使用
2. **确保 ffmpeg 已安装** — 否则报错 `ffmpeg: command not found`
3. **YouTube 支持是便利功能** — 使用者需自行确保有处理该内容的权利，遵守 YouTube ToS
4. **GPU 加速** — 查看启动日志 `device=mps` / `device=cuda`；如果显示 `cpu` 则 torch 是 CPU 版
5. **页面刷新不会中断后台任务** — 任务在服务端继续运行
6. **无移动端、无批量处理** — 适合桌面端单曲处理

---

## 八、社区与支持

| 平台 | 链接 |
|------|------|
| GitHub | https://github.com/stemdeckapp/stemdeck |
| Discord | https://discord.gg/2MVsWqaPRe |
| Reddit | https://www.reddit.com/r/StemDeckApp/ |
| Instagram | https://www.instagram.com/stemdeck |
| X / Twitter | https://x.com/StemDeckApp |
| 官网 | https://stemdeck.app |
