# 插图生成 API 接入文档

> Novel Voice Cast 提供两套插图生成方案，可按需选择。

---

## 方案对比

| 维度 | PuLID v1.1 (SDXL) | Agnes AI (云端) |
|------|-------------------|----------------|
| **运行位置** | Linux 服务器（RTX 4090） | 云端 API |
| **费用** | 免费（自有硬件） | **免费**（`$0 / image`） |
| **角色一致性** | ✅ 强（参考图保持长相） | ⚠️ 图生图模式可参考，但不强 |
| **多图合成** | ❌ 不支持 | ✅ 支持（多张参考图合成一张） |
| **速度** | ~3-4 秒/张 | ~10-30 秒/张 |
| **并发** | ❌ 不支持 | ✅ 支持 |
| **需要代理** | ❌ 不需要 | ✅ 需要（`http://127.0.0.1:7890`） |
| **适用场景** | 角色一致性要求高的插图 | 快速出图、多图合成、复杂场景 |

---

## 技术栈

| 组件 | 说明 |
|------|------|
| **SDXL base 1.0** | 基座画图模型（2.6B 参数） |
| **PuLID v1.1** | 零样本身份保持插件，根据参考图保持角色长相 |
| **InsightFace** | 人脸检测，提取面部嵌入 |
| **FastAPI** | HTTP 服务框架 |

全部使用社区预训练权重，不训练、不微调。

## 服务器信息

| 项目 | 值 |
|------|----|
| 服务器地址 | `172.31.102.189:8001` |
| GPU | RTX 4090 (24GB) |
| 峰值显存 | ~14.7 GB |
| 生成速度 | ~3-4 秒/张（25 步） |

## 工作模式

### 模式一：有参考图（主模式）

用户提供角色参考图 → 生成新场景插图时保持该角色的长相特征。

适用场景：
- 小说有主要角色，用户能提供角色画像
- 需要角色在多张插图中长相一致

### 模式二：无参考图（回退）

纯文本 prompt 生成，不依赖任何参考图。

适用场景：
- 用户没有角色参考图
- 一次性场景插图（风景、建筑等）

---

## API 接口

### 健康检查

```
GET /health
```

返回：`{"status": "ok", "device": "cuda:1"}`

### 生成插图

```
POST /generate
Content-Type: multipart/form-data
```

**参数：**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `prompt` | string | 是 | — | 场景描述（英文效果最佳） |
| `ref_image` | file | 否 | — | 角色参考图（PNG/JPG），不传=无参考图模式 |
| `neg_prompt` | string | 否 | 见下方 | 负面提示词 |
| `seed` | int | 否 | -1（随机） | 随机种子 |
| `steps` | int | 否 | 25 | 推理步数（推荐 20-30） |
| `cfg` | float | 否 | 7.0 | 提示词引导强度 |
| `id_scale` | float | 否 | 0.8 | 身份保持强度（0.5-1.5，越高越像参考图） |
| `num_zero` | int | 否 | 20 | 身份可编辑性（10-30，越低越像参考图） |
| `height` | int | 否 | 1152 | 图片高度 |
| `width` | int | 否 | 896 | 图片宽度 |

**默认负面提示词：**
```
flaws in the eyes, flaws in the face, flaws, lowres, non-HDRi, low quality,
worst quality, artifacts noise, text, watermark, glitch, deformed, mutated,
ugly, disfigured, hands, low resolution, partially rendered objects,
deformed or partially rendered eyes, deformed, deformed eyeballs,
cross-eyed, blurry
```

**返回：** PNG 图片二进制

---

## 调用示例

### 安装依赖

```bash
pip install requests Pillow
```

### 无参考图

```python
import requests

BASE = 'http://172.31.102.189:8001'
resp = requests.post(f'{BASE}/generate', data={
    'prompt': 'anime style, a quiet medieval village street at sunset, '
              'cobblestone path, warm golden light, masterpiece',
    'steps': 20,
    'cfg': 7.0,
})
with open('output.png', 'wb') as f:
    f.write(resp.content)
```

### 有参考图

```python
import requests

BASE = 'http://172.31.102.189:8001'
with open('角色参考图.png', 'rb') as f:
    resp = requests.post(f'{BASE}/generate', data={
        'prompt': 'portrait of a cute wolf girl with brown hair and wolf ears, '
                  'medieval traveler outfit, smiling, warm autumn forest, anime style',
        'id_scale': 0.8,
        'num_zero': 20,
        'steps': 25,
    }, files={'ref_image': f})
with open('output.png', 'wb') as f:
    f.write(resp.content)
```

### 同一角色、多个场景（批量）

```python
import requests
from pathlib import Path

BASE = 'http://172.31.102.189:8001'

scenes = [
    'walking through a wheat field at sunset',
    'sitting by a campfire at night, starry sky',
    'standing on a medieval village market street',
]

with open('角色参考图.png', 'rb') as ref_f:
    ref_data = ref_f.read()

for i, scene in enumerate(scenes):
    prompt = f'portrait of a cute wolf girl with brown hair and wolf ears, {scene}, anime style, masterpiece'
    resp = requests.post(f'{BASE}/generate', data={
        'prompt': prompt,
        'id_scale': 0.8,
        'num_zero': 20,
        'seed': 42 + i,
    }, files={'ref_image': ('ref.png', ref_data, 'image/png')})
    Path(f'scene_{i+1}.png').write_bytes(resp.content)
```

---

## 方案一：PuLID v1.1 (SDXL) — 本地 GPU

> 部署在 Linux 服务器（RTX 4090 24GB），通过 HTTP API 调用。

### 技术栈

| 组件 | 说明 |
|------|------|
| **SDXL base 1.0** | 基座画图模型（2.6B 参数） |
| **PuLID v1.1** | 零样本身份保持插件，根据参考图保持角色长相 |
| **InsightFace** | 人脸检测，提取面部嵌入 |
| **FastAPI** | HTTP 服务框架 |

### 服务器信息

| 项目 | 值 |
|------|----|
| 服务器地址 | `172.31.102.189:8001` |
| GPU | RTX 4090 (24GB) |
| 峰值显存 | ~14.7 GB |
| 生成速度 | ~3-4 秒/张（25 步） |

### 工作模式

**有参考图（主模式）**：用户提供角色参考图 → 生成新场景插图时保持该角色的长相特征。适用于有主要角色、需要多张插图角色一致的情况。

**无参考图（回退）**：纯文本 prompt 生成，适用于风景、建筑等一次性场景。

### 接口

```
POST /generate
Content-Type: multipart/form-data
```

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `prompt` | string | 是 | — | 场景描述（英文效果最佳） |
| `ref_image` | file | 否 | — | 角色参考图（PNG/JPG），不传=无参考图模式 |
| `neg_prompt` | string | 否 | 见下方 | 负面提示词 |
| `seed` | int | 否 | -1（随机） | 随机种子 |
| `steps` | int | 否 | 25 | 推理步数（推荐 20-30） |
| `cfg` | float | 否 | 7.0 | 提示词引导强度 |
| `id_scale` | float | 否 | 0.8 | 身份保持强度（0.5-1.5，越高越像） |
| `num_zero` | int | 否 | 20 | 身份可编辑性（10-30，越低越像） |
| `height` | int | 否 | 1152 | 图片高度 |
| `width` | int | 否 | 896 | 图片宽度 |

默认负面词：
```
flaws in the eyes, flaws in the face, flaws, lowres, non-HDRi, low quality,
worst quality, artifacts noise, text, watermark, glitch, deformed, mutated,
ugly, disfigured, hands, low resolution, partially rendered objects,
deformed or partially rendered eyes, cross-eyed, blurry
```

### 调用示例

```python
import requests

BASE = 'http://172.31.102.189:8001'

# 无参考图
resp = requests.post(f'{BASE}/generate', data={
    'prompt': 'anime style, a quiet medieval village street at sunset, masterpiece',
    'steps': 20, 'cfg': 7.0,
})
with open('output.png', 'wb') as f:
    f.write(resp.content)

# 有参考图
with open('角色参考图.png', 'rb') as f:
    resp = requests.post(f'{BASE}/generate', data={
        'prompt': 'portrait of a cute wolf girl with brown hair and wolf ears, ...',
        'id_scale': 0.8, 'num_zero': 20, 'steps': 25,
    }, files={'ref_image': f})
with open('output.png', 'wb') as f:
    f.write(resp.content)
```

---

## 方案二：Agnes AI — 云端 API

> 免费云端生图 API，支持文生图、图生图、多图合成。
> 文档：https://wiki.agnes-ai.com/llms.txt

### 模型

| 模型 | 能力 | 特点 |
|------|------|------|
| `agnes-image-2.0-flash` | 文生图、图生图、多图合成 | 通用，速度快 |
| `agnes-image-2.1-flash` | 文生图、图生图 | 升级版，擅长复杂场景 |

### 前置条件

- **代理**：需要 `http://127.0.0.1:7890`
- **API Key**：存放在 `config/agnes_api_key`
- **请求超时**：建议 60s-180s

### 接口

```
POST https://apihub.agnes-ai.com/v1/images/generations
Authorization: Bearer <API_KEY>
Content-Type: application/json
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `model` | string | 是 | `agnes-image-2.0-flash` 或 `agnes-image-2.1-flash` |
| `prompt` | string | 是 | 文本描述 |
| `size` | string | 是 | 如 `1024x1024`、`1024x768` |
| `extra_body.image` | string[] | 图生图时必填 | 输入图片 URL 或 Data URI Base64 |
| `extra_body.response_format` | string | 否 | `url` 或 `b64_json` |

> ⚠️ `response_format` 必须放在 `extra_body` 内，不是顶层。

### 调用示例

```python
import base64, json, urllib.request

PROXY = 'http://127.0.0.1:7890'
API_BASE = 'https://apihub.agnes-ai.com/v1/images/generations'

# 读取 API Key
with open('config/agnes_api_key') as f:
    API_KEY = f.read().strip()

# 文生图
body = json.dumps({
    'model': 'agnes-image-2.0-flash',
    'prompt': 'A cute orange cat sitting on a windowsill, soft morning light',
    'size': '1024x1024',
    'extra_body': {'response_format': 'b64_json'}
}).encode('utf-8')

# 图生图（本地图片转 Data URI）
with open('参考图.png', 'rb') as f:
    b64 = base64.b64encode(f.read()).decode('utf-8')
data_uri = f'data:image/png;base64,{b64}'

body = json.dumps({
    'model': 'agnes-image-2.0-flash',
    'prompt': 'Transform this character into a medieval tavern scene, ...',
    'size': '1024x1024',
    'extra_body': {
        'image': [data_uri],
        'response_format': 'b64_json',
    }
}).encode('utf-8')

proxy_h = urllib.request.ProxyHandler({'http': PROXY, 'https': PROXY})
opener = urllib.request.build_opener(proxy_h)
req = urllib.request.Request(API_BASE, data=body, headers={
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {API_KEY}',
}, method='POST')

with opener.open(req, timeout=180) as resp:
    result = json.loads(resp.read().decode('utf-8'))

img_bytes = base64.b64decode(result['data'][0]['b64_json'])
with open('output.png', 'wb') as f:
    f.write(img_bytes)
```

---

## prompt 写作建议

| 原则 | 说明 |
|------|------|
| 质量词 | 末尾加 `masterpiece, high quality, best quality` |
| 风格词 | `anime style`, `illustration`, `digital painting` |
| 角色描述 | 发色、发型、服装、表情 |
| 场景描述 | 环境、光线、氛围 |

---

## 测试脚本

### PuLID 测试

```bash
# 完整测试
python scripts/test_illustration_api.py

# 仅无参考图
python scripts/test_illustration_api.py --no-ref-only

# 指定服务器
python scripts/test_illustration_api.py --host <IP> --port <端口>
```

### Agnes 测试

参考图在 `E:\projects\free-api\` 下的 `test_agnes_image.py`，或直接用上面示例代码调用。

---

## 注意事项

**PuLID：**
- 角色一致性有限：能保持发色、服色等大体特征，但五官细节可能有变化
- 动漫参考图检测较弱：InsightFace 训练于真实人脸，动漫图可能检测失败
- 首次启动需联网下载 EVA02-CLIP 模型，缓存后离线可用
- 不支持并发

**Agnes：**
- 需要代理 `http://127.0.0.1:7890`
- 图生图时本地图片需转为 Data URI Base64 格式
- 请求超时建议设 180s，图片生成可能需要较长时间
- API Key 在 `config/agnes_api_key`，已加入 `.gitignore`

**敏感文件：**
- `config/agnes_api_key` — Agnes API Key
- `config/sensenova_apikeys` — 预留，后续使用

以上文件已加入 `.gitignore`，不会提交到 GitHub。