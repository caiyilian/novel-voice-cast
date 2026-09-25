"""验证「移除引号引用」能否消除泄漏（A/B/C 对照）。

假说
----
instruct 里用「」引用正文原词，与紧随的正文形成连续重复词组，
让自回归模型认为序列未结束 → 继续念 instruct。

三个变体
--------
A 原版      ：保持原样（对照组）
B 去引号    ：只删「」，保留词
C 删引用词  ：把「X」整体删除，只留动作描述

判定：ASR 覆盖率 < 1.35 视为干净（正常应接近 1.0）。

若 C 显著优于 A，说明「instruct 出现正文原词」是真原因；
若 B≈C，说明引号符号本身也有影响。
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, "E:/projects/cosyvoice")
sys.path.insert(0, "E:/projects/cosyvoice/third_party/Matcha-TTS")

import torch
import torchaudio
from cosyvoice.cli.cosyvoice import AutoModel

PROJECT_ROOT = Path("E:/projects/novel-voice-cast")
OUT = PROJECT_ROOT / "output" / "_quote_ab"
OUT.mkdir(parents=True, exist_ok=True)
REF = str(PROJECT_ROOT / "backend/data/presets/design_male_deep.wav")
REPEATS = 3


def norm(s: str) -> str:
    return re.sub(r"[^\u4e00-\u9fff0-9a-zA-Z]", "", str(s or ""))


def variant_b(c: str) -> str:
    """B：只去掉引号符号，保留词。"""
    return c.replace("「", "").replace("」", "")


def variant_c(c: str) -> str:
    """C：把「X」整体删除（连词一起去掉），只留动作描述。"""
    out = re.sub(r"「[^」]*」", "", c)
    out = re.sub(r"[，、；]{2,}", "，", out)
    out = re.sub(r"^[，、；]+", "", out)
    out = re.sub(r"[，、；]+$", "", out)
    return out.strip()


def main() -> int:
    report = json.loads((PROJECT_ROOT / "output" / "tts_quality_report.json").read_text(encoding="utf-8"))
    leak = []
    for r in report["records"]:
        if r["verdict"] != "LEAK":
            continue
        qs = re.findall(r"「([^」]+)」", r["control"])
        if qs and any(x in r["original"] for x in qs):
            leak.append(r)
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    leak = leak[:limit]
    print(f"选取含引号的 LEAK {len(leak)} 条\n", flush=True)

    model = AutoModel(
        model_dir="E:/projects/cosyvoice/pretrained_models/Fun-CosyVoice3-0.5B", fp16=True
    )
    print("CosyVoice 就绪\n", flush=True)

    manifest = []
    for r in leak:
        idx = r["index"]
        variants = {
            "A_orig": r["control"],
            "B_noquote": variant_b(r["control"]),
            "C_dropword": variant_c(r["control"]),
        }
        print(f"--- idx {idx} [{r['speaker']}] ---", flush=True)
        for tag, ctrl in variants.items():
            print(f"  [{tag}] {ctrl[:70]}", flush=True)
        for tag, ctrl in variants.items():
            instruct = f"You are a helpful assistant. {ctrl}<|endofprompt|>"
            for k in range(1, REPEATS + 1):
                try:
                    chunks = []
                    for item in model.inference_instruct2(r["original"], instruct, REF, stream=False):
                        chunks.append(item["tts_speech"])
                    speech = chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=-1)
                    p = OUT / f"{idx}_{tag}_r{k}.wav"
                    torchaudio.save(str(p), speech, model.sample_rate)
                    manifest.append({"index": idx, "speaker": r["speaker"], "tag": tag,
                                     "run": k, "path": str(p),
                                     "dur": round(speech.shape[-1] / model.sample_rate, 2),
                                     "original": r["original"]})
                except Exception as exc:  # noqa: BLE001
                    print(f"    {tag} r{k}: 失败 {type(exc).__name__}", flush=True)
        print(f"  idx {idx} 生成完毕", flush=True)

    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n生成 {len(manifest)} 个音频 -> {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
