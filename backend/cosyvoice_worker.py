"""CosyVoice 3 batch worker.

Runs inside the CosyVoice virtualenv (invoked by ``run_full.py``). Reads a
batch spec JSON from ``argv[1]`` containing the task list, loads the CosyVoice
3 model, and synthesizes each task via ``inference_instruct`` (zero-shot voice
cloning + natural-language performance control), with a resumable checkpoint
written back to the spec's ``results_path``.

Unlike the VoxCPM worker, the natural-language ``instruct_text`` is passed
through uncompressed — this is the whole point of using CosyVoice 3.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time


def _load_spec(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_checkpoint(results_path: str, results: dict) -> None:
    payload = {"results": results}
    temporary = results_path + f".{os.getpid()}.tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    for attempt in range(6):
        try:
            os.replace(temporary, results_path)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.05 * (2 ** attempt))


def _wav_meta(path: str, torchaudio) -> dict:
    info = torchaudio.info(path)
    frames = int(info.num_frames)
    sample_rate = int(info.sample_rate)
    if frames <= 0 or sample_rate <= 0:
        raise RuntimeError("generated WAV has invalid stream metadata")
    duration = frames / sample_rate
    if duration < 0.08:
        raise RuntimeError(f"generated WAV is implausibly short: {duration:.3f}s")
    size = os.path.getsize(path)
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "wav_sha256": digest.hexdigest(),
        "wav_size": size,
        "wav_frames": frames,
        "wav_sample_rate": sample_rate,
        "wav_channels": int(info.num_channels),
        "wav_duration_seconds": round(duration, 6),
    }


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: cosyvoice_worker.py <spec.json>", file=sys.stderr)
        return 2

    spec = _load_spec(argv[1])
    tasks = spec["tasks"]
    model_path = spec["model_path"]
    results_path = spec["results_path"]
    task_attempts = int(spec.get("task_attempts", 3))

    sys.path.insert(0, model_path)
    # CosyVoice 3 uses the CosyVoice2 CLI class (loads the 0.5B instruct model).
    from cosyvoice.cli.cosyvoice import CosyVoice2
    import torchaudio

    model = CosyVoice2(model_path, load_jit=False, load_trt=False, fp16=False)
    sample_rate = int(getattr(model, "sample_rate", 24000))

    results: dict = {}
    try:
        with open(results_path, "r", encoding="utf-8") as handle:
            prior = json.load(handle)
        if isinstance(prior.get("results"), dict):
            results = prior["results"]
    except (OSError, ValueError, TypeError):
        pass

    for position, task in enumerate(tasks, 1):
        key = str(task.get("task_key", task["index"]))
        if results.get(key, {}).get("status") == "ok":
            print(f"CosyVoice [{position}/{len(tasks)}] key={key} status=cached", flush=True)
            continue
        last_error = None
        temporary_wav = task["output_path"] + f".{os.getpid()}.tmp.wav"
        for attempt in range(1, task_attempts + 1):
            try:
                chunks = []
                for item in model.inference_instruct(
                    task["text"],
                    task["instruct_text"] or "",
                    task["reference_audio"],
                    stream=False,
                ):
                    chunks.append(item["tts_speech"])
                if not chunks:
                    raise RuntimeError("CosyVoice produced no speech")
                speech = chunks[0]
                if hasattr(speech, "shape") and len(speech.shape) == 2 and speech.shape[0] == 1:
                    speech = speech.squeeze(0)
                os.makedirs(os.path.dirname(task["output_path"]) or ".", exist_ok=True)
                torchaudio.save(temporary_wav, speech.unsqueeze(0), sample_rate)
                meta = _wav_meta(temporary_wav, torchaudio)
                os.replace(temporary_wav, task["output_path"])
                results[key] = {
                    "status": "ok",
                    "index": task["index"],
                    "fingerprint": task["fingerprint"],
                    "output_path": task["output_path"],
                    "instruct_text": task.get("instruct_text", ""),
                    "generation_attempts": attempt,
                    **meta,
                }
                break
            except Exception as exc:
                last_error = exc
                try:
                    os.unlink(temporary_wav)
                except FileNotFoundError:
                    pass
                if attempt < task_attempts:
                    time.sleep(min(5.0, 1.0 * attempt))
        if results.get(key, {}).get("status") != "ok":
            results[key] = {
                "status": "error",
                "index": task["index"],
                "fingerprint": task["fingerprint"],
                "output_path": task["output_path"],
                "generation_attempts": task_attempts,
                "error": str(last_error),
            }
        _save_checkpoint(results_path, results)
        print(f"CosyVoice [{position}/{len(tasks)}] key={key} status={results[key]['status']}", flush=True)

    if any(results.get(str(task.get("task_key", task["index"])), {}).get("status") != "ok" for task in tasks):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
