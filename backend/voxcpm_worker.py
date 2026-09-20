"""VoxCPM batch worker.

Runs inside the VoxCPM virtualenv (invoked by ``run_full.py``).  Reads a batch
spec JSON from ``argv[1]`` containing the task list and all generation options,
synthesizes each task with per-task quality checks and an atomic resumable
checkpoint, then writes results back to the spec's ``results_path``.

This replaces the previous ``create_voxcpm_script`` f-string that generated
this same logic as a throwaway script on every run.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time


def _load_spec(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inspect_wav(path: str, sf) -> dict:
    info = sf.info(path)
    if info.frames <= 0 or info.samplerate <= 0 or info.channels <= 0:
        raise RuntimeError("generated WAV has invalid stream metadata")
    duration = info.frames / info.samplerate
    if duration < 0.08:
        raise RuntimeError(f"generated WAV is implausibly short: {duration:.3f}s")
    return {
        "wav_sha256": _file_sha256(path),
        "wav_size": os.path.getsize(path),
        "wav_frames": int(info.frames),
        "wav_sample_rate": int(info.samplerate),
        "wav_channels": int(info.channels),
        "wav_duration_seconds": round(duration, 6),
    }


def _valid_result(result, task, sf) -> bool:
    if not isinstance(result, dict):
        return False
    key = str(task.get("task_key", task["index"]))
    if (
        result.get("status") != "ok"
        or result.get("task_key") != key
        or result.get("index") != task["index"]
        or result.get("chunk_index", 0) != task.get("chunk_index", 0)
        or result.get("fingerprint") != task["fingerprint"]
        or result.get("output_path") != task["output_path"]
    ):
        return False
    try:
        actual = _inspect_wav(task["output_path"], sf)
    except Exception:
        return False
    return all(result.get(name) == value for name, value in actual.items())


def _save_checkpoint(results_path: str, checkpoint_version: str, generation_signature: str, source_hash: str, tasks: list, results: dict) -> None:
    payload = {
        "version": checkpoint_version,
        "generation_signature": generation_signature,
        "source_hash": source_hash,
        "expected": len(tasks),
        "completed": sum(
            1
            for task in tasks
            if results.get(str(task.get("task_key", task["index"])), {}).get("status") == "ok"
        ),
        "results": results,
    }
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


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: voxcpm_worker.py <spec.json>", file=sys.stderr)
        return 2

    spec = _load_spec(argv[1])
    tasks = spec["tasks"]
    model_path = spec["model_path"]
    module_path = spec["module_path"]
    results_path = spec["results_path"]
    checkpoint_version = spec["checkpoint_version"]
    source_hash = spec["source_hash"]
    generation_signature = spec["generation_signature"]
    cfg_value = spec["cfg_value"]
    inference_timesteps = spec["inference_timesteps"]
    normalize = spec["normalize"]
    reuse_reference_cache = spec["reuse_reference_cache"]
    retry_badcase = spec["retry_badcase"]
    retry_badcase_max_times = spec["retry_badcase_max_times"]
    retry_badcase_ratio_threshold = spec["retry_badcase_ratio_threshold"]
    max_len = spec["max_len"]
    task_attempts = spec["task_attempts"]

    sys.path.insert(0, module_path)
    try:
        from models.voxcpm import VoxCPM
    except ImportError:
        from voxcpm import VoxCPM
    import soundfile as sf

    results: dict = {}
    try:
        with open(results_path, "r", encoding="utf-8") as handle:
            prior = json.load(handle)
        if (
            prior.get("version") == checkpoint_version
            and prior.get("generation_signature") == generation_signature
            and isinstance(prior.get("results"), dict)
        ):
            results = prior["results"]
    except (OSError, ValueError, TypeError):
        pass

    try:
        model = VoxCPM.from_pretrained(model_path, load_denoiser=False)
    except Exception as exc:
        results["_model"] = {"status": "error", "error": f"model load failed: {exc}"}
        _save_checkpoint(results_path, checkpoint_version, generation_signature, source_hash, tasks, results)
        raise

    prompt_caches = {}
    supports_prompt_cache = bool(
        reuse_reference_cache
        and hasattr(model, "tts_model")
        and hasattr(model.tts_model, "build_prompt_cache")
        and hasattr(model.tts_model, "generate_with_prompt_cache")
    )

    def normalize_text(text: str) -> str:
        text = re.sub(r"\s+", " ", str(text).replace("\n", " ")).strip()
        if normalize:
            if getattr(model, "text_normalizer", None) is None:
                from voxcpm.utils.text_normalize import TextNormalizer
                model.text_normalizer = TextNormalizer()
            text = model.text_normalizer.normalize(text)
        return text

    def generate(task: dict, generation_attempt: int):
        variants = task.get("control_variants") or [task.get("style_control", "")]
        control = str(variants[min(generation_attempt - 1, len(variants) - 1)]).strip()
        text = str(task["text"]).replace("\n", " ")
        final_text = f"({control}){text}" if control else text
        if not supports_prompt_cache:
            wav = model.generate(
                text=final_text,
                reference_wav_path=task["reference_audio"],
                cfg_value=cfg_value,
                inference_timesteps=inference_timesteps,
                normalize=normalize,
                retry_badcase=retry_badcase,
                retry_badcase_max_times=retry_badcase_max_times,
            )
            return wav, {"prompt_cache": False, "used_control": control}
        reference = task["reference_audio"]
        if reference not in prompt_caches:
            prompt_caches[reference] = model.tts_model.build_prompt_cache(reference_wav_path=reference)
        wav, target_tokens, audio_features = model.tts_model.generate_with_prompt_cache(
            target_text=normalize_text(final_text),
            prompt_cache=prompt_caches[reference],
            cfg_value=cfg_value,
            inference_timesteps=inference_timesteps,
            retry_badcase=retry_badcase,
            retry_badcase_max_times=retry_badcase_max_times,
            retry_badcase_ratio_threshold=retry_badcase_ratio_threshold,
            max_len=max_len,
        )
        token_count = max(1, int(target_tokens.numel()))
        feature_count = int(audio_features.shape[0])
        audio_text_ratio = feature_count / token_count
        if retry_badcase and audio_text_ratio >= retry_badcase_ratio_threshold:
            raise RuntimeError(f"badcase remained after retries: audio_text_ratio={audio_text_ratio:.3f}")
        return wav.squeeze(0).cpu().numpy(), {
            "prompt_cache": True,
            "target_token_count": token_count,
            "audio_feature_count": feature_count,
            "audio_text_ratio": round(audio_text_ratio, 6),
            "used_control": control,
        }

    _save_checkpoint(results_path, checkpoint_version, generation_signature, source_hash, tasks, results)
    for position, task in enumerate(tasks, 1):
        key = str(task.get("task_key", task["index"]))
        if _valid_result(results.get(key), task, sf):
            print(f"VoxCPM [{position}/{len(tasks)}] key={key} status=cached", flush=True)
            continue
        last_error = None
        for generation_attempt in range(1, task_attempts + 1):
            temporary_wav = task["output_path"] + f".{os.getpid()}.tmp.wav"
            try:
                wav, diagnostics = generate(task, generation_attempt)
                os.makedirs(os.path.dirname(task["output_path"]) or ".", exist_ok=True)
                sf.write(temporary_wav, wav, model.tts_model.sample_rate)
                wave_meta = _inspect_wav(temporary_wav, sf)
                duration = wave_meta["wav_duration_seconds"]
                minimum = float(task.get("min_duration_seconds", 0.0))
                maximum = float(task.get("max_duration_seconds", float("inf")))
                if duration < minimum:
                    raise RuntimeError(
                        "audio is anomalously fast; retrying a fresh VoxCPM take: "
                        f"{duration:.3f}s < {minimum:.3f}s"
                    )
                if duration > maximum:
                    raise RuntimeError(
                        "audio is too slow or leaked control text: "
                        f"{duration:.3f}s > {maximum:.3f}s"
                    )
                os.replace(temporary_wav, task["output_path"])
                results[key] = {
                    "task_key": key,
                    "index": task["index"],
                    "chunk_index": task.get("chunk_index", 0),
                    "status": "ok",
                    "fingerprint": task["fingerprint"],
                    "output_path": task["output_path"],
                    "generation_attempts": generation_attempt,
                    **diagnostics,
                    **wave_meta,
                }
                break
            except Exception as exc:
                last_error = exc
                try:
                    os.unlink(temporary_wav)
                except FileNotFoundError:
                    pass
                if generation_attempt < task_attempts:
                    time.sleep(min(10.0, 1.5 * generation_attempt))
        if results.get(key, {}).get("status") != "ok":
            results[key] = {
                "task_key": key,
                "index": task["index"],
                "chunk_index": task.get("chunk_index", 0),
                "status": "error",
                "fingerprint": task["fingerprint"],
                "output_path": task["output_path"],
                "generation_attempts": task_attempts,
                "error": str(last_error),
            }
        _save_checkpoint(results_path, checkpoint_version, generation_signature, source_hash, tasks, results)
        print(f"VoxCPM [{position}/{len(tasks)}] key={key} status={results[key]['status']}", flush=True)

    if any(
        results.get(str(task.get("task_key", task["index"])), {}).get("status") != "ok"
        for task in tasks
    ):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
