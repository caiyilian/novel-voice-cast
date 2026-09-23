"""分布式 CosyVoice TTS：本机 + 服务器多进程并行合成，支持断点续跑。

流程：
    1. 复用 run_full 生成全部 task（text + instruct + reference_audio）
    2. 回收各服务器 out/ 里已生成的 wav（tar 批量传回，按 mtime 过滤）
    3. 计算缺失 task，round-robin 分片到本机 + 各服务器 worker
    4. 本机多进程跑，服务器 nohup + CUDA_VISIBLE_DEVICES 跑
    5. 轮询完成后 tar 批量回传，汇总计时

用法：
    python scripts/distributed_tts.py                    # 回收 + 补缺
    python scripts/distributed_tts.py --harvest-only     # 只回收服务器成果
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SSH_KEY = str(Path.home() / ".ssh" / "id_rsa_lab")
SSH_OPT = ["-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=15"]

# 服务器：name -> (host, port, user, dir, [(gpu_id, worker_count), ...])
SERVERS = {
    "wyl":    ("172.31.102.189", 2222, "enine", "/sda/cyl/cosyvoice", [("0", 1), ("1", 1), ("2", 1), ("3", 1)]),
    "lab171": ("172.31.102.171", 2222, "enine", "/sda/cyl/cosyvoice", [("0", 1), ("1", 1)]),
    "lab237": ("172.31.102.237", 2222, "enine", "/mnt/sda/cyl/cosyvoice", [("0", 1), ("1", 1)]),
    "lab233": ("172.31.111.233", 60002, "ubuntu", "/public/cyl/cosyvoice", [("0", 2), ("1", 2), ("2", 1), ("3", 2), ("4", 1), ("5", 2)]),
}
LOCAL_WORKERS = 2

SEGMENTS = ROOT / "output" / "segments"

# 纯标点/省略号（无语音内容）的句子 CosyVoice 无法合成，用静音替代。
PUNCT = set("…。，！？、；：\u201c\u201d\u2018\u2019（）《》—~～ \t\n\r\"'")
SILENCE_SR = 24000


def is_speechless(text: str) -> bool:
    return sum(1 for c in text if c not in PUNCT) == 0


def write_silence(path: Path, text: str) -> None:
    import numpy as np
    import soundfile as sf

    seconds = min(2.5, max(0.8, 0.6 * max(1, len(text))))
    data = np.zeros(int(SILENCE_SR * seconds), dtype="float32")
    sf.write(str(path), data, SILENCE_SR, subtype="FLOAT")


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", **kw)


def ssh(host, port, user, command, timeout=None):
    return run(["ssh", *SSH_OPT, "-p", str(port), "-i", SSH_KEY, f"{user}@{host}", command], timeout=timeout)


def ssh_raw(host, port, user, command, timeout=900):
    """返回 bytes 的 ssh（用于 tar 流）。"""
    return subprocess.run(
        ["ssh", *SSH_OPT, "-p", str(port), "-i", SSH_KEY, f"{user}@{host}", command],
        capture_output=True, timeout=timeout,
    )


def scp_up(host, port, user, local_path, remote_path):
    return run(["scp", "-q", *SSH_OPT, "-P", str(port), "-i", SSH_KEY, str(local_path), f"{user}@{host}:{remote_path}"])


def harvest(host, port, user, remote_out: str, since: str, dest: Path) -> int:
    """把远端 out/ 里 mtime >= since 的 wav 用 tar 批量拉回本地。"""
    cmd = (
        f"cd {remote_out} 2>/dev/null && tar cf - "
        f"$(find . -maxdepth 1 -name '*.wav' -newermt '{since}' -printf '%f ') 2>/dev/null || true"
    )
    try:
        result = ssh_raw(host, port, user, cmd)
    except subprocess.TimeoutExpired:
        return 0
    if not result.stdout:
        return 0
    try:
        with tarfile.open(fileobj=io.BytesIO(result.stdout), mode="r:") as tf:
            members = [m for m in tf.getmembers() if m.isfile()]
            for m in members:
                m.name = Path(m.name).name  # 去掉 ./ 前缀
            tf.extractall(path=dest, members=members)
        return len(members)
    except (tarfile.TarError, OSError):
        return 0


def build_tasks(config):
    from scripts import run_full as rf

    perf = rf.read_json(rf.ROOT / "backend" / "data" / "performance_directions.json", {})
    perf_results = perf.get("results", {}) if isinstance(perf, dict) else perf
    if not perf_results:
        cp = rf.read_json(rf.ROOT / "backend" / "data" / "performance_directions.checkpoint.json", {})
        perf_results = cp.get("results", {})

    dialogues, _c, _n = rf.step_parse(config)
    gender_results = rf.read_json(rf.ROOT / "backend" / "data" / "gender_results.json", {})
    tasks = []
    for index, dialogue in enumerate(dialogues):
        speaker = rf.effective_speaker(dialogue.get("speaker", ""))
        gender = gender_results.get(speaker, {}).get("gender", "male")
        if gender not in {"male", "female"}:
            gender = "male"
        tasks.append(rf.make_tts_task(config, index, dialogue, gender, perf_results.get(str(index), {})))
    return tasks


def done_indices(since_ts: float) -> set:
    out = set()
    for f in SEGMENTS.glob("*.wav"):
        try:
            st = f.stat()
            if st.st_mtime >= since_ts and st.st_size > 0:
                out.add(int(f.stem))
        except (ValueError, OSError):
            continue
    return out


def make_spec(chunk, server_dir):
    return {
        "tasks": [
            {
                "task_key": str(t["index"]),
                "index": t["index"],
                "text": t["text"],
                "instruct_text": t.get("instruct_text", ""),
                "reference_audio": f"{server_dir}/refs/{Path(t['reference_audio']).name}",
                "output_path": f"{server_dir}/out/{t['index']:05d}.wav",
                "fingerprint": t["fingerprint"],
            }
            for t in chunk
        ],
        "repo_path": f"{server_dir}/src",
        "model_path": f"{server_dir}/models/Fun-CosyVoice3-0.5B",
        "results_path": f"{server_dir}/_results_gen.json",
        "task_attempts": 1,
        "fp16": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--harvest-only", action="store_true")
    parser.add_argument("--skip-harvest", action="store_true", help="跳过开头的成果回收")
    parser.add_argument("--since", default=None, help="成果起点 YYYY-MM-DD HH:MM（默认今天 00:00）")
    args = parser.parse_args()

    import yaml
    from scripts import run_full as rf

    config = yaml.safe_load(open(ROOT / "config" / "config.yaml", encoding="utf-8"))
    cutoff = (
        datetime.strptime(args.since, "%Y-%m-%d %H:%M").timestamp()
        if args.since
        else datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    )
    since_str = datetime.fromtimestamp(cutoff).strftime("%Y-%m-%d %H:%M")

    tasks = build_tasks(config)
    print(f"task 总数 {len(tasks)}", flush=True)
    refs = {Path(t["reference_audio"]).name: t["reference_audio"] for t in tasks}

    # ---------- 1. 回收服务器成果 ----------
    print(f"=== 回收服务器成果（>= {since_str}）===", flush=True)
    if args.skip_harvest:
        print("  (已跳过)", flush=True)
    else:
        for name, (host, port, user, dir_, _g) in SERVERS.items():
            n = harvest(host, port, user, f"{dir_}/out", since_str, SEGMENTS)
            print(f"  {name}: 回收 {n} 个", flush=True)

    done = done_indices(cutoff)
    print(f"本机有效 segments: {len(done)} / {len(tasks)}", flush=True)
    if args.harvest_only:
        return 0

    remaining = [t for t in tasks if t["index"] not in done]
    # 纯标点句直接写静音（CosyVoice 对无语音内容文本必然失败）
    speechless = [t for t in remaining if is_speechless(t["text"])]
    for t in speechless:
        write_silence(SEGMENTS / f"{t['index']:05d}.wav", t["text"])
    if speechless:
        print(f"静音替代 {len(speechless)} 句: {[t['index'] for t in speechless]}", flush=True)
    remaining = [t for t in remaining if not is_speechless(t["text"])]
    print(f"待生成: {len(remaining)}", flush=True)
    if not remaining:
        print("=== 已全部就绪 ===", flush=True)
        return 0

    # ---------- 2. 分片 ----------
    workers = [("local", "0", i) for i in range(LOCAL_WORKERS)]
    for name, (_h, _p, _u, _d, gpus) in SERVERS.items():
        for gpu, cnt in gpus:
            for i in range(cnt):
                workers.append((name, gpu, i))
    print(f"总 worker {len(workers)}", flush=True)

    shards = {wid: [] for wid in workers}
    for pos, task in enumerate(remaining):
        shards[workers[pos % len(workers)]].append(task)

    # ---------- 3. 分发 ----------
    worker_local = ROOT / "backend" / "cosyvoice_worker.py"
    print("=== 传参考音频 + worker ===", flush=True)
    for name, (host, port, user, dir_, _g) in SERVERS.items():
        ssh(host, port, user, f"mkdir -p {dir_}/refs {dir_}/out")
        scp_up(host, port, user, worker_local, f"{dir_}/worker.py")
        for basename, local in refs.items():
            scp_up(host, port, user, local, f"{dir_}/refs/{basename}")
        print(f"  {name} 就绪", flush=True)

    print("=== 开始合成，计时 ===", flush=True)
    t0 = time.time()
    marker = datetime.now().strftime("%Y-%m-%d %H:%M")

    def run_local_all():
        chunk = []
        for wid in [("local", "0", i) for i in range(LOCAL_WORKERS)]:
            chunk += shards[wid]
        if not chunk:
            return 0
        try:
            rf.run_cosyvoice_tasks(chunk, config)
        except Exception as exc:  # 个别句子失败不应中断整体
            print(f"  [local] 部分失败（已跳过）: {exc}", flush=True)
        return len(chunk)

    def run_server(name, srv):
        host, port, user, dir_, _g = srv
        server_workers = [w for w in workers if w[0] == name]
        launch, total = [], 0
        for wid in server_workers:
            chunk = shards[wid]
            if not chunk:
                continue
            total += len(chunk)
            spec = make_spec(chunk, dir_)
            spec["results_path"] = f"{dir_}/_results_{wid[1]}_{wid[2]}.json"
            local_spec = ROOT / f"_spec_{name}_{wid[1]}_{wid[2]}.json"
            local_spec.write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
            scp_up(host, port, user, local_spec, f"{dir_}/_spec_{name}_{wid[1]}_{wid[2]}.json")
            launch.append(
                f"CUDA_VISIBLE_DEVICES={wid[1]} nohup {dir_}/.venv/bin/python {dir_}/worker.py "
                f"{dir_}/_spec_{name}_{wid[1]}_{wid[2]}.json > /tmp/tts_{name}_{wid[1]}_{wid[2]}.log 2>&1 &"
            )
        if not launch:
            return 0
        ssh(host, port, user, f"cd {dir_} && " + " ".join(launch) + " echo STARTED")
        while True:
            done_n = 0
            for wid in server_workers:
                if not shards[wid]:
                    continue
                res = f"{dir_}/_results_{wid[1]}_{wid[2]}.json"
                r = ssh(host, port, user, f"test -f {res} && grep -c '\"status\": \"ok\"' {res} || echo 0")
                try:
                    done_n += int(r.stdout.strip())
                except ValueError:
                    pass
            if done_n >= total:
                break
            time.sleep(15)
        return harvest(host, port, user, f"{dir_}/out", marker, SEGMENTS)

    with ThreadPoolExecutor(max_workers=len(SERVERS) + 1) as pool:
        futures = [pool.submit(run_local_all)]
        for name, srv in SERVERS.items():
            futures.append(pool.submit(run_server, name, srv))
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as exc:
                print(f"  [worker] 失败（已跳过）: {exc}", flush=True)

    elapsed = time.time() - t0
    final = len(done_indices(cutoff))
    print(f"=== 完成，耗时 {elapsed:.1f}s = {elapsed/60:.2f} 分钟 ===", flush=True)
    print(f"=== segments 就绪 {final} / {len(tasks)} ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
