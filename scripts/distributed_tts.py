"""分布式 CosyVoice TTS 基准测试：本机 + 服务器并行合成，测算耗时。

复用 run_full 生成 task，把参考音频 + worker 脚本 scp 到各服务器，
按 worker 数分片后本机多进程 + 服务器 ssh 并行合成，回传 wav 汇总计时。
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SSH_KEY = str(Path.home() / ".ssh" / "id_rsa_lab")
SSH_OPT = ["-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=15"]

# name -> (host, port, user, cosyvoice根目录, worker数)
SERVERS = {
    "wyl":    ("172.31.102.189", 2222, "enine", "/sda/cyl/cosyvoice", 6),
    "lab171": ("172.31.102.171", 2222, "enine", "/sda/cyl/cosyvoice", 5),
    "lab237": ("172.31.102.237", 2222, "enine", "/mnt/sda/cyl/cosyvoice", 3),
    "lab233": ("172.31.111.233", 60002, "ubuntu", "/public/cyl/cosyvoice", 20),
}

LOCAL_WORKERS = 3


def run(cmd: list[str], **kw):
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", **kw)


def scp(host, port, user, src, dst):
    return run(["scp", "-q", *SSH_OPT, "-P", str(port), "-i", SSH_KEY, src, f"{user}@{host}:{dst}"])


def ssh(host, port, user, command, timeout=None):
    return run(["ssh", *SSH_OPT, "-p", str(port), "-i", SSH_KEY, f"{user}@{host}", command], timeout=timeout)


def load_performance_results():
    from scripts import run_full as rf
    perf = rf.read_json(rf.ROOT / "backend" / "data" / "performance_directions.json", {})
    results = perf.get("results", {}) if isinstance(perf, dict) else perf
    if not results:
        cp = rf.read_json(rf.ROOT / "backend" / "data" / "performance_directions.checkpoint.json", {})
        results = cp.get("results", {})
    return results


def build_tasks(config, perf_results):
    from scripts import run_full as rf
    dialogues, _chars, _novel = rf.step_parse(config)
    gender_results = rf.read_json(rf.ROOT / "backend" / "data" / "gender_results.json", {})
    tasks = []
    for index, dialogue in enumerate(dialogues):
        speaker = rf.effective_speaker(dialogue.get("speaker", ""))
        gender = gender_results.get(speaker, {}).get("gender", "male")
        if gender not in {"male", "female"}:
            gender = "male"
        task = rf.make_tts_task(config, index, dialogue, gender, perf_results.get(str(index), {}))
        tasks.append(task)
    return tasks


def main() -> int:
    import yaml
    from scripts import run_full as rf

    config = yaml.safe_load(open(ROOT / "config" / "config.yaml", encoding="utf-8"))

    perf_results = load_performance_results()
    print(f"performance 结果: {len(perf_results)} 句", flush=True)
    if len(perf_results) < 3000:
        print("⚠️ performance 还没跑完，先只测已有部分", flush=True)

    tasks = build_tasks(config, perf_results)
    print(f"task 总数: {len(tasks)}", flush=True)

    # 参考音频集合
    refs = {Path(t["reference_audio"]).name: t["reference_audio"] for t in tasks}
    print(f"参考音频 {len(refs)} 个", flush=True)

    # worker 脚本路径（本地）
    worker_local = ROOT / "backend" / "cosyvoice_worker.py"

    # 1) 传参考音频 + worker 脚本到各服务器
    print("=== 传参考音频 + worker 到服务器 ===", flush=True)
    for name, (host, port, user, dir_, _w) in SERVERS.items():
        ssh(host, port, user, f"mkdir -p {dir_}/refs {dir_}/out")
        scp(host, port, user, str(worker_local), f"{dir_}/worker.py")
        for basename, local in refs.items():
            scp(host, port, user, local, f"{dir_}/refs/{basename}")
        print(f"  {name} 就绪", flush=True)

    # 2) 分片
    worker_ids = [("local", i) for i in range(LOCAL_WORKERS)]
    for name, (_h, _p, _u, _d, w) in SERVERS.items():
        worker_ids += [(name, i) for i in range(w)]
    shards = {wid: [] for wid in worker_ids}
    for pos, task in enumerate(tasks):
        wid = worker_ids[pos % len(worker_ids)]
        shards[wid].append(task)

    # 3) 计时开始
    print(f"=== 开始合成，共 {len(worker_ids)} worker ===", flush=True)
    t0 = time.time()

    futures = []
    with ThreadPoolExecutor(max_workers=len(SERVERS) + 1) as pool:
        # 本机
        def run_local():
            local_tasks = []
            for wid in [("local", i) for i in range(LOCAL_WORKERS)]:
                local_tasks += shards.get(wid, [])
            if not local_tasks:
                return 0
            from scripts import run_full as rf
            rf.run_cosyvoice_tasks(local_tasks, config)
            return len(local_tasks)

        futures.append(pool.submit(run_local))

        # 服务器
        def run_server(name, srv):
            host, port, user, dir_, _w = srv
            chunk = []
            for wid in [(name, i) for i in range(_w)]:
                chunk += shards.get(wid, [])
            if not chunk:
                return 0
            remote_tasks = [
                {
                    "task_key": str(t["index"]),
                    "index": t["index"],
                    "text": t["text"],
                    "instruct_text": t.get("instruct_text", ""),
                    "reference_audio": f"{dir_}/refs/{Path(t['reference_audio']).name}",
                    "output_path": f"{dir_}/out/{t['index']:05d}.wav",
                    "fingerprint": t["fingerprint"],
                }
                for t in chunk
            ]
            spec = {
                "tasks": remote_tasks,
                "repo_path": f"{dir_}/src",
                "model_path": f"{dir_}/models/Fun-CosyVoice3-0.5B",
                "results_path": f"{dir_}/_results_{name}.json",
                "task_attempts": 2,
                "fp16": True,
            }
            local_spec = ROOT / f"_spec_{name}.json"
            local_spec.write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
            scp(host, port, user, str(local_spec), f"{dir_}/_spec_{name}.json")
            r = ssh(host, port, user,
                    f"cd {dir_} && .venv/bin/python worker.py _spec_{name}.json > /tmp/tts_{name}.log 2>&1; echo EXIT:$?",
                    timeout=3600)
            # 回传 wav
            for t in chunk:
                idx = t["index"]
                scp(host, port, user, f"{dir_}/out/{idx:05d}.wav", str(ROOT / "output" / "segments" / f"{idx:05d}.wav"))
            return len(chunk)

        for name, srv in SERVERS.items():
            futures.append(pool.submit(run_server, name, srv))

        done = 0
        for fut in as_completed(futures):
            done += fut.result() or 0

    elapsed = time.time() - t0
    print(f"=== 合成完成 {done} 句，耗时 {elapsed:.1f} 秒 = {elapsed/60:.2f} 分钟 ===", flush=True)
    print(f"=== 平均 {elapsed/done:.2f} 秒/句，吞吐 {done/elapsed:.1f} 句/秒 ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
