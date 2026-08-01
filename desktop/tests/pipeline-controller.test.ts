import { existsSync } from "node:fs"
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join, resolve } from "node:path"
import { fileURLToPath } from "node:url"
import { afterEach, describe, expect, it } from "vitest"
import type { PipelineEvent, PipelineSnapshot } from "../src/preload/types"
import {
  buildPipelineInvocation,
  PipelineController,
  validateStartRequest,
} from "../src/main/pipeline-controller"
import { resolveProjectPaths } from "../src/main/project-paths"

const ROOT = fileURLToPath(new URL("../..", import.meta.url))
const temporaryDirectories: string[] = []

afterEach(async () => {
  await Promise.all(temporaryDirectories.splice(0).map((path) => rm(path, { recursive: true, force: true })))
})

describe("pipeline command", () => {
  it("uses an argument array with desktop events, stop file, and optional resume range", () => {
    const paths = {
      root: "E:/project",
      python: "E:/project/.venv/Scripts/python.exe",
      runFull: "E:/project/scripts/run_full.py",
      config: "E:/project/config/config.yaml",
      output: "E:/project/output",
      logs: "E:/project/logs",
    }
    const invocation = buildPipelineInvocation(
      paths,
      {
        novelPath: "E:/中文 小说/novel.txt",
        labelsPath: "E:/中文 小说/labels.txt",
        fromStage: "tts",
        toStage: "splice",
      },
      "C:/Temp/run.stop",
      "E:/project/logs/desktop.log",
    )

    expect(invocation.executable).toBe(paths.python)
    expect(invocation.args).toContain("--desktop-events")
    expect(invocation.args).toContain("--stream-tts")
    expect(invocation.args).toContain("--stop-file")
    expect(invocation.args.slice(-4)).toEqual(["--from-stage", "tts", "--to-stage", "splice"])
    expect(invocation.displayCommand).toContain('"E:/中文 小说/novel.txt"')
  })

  it("rejects malformed or reversed stage requests", () => {
    expect(() => validateStartRequest(null)).toThrow("无效")
    expect(() => validateStartRequest({ novelPath: "a", labelsPath: "b", fromStage: "bad" })).toThrow("阶段")
    expect(() => validateStartRequest({ novelPath: "a", labelsPath: "b", fromStage: "video", toStage: "parse" })).toThrow("不能晚于")
  })
})

const python = resolve(ROOT, process.platform === "win32" ? ".venv/Scripts/python.exe" : ".venv/bin/python")

describe.skipIf(!existsSync(python))("real parse-only pipeline controller", () => {
  it("directly spawns run_full.py and reaches completed with the tiny fixture", async () => {
    const directory = await mkdtemp(join(tmpdir(), "nvc-controller-"))
    temporaryDirectories.push(directory)
    const output = join(directory, "output")
    const configPath = join(directory, "config.json")
    await writeFile(
      configPath,
      JSON.stringify({
        novel: { text_path: "unused.txt", labels_path: "unused-labels.txt" },
        output: { dir: output, filename: "fixture", format: "mp3" },
        features: { emotion_label: true, performance_direction: true },
        bgm: { enabled: false },
      }),
      "utf8",
    )
    const events: PipelineEvent[] = []
    let finish: ((state: PipelineSnapshot) => void) | undefined
    const terminal = new Promise<PipelineSnapshot>((resolveTerminal) => {
      finish = resolveTerminal
    })
    const controller = new PipelineController({
      projectRoot: ROOT,
      configPath,
      publish: (event) => {
        events.push(event)
        if (
          event.type === "state"
          && ["completed", "failed", "interrupted"].includes(event.state.status)
        ) {
          finish?.(event.state)
        }
      },
    })

    const started = await controller.start({
      novelPath: resolve(ROOT, "desktop/fixtures/novel.txt"),
      labelsPath: resolve(ROOT, "desktop/fixtures/labels.txt"),
      fromStage: "parse",
      toStage: "parse",
    })
    expect(started.status).toBe("running")
    await expect(controller.start({
      novelPath: resolve(ROOT, "desktop/fixtures/novel.txt"),
      labelsPath: resolve(ROOT, "desktop/fixtures/labels.txt"),
    })).rejects.toThrow("已经在运行")
    let timeout: NodeJS.Timeout | undefined
    const result = await Promise.race([
      terminal,
      new Promise<never>((_resolve, reject) => {
        timeout = setTimeout(() => reject(new Error("timeout")), 30_000)
      }),
    ])
    if (timeout) clearTimeout(timeout)

    expect(result.status).toBe("completed")
    expect(result.exitCode).toBe(0)
    expect(result.command).toContain("scripts\\run_full.py")
    expect(result.stages).toHaveLength(13)
    expect(result.stages[0]).toMatchObject({ name: "parse", status: "complete", percent: 100 })
    expect(result.currentStage).toBe("parse")
    expect(result.stagePercent).toBe(100)
    expect(result.manifestStatus).toBe("valid")
    expect(result.manifestMessage).toContain("完整完成")
    expect(result.totalElapsedSeconds).toBeGreaterThanOrEqual(0)
    expect(result.outputDirectoryAvailable).toBe(true)
    await expect(controller.getOpenableOutputDirectory()).resolves.toBe(output)
    expect(result.logs.some((entry) => entry.message.includes("解析完成"))).toBe(true)
    const outputLines = events.filter((event) => event.type === "output").map((event) => event.line)
    expect(outputLines.some((line) => line.startsWith("[STAGE]"))).toBe(true)
    expect(resolveProjectPaths(ROOT, configPath)).toMatchObject({ python, output })
  }, 40_000)

  it("stops only its slow fixture process and resumes from its checkpoint", async () => {
    const root = await mkdtemp(join(tmpdir(), "nvc-stop-resume-"))
    temporaryDirectories.push(root)
    const output = join(root, "output")
    const scripts = join(root, "scripts")
    await Promise.all([mkdir(output), mkdir(scripts)])
    const novelPath = join(root, "novel.txt")
    const labelsPath = join(root, "labels.txt")
    const configPath = join(root, "config.yaml")
    await writeFile(novelPath, "第一章\n「你好。」\n", "utf8")
    await writeFile(labelsPath, "甲\n", "utf8")
    await writeFile(configPath, `output:\n  dir: ${JSON.stringify(output)}\n`, "utf8")
    await writeFile(join(scripts, "run_full.py"), `
import argparse, json, sys, time
from datetime import datetime, timezone
from pathlib import Path

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--stop-file")
args, _ = parser.parse_known_args()
root = Path(__file__).resolve().parents[1]
output = root / "output"
manifest_path = output / "run_full_manifest.json"
checkpoint = output / "slow-fixture.checkpoint"
started = datetime.now(timezone.utc).isoformat()

def write_manifest(status, stage_status=None):
    payload = {
        "version": 1,
        "root": str(root),
        "selected_stages": ["parse"],
        "run_started_at": started,
        "run_status": status,
        "stages": {},
    }
    if stage_status:
        payload["run_finished_at"] = datetime.now(timezone.utc).isoformat()
        payload["stages"]["parse"] = {
            "status": stage_status,
            "elapsed_seconds": 0.2,
            "artifacts": [],
        }
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

def stage(status):
    print("[STAGE] " + json.dumps({
        "version": 1,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "stage": "parse",
        "index": 1,
        "total": 13,
        "status": status,
        "elapsed_seconds": 0.2 if status != "running" else 0,
        "operation": "慢任务测试",
    }), flush=True)

write_manifest("running")
stage("running")
if checkpoint.exists():
    write_manifest("complete", "complete")
    stage("complete")
    raise SystemExit(0)
for _ in range(600):
    if Path(args.stop_file).exists():
        checkpoint.write_text("resume", encoding="utf-8")
        write_manifest("interrupted", "interrupted")
        stage("interrupted")
        raise SystemExit(130)
    time.sleep(0.05)
raise SystemExit(3)
`, "utf8")

    let resolveTerminal: ((state: PipelineSnapshot) => void) | undefined
    const nextTerminal = () => new Promise<PipelineSnapshot>((resolveState) => {
      resolveTerminal = resolveState
    })
    const controller = new PipelineController({
      projectRoot: root,
      configPath,
      pythonPath: python,
      stopGraceMilliseconds: 2_000,
      publish: (event) => {
        if (event.type === "state" && ["completed", "failed", "interrupted"].includes(event.state.status)) {
          resolveTerminal?.(event.state)
          resolveTerminal = undefined
        }
      },
    })
    const request = { novelPath, labelsPath, fromStage: "parse" as const, toStage: "parse" as const }

    const interruptedPromise = nextTerminal()
    await controller.start(request)
    await new Promise((resolveWait) => setTimeout(resolveWait, 200))
    expect((await controller.stop()).status).toBe("stopping")
    const interrupted = await interruptedPromise
    expect(interrupted.status).toBe("interrupted")
    expect(interrupted.manifestStatus).toBe("valid")
    expect(existsSync(join(output, "slow-fixture.checkpoint"))).toBe(true)

    const resumedPromise = nextTerminal()
    await controller.start(request)
    const resumed = await resumedPromise
    expect(resumed.status).toBe("completed")
    expect(resumed.manifestStatus).toBe("valid")
    expect(resumed.stages[0]).toMatchObject({ status: "complete", percent: 100 })
  }, 20_000)
})
