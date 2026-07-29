import { execFile } from "node:child_process"
import { randomUUID } from "node:crypto"
import { mkdir, rm, writeFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { spawn, type ChildProcess } from "node:child_process"
import type {
  PipelineEvent,
  PipelineSnapshot,
  PipelineStage,
  PipelineStartRequest,
  StageRuntimeStatus,
} from "../preload/types"
import { PIPELINE_STAGES } from "../preload/types"
import {
  appendPipelineLog,
  applyStructuredEvent,
  createStageRuntime,
  parseStructuredLine,
} from "./pipeline-events"
import { validateTextFile } from "./text-file"
import { resolveProjectPaths, type ProjectPaths } from "./project-paths"

const RUNNING_STATES = new Set(["starting", "running", "stopping"])

export interface PipelineControllerOptions {
  projectRoot?: string
  configPath?: string
  stopGraceMilliseconds?: number
  publish?: (event: PipelineEvent) => void
}

function timestampForFilename(date = new Date()): string {
  return date.toISOString().replace(/[-:]/g, "").replace(/\.\d{3}Z$/, "Z")
}

export function buildPipelineInvocation(
  paths: ProjectPaths,
  request: PipelineStartRequest,
  stopFile: string,
  logPath: string,
): { executable: string; args: string[]; displayCommand: string } {
  const args = [
    "-u",
    paths.runFull,
    "--config",
    paths.config,
    "--novel",
    request.novelPath,
    "--labels",
    request.labelsPath,
    "--stream-tts",
    "--desktop-events",
    "--stop-file",
    stopFile,
    "--log",
    logPath,
  ]
  if (request.fromStage) args.push("--from-stage", request.fromStage)
  if (request.toStage) args.push("--to-stage", request.toStage)
  const displayCommand = [paths.python, ...args]
    .map((value) => (/\s/.test(value) ? JSON.stringify(value) : value))
    .join(" ")
  return { executable: paths.python, args, displayCommand }
}

export function validateStartRequest(request: unknown): asserts request is PipelineStartRequest {
  if (!request || typeof request !== "object") throw new TypeError("无效的流水线启动参数")
  const value = request as Record<string, unknown>
  if (typeof value.novelPath !== "string" || typeof value.labelsPath !== "string") {
    throw new TypeError("流水线启动参数缺少小说或角色标注路径")
  }
  for (const key of ["fromStage", "toStage"] as const) {
    const stage = value[key]
    if (stage !== undefined && !PIPELINE_STAGES.includes(stage as PipelineStage)) {
      throw new TypeError(`无效的流水线阶段：${String(stage)}`)
    }
  }
  if (value.fromStage && value.toStage) {
    if (
      PIPELINE_STAGES.indexOf(value.fromStage as PipelineStage)
      > PIPELINE_STAGES.indexOf(value.toStage as PipelineStage)
    ) {
      throw new TypeError("起始阶段不能晚于结束阶段")
    }
  }
}

export class PipelineController {
  private readonly options: PipelineControllerOptions
  private child: ChildProcess | null = null
  private stopTimer: NodeJS.Timeout | null = null
  private stopFile: string | null = null
  private stdoutBuffer = ""
  private stderrBuffer = ""
  private snapshot: PipelineSnapshot = {
    status: "idle",
    pid: null,
    command: "",
    projectRoot: "",
    outputDirectory: "",
    manifestPath: "",
    logPath: "",
    startedAt: null,
    finishedAt: null,
    exitCode: null,
    error: null,
    request: null,
    currentStage: null,
    currentStageIndex: null,
    stagePercent: 0,
    operation: "等待开始",
    stages: createStageRuntime(),
    logs: [],
  }

  constructor(options: PipelineControllerOptions = {}) {
    this.options = options
  }

  getState(): PipelineSnapshot {
    return {
      ...this.snapshot,
      request: this.snapshot.request ? { ...this.snapshot.request } : null,
      stages: this.snapshot.stages.map((stage) => ({ ...stage })),
      logs: this.snapshot.logs.map((entry) => ({ ...entry })),
    }
  }

  private publish(event: PipelineEvent): void {
    this.options.publish?.(event)
  }

  private publishState(): void {
    this.publish({ type: "state", state: this.getState() })
  }

  private update(patch: Partial<PipelineSnapshot>): void {
    this.snapshot = { ...this.snapshot, ...patch }
    this.publishState()
  }

  async start(request: PipelineStartRequest): Promise<PipelineSnapshot> {
    validateStartRequest(request)
    if (RUNNING_STATES.has(this.snapshot.status)) {
      throw new Error("流水线已经在运行，不能重复启动")
    }
    const [novel, labels] = await Promise.all([
      validateTextFile(request.novelPath),
      validateTextFile(request.labelsPath),
    ])
    if (!novel.ok) throw new Error(`小说原文无效：${novel.error}`)
    if (!labels.ok) throw new Error(`角色标注无效：${labels.error}`)

    const normalizedRequest: PipelineStartRequest = {
      ...request,
      novelPath: novel.file.path,
      labelsPath: labels.file.path,
    }
    const paths = resolveProjectPaths(this.options.projectRoot, this.options.configPath)
    const runId = `${process.pid}-${timestampForFilename()}-${randomUUID().slice(0, 8)}`
    const controlDirectory = join(tmpdir(), "novel-voice-cast-desktop")
    await mkdir(controlDirectory, { recursive: true })
    const stopFile = join(controlDirectory, `${runId}.stop`)
    await rm(stopFile, { force: true })
    const logPath = join(paths.logs, `desktop-${timestampForFilename()}.log`)
    const invocation = buildPipelineInvocation(paths, normalizedRequest, stopFile, logPath)

    this.stopFile = stopFile
    this.stdoutBuffer = ""
    this.stderrBuffer = ""
    this.update({
      status: "starting",
      pid: null,
      command: invocation.displayCommand,
      projectRoot: paths.root,
      outputDirectory: paths.output,
      manifestPath: join(paths.output, "run_full_manifest.json"),
      logPath,
      startedAt: new Date().toISOString(),
      finishedAt: null,
      exitCode: null,
      error: null,
      request: normalizedRequest,
      currentStage: null,
      currentStageIndex: null,
      stagePercent: 0,
      operation: "正在启动 Python 流水线",
      stages: createStageRuntime(normalizedRequest),
      logs: [],
    })

    try {
      const child = spawn(invocation.executable, invocation.args, {
        cwd: paths.root,
        windowsHide: true,
        detached: process.platform !== "win32",
        stdio: ["ignore", "pipe", "pipe"],
        env: {
          ...process.env,
          PYTHONUTF8: "1",
          PYTHONUNBUFFERED: "1",
        },
      })
      if (!child.stdout || !child.stderr) throw new Error("无法连接 Python 标准输出")
      this.child = child
      child.stdout.setEncoding("utf8")
      child.stderr.setEncoding("utf8")
      child.stdout.on("data", (chunk: string) => this.consumeOutput("stdout", chunk))
      child.stderr.on("data", (chunk: string) => this.consumeOutput("stderr", chunk))
      child.once("error", (error) => {
        this.update({ status: "failed", error: error.message, finishedAt: new Date().toISOString() })
      })
      child.once("close", (code) => this.onClose(code))
      this.update({ status: "running", pid: child.pid ?? null })
      return this.getState()
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)
      this.update({ status: "failed", error: message, finishedAt: new Date().toISOString() })
      throw error
    }
  }

  async stop(): Promise<PipelineSnapshot> {
    if (!this.child || !RUNNING_STATES.has(this.snapshot.status)) return this.getState()
    if (this.snapshot.status === "stopping") return this.getState()
    this.update({ status: "stopping" })
    if (this.stopFile) {
      await writeFile(this.stopFile, "stop\n", "utf8")
    }
    const pid = this.child.pid
    if (pid) {
      this.stopTimer = setTimeout(() => void this.forceStop(pid), this.options.stopGraceMilliseconds ?? 20_000)
      this.stopTimer.unref()
    }
    return this.getState()
  }

  private consumeOutput(stream: "stdout" | "stderr", chunk: string): void {
    const key = stream === "stdout" ? "stdoutBuffer" : "stderrBuffer"
    const combined = this[key] + chunk
    const lines = combined.split(/\r?\n/)
    this[key] = lines.pop() ?? ""
    for (const line of lines) {
      this.consumeLine(stream, line)
    }
  }

  private consumeLine(stream: "stdout" | "stderr", line: string): void {
    const timestamp = new Date().toISOString()
    this.publish({ type: "output", stream, line, timestamp })
    const event = parseStructuredLine(line)
    if (event) {
      this.snapshot = applyStructuredEvent(this.snapshot, event)
    } else if (line.trim()) {
      this.snapshot = appendPipelineLog(this.snapshot, {
        timestamp,
        level: stream === "stderr" ? "ERROR" : "INFO",
        message: line,
        stream,
        stage: this.snapshot.currentStage,
      })
    } else {
      return
    }
    this.publishState()
  }

  private flushOutput(): void {
    for (const stream of ["stdout", "stderr"] as const) {
      const key = stream === "stdout" ? "stdoutBuffer" : "stderrBuffer"
      if (this[key]) {
        this.consumeLine(stream, this[key])
        this[key] = ""
      }
    }
  }

  private onClose(code: number | null): void {
    this.flushOutput()
    if (this.stopTimer) clearTimeout(this.stopTimer)
    this.stopTimer = null
    const wasStopping = this.snapshot.status === "stopping"
    this.child = null
    const status = wasStopping ? "interrupted" : code === 0 ? "completed" : "failed"
    const terminalStageStatus: StageRuntimeStatus | null = (
      status === "interrupted" ? "interrupted" : status === "failed" ? "failed" : null
    )
    const stages = terminalStageStatus
      ? this.snapshot.stages.map((stage) => (
          stage.status === "running" ? { ...stage, status: terminalStageStatus } : stage
        ))
      : this.snapshot.stages
    this.update({
      status,
      stages,
      pid: null,
      exitCode: code,
      finishedAt: new Date().toISOString(),
      error: status === "failed" ? `Python 流水线退出码：${code ?? "unknown"}` : null,
    })
    if (this.stopFile) void rm(this.stopFile, { force: true })
    this.stopFile = null
  }

  private async forceStop(pid: number): Promise<void> {
    if (!this.child || this.child.pid !== pid) return
    if (process.platform === "win32") {
      await new Promise<void>((resolve) => {
        execFile("taskkill", ["/PID", String(pid), "/T", "/F"], { windowsHide: true }, () => resolve())
      })
      return
    }
    try {
      process.kill(-pid, "SIGKILL")
    } catch {
      this.child.kill("SIGKILL")
    }
  }
}
