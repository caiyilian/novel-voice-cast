import { access, readFile, readdir, stat } from "node:fs/promises"
import { isAbsolute, resolve } from "node:path"
import {
  PIPELINE_STAGES,
  type ManifestStatus,
  type PipelineArtifact,
  type PipelineStage,
  type PipelineStartRequest,
  type StageRuntimeStatus,
} from "../preload/types"

export interface ManifestStageResult {
  name: PipelineStage
  status: Extract<StageRuntimeStatus, "complete" | "skipped" | "failed" | "interrupted">
  elapsedSeconds: number
  artifacts: string[]
}

export interface RunResult {
  manifestStatus: ManifestStatus
  manifestMessage: string
  runStatus: string | null
  runError: string | null
  totalElapsedSeconds: number
  stages: ManifestStageResult[]
  artifacts: PipelineArtifact[]
}

export interface ReadRunResultOptions {
  manifestPath: string
  outputDirectory: string
  projectRoot: string
  expectedStartedAt: string
  request: PipelineStartRequest
  now?: Date
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value)
}

function expectedStages(request: PipelineStartRequest): PipelineStage[] {
  const start = request.fromStage ? PIPELINE_STAGES.indexOf(request.fromStage) : 0
  const end = request.toStage ? PIPELINE_STAGES.indexOf(request.toStage) : PIPELINE_STAGES.length - 1
  return PIPELINE_STAGES.slice(start, end + 1)
}

function samePath(left: string, right: string): boolean {
  const normalizedLeft = resolve(left).replaceAll("/", "\\").toLowerCase()
  const normalizedRight = resolve(right).replaceAll("/", "\\").toLowerCase()
  return normalizedLeft === normalizedRight
}

function emptyResult(
  status: ManifestStatus,
  message: string,
  elapsedSeconds: number,
): RunResult {
  return {
    manifestStatus: status,
    manifestMessage: message,
    runStatus: null,
    runError: null,
    totalElapsedSeconds: elapsedSeconds,
    stages: [],
    artifacts: [],
  }
}

async function pathExists(path: string): Promise<boolean> {
  try {
    await access(path)
    return true
  } catch {
    return false
  }
}

async function collectArtifacts(
  projectRoot: string,
  outputDirectory: string,
  stageResults: ManifestStageResult[],
): Promise<PipelineArtifact[]> {
  const values = stageResults.flatMap((stage) => stage.artifacts).map((path) => (
    isAbsolute(path) ? resolve(path) : resolve(projectRoot, path)
  ))
  values.push(resolve(outputDirectory, "full_volume.mp3"))
  values.push(resolve(outputDirectory, "full_volume_bgm.mp3"))
  try {
    const entries = await readdir(outputDirectory, { withFileTypes: true })
    values.push(...entries
      .filter((entry) => entry.isFile() && entry.name.toLowerCase().endsWith(".mp4"))
      .map((entry) => resolve(outputDirectory, entry.name)))
  } catch {
    // A missing output directory is reported by the artifact existence flags.
  }
  const unique = [...new Map(values.map((path) => [path.toLowerCase(), path])).values()]
  return Promise.all(unique.map(async (path) => ({ path, exists: await pathExists(path) })))
}

export async function readRunResult(options: ReadRunResultOptions): Promise<RunResult> {
  const expectedStarted = Date.parse(options.expectedStartedAt)
  const now = options.now?.getTime() ?? Date.now()
  const fallbackElapsed = Number.isFinite(expectedStarted)
    ? Math.max(0, (now - expectedStarted) / 1000)
    : 0
  let text: string
  try {
    text = await readFile(options.manifestPath, "utf8")
  } catch (error) {
    const code = isRecord(error) ? error.code : undefined
    return emptyResult(
      code === "ENOENT" ? "missing" : "invalid",
      code === "ENOENT" ? "本次运行没有生成 manifest" : `无法读取 manifest：${String(error)}`,
      fallbackElapsed,
    )
  }

  let raw: unknown
  try {
    raw = JSON.parse(text)
  } catch {
    return emptyResult("invalid", "manifest 不是有效的 JSON", fallbackElapsed)
  }
  if (
    !isRecord(raw)
    || raw.version !== 1
    || typeof raw.root !== "string"
    || !Array.isArray(raw.selected_stages)
    || typeof raw.run_started_at !== "string"
    || typeof raw.run_status !== "string"
    || !isRecord(raw.stages)
  ) {
    return emptyResult("invalid", "manifest 缺少必需字段或版本不受支持", fallbackElapsed)
  }
  if (!["running", "complete", "failed", "interrupted"].includes(raw.run_status)) {
    return emptyResult("invalid", `manifest 包含未知运行状态：${raw.run_status}`, fallbackElapsed)
  }

  const actualStarted = Date.parse(raw.run_started_at)
  const selected = raw.selected_stages.filter((stage): stage is PipelineStage => (
    typeof stage === "string" && PIPELINE_STAGES.includes(stage as PipelineStage)
  ))
  const expected = expectedStages(options.request)
  const selectionMatches = selected.length === raw.selected_stages.length
    && selected.length === expected.length
    && selected.every((stage, index) => stage === expected[index])
  if (
    !samePath(raw.root, options.projectRoot)
    || !Number.isFinite(actualStarted)
    || actualStarted < expectedStarted - 1000
    || !selectionMatches
  ) {
    return emptyResult("stale", "manifest 不属于本次运行（项目、开始时间或阶段范围不匹配）", fallbackElapsed)
  }

  const terminal = ["complete", "failed", "interrupted"].includes(raw.run_status)
  const finishedAt = typeof raw.run_finished_at === "string" ? Date.parse(raw.run_finished_at) : Number.NaN
  if (terminal && (!Number.isFinite(finishedAt) || finishedAt < actualStarted)) {
    return emptyResult("invalid", "终态 manifest 缺少有效的结束时间", fallbackElapsed)
  }

  const allowedStageStatuses = ["complete", "skipped", "failed", "interrupted"] as const
  const stageResults: ManifestStageResult[] = []
  for (const stage of expected) {
    const value = raw.stages[stage]
    if (!isRecord(value)) continue
    if (
      typeof value.status !== "string"
      || !allowedStageStatuses.includes(value.status as (typeof allowedStageStatuses)[number])
      || typeof value.elapsed_seconds !== "number"
      || !Number.isFinite(value.elapsed_seconds)
    ) continue
    const artifacts = Array.isArray(value.artifacts)
      ? value.artifacts.filter((path): path is string => typeof path === "string" && path.length > 0)
      : []
    stageResults.push({
      name: stage,
      status: value.status as ManifestStageResult["status"],
      elapsedSeconds: Math.max(0, value.elapsed_seconds),
      artifacts,
    })
  }
  if (
    raw.run_status === "complete"
    && (
      stageResults.length !== expected.length
      || stageResults.some((stage) => stage.status !== "complete" && stage.status !== "skipped")
    )
  ) {
    return emptyResult("invalid", "complete manifest 未覆盖全部所选阶段的完成或跳过状态", fallbackElapsed)
  }
  const totalElapsedSeconds = terminal ? Math.max(0, (finishedAt - actualStarted) / 1000) : fallbackElapsed
  return {
    manifestStatus: "valid",
    manifestMessage: raw.run_status === "complete" ? "manifest 已确认本次流水线完整完成" : `manifest 状态：${raw.run_status}`,
    runStatus: raw.run_status,
    runError: typeof raw.run_error === "string" ? raw.run_error : null,
    totalElapsedSeconds,
    stages: stageResults,
    artifacts: await collectArtifacts(options.projectRoot, options.outputDirectory, stageResults),
  }
}

export async function requireOpenableDirectory(path: string): Promise<string> {
  if (!path) throw new Error("尚无本次流水线的输出目录")
  try {
    if (!(await stat(path)).isDirectory()) throw new Error("不是目录")
  } catch {
    throw new Error(`输出目录不存在：${path}`)
  }
  return resolve(path)
}
