import {
  PIPELINE_STAGES,
  type PipelineLogEntry,
  type PipelineSnapshot,
  type PipelineStage,
  type PipelineStartRequest,
  type StageRuntime,
  type StageRuntimeStatus,
} from "../preload/types"

export const MAX_PIPELINE_LOGS = 800

interface EventBase {
  version: 1
  timestamp: string
}

export interface StageEvent extends EventBase {
  kind: "stage"
  stage: PipelineStage
  index: number
  total: number
  status: Extract<StageRuntimeStatus, "running" | "complete" | "skipped" | "failed" | "interrupted">
  elapsed_seconds: number
  operation: string
  command?: string
  error?: string
}

export interface ProgressEvent extends EventBase {
  kind: "progress"
  stage: PipelineStage
  current: number
  total: number
  percent: number
  status: string
  operation: string
  command?: string
}

export interface LogEvent extends EventBase {
  kind: "log"
  level: string
  message: string
  stage?: PipelineStage
}

export type StructuredPipelineEvent = StageEvent | ProgressEvent | LogEvent

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value)
}

function isStage(value: unknown): value is PipelineStage {
  return typeof value === "string" && PIPELINE_STAGES.includes(value as PipelineStage)
}

function number(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null
}

export function parseStructuredLine(line: string): StructuredPipelineEvent | null {
  const markers = ["[STAGE] ", "[PROGRESS] ", "[LOG] "] as const
  const marker = markers.find((candidate) => line.startsWith(candidate))
  if (!marker) return null
  let raw: unknown
  try {
    raw = JSON.parse(line.slice(marker.length))
  } catch {
    return null
  }
  if (!isRecord(raw) || raw.version !== 1 || typeof raw.timestamp !== "string") return null

  if (marker === "[STAGE] ") {
    const statuses = ["running", "complete", "skipped", "failed", "interrupted"]
    if (
      !isStage(raw.stage)
      || number(raw.index) === null
      || number(raw.total) === null
      || typeof raw.status !== "string"
      || !statuses.includes(raw.status)
      || number(raw.elapsed_seconds) === null
      || typeof raw.operation !== "string"
    ) return null
    return { ...(raw as Omit<StageEvent, "kind">), kind: "stage" }
  }

  if (marker === "[PROGRESS] ") {
    if (
      !isStage(raw.stage)
      || number(raw.current) === null
      || number(raw.total) === null
      || number(raw.percent) === null
      || typeof raw.status !== "string"
      || typeof raw.operation !== "string"
    ) return null
    return { ...(raw as Omit<ProgressEvent, "kind">), kind: "progress" }
  }

  if (typeof raw.level !== "string" || typeof raw.message !== "string") return null
  if (raw.stage !== undefined && !isStage(raw.stage)) return null
  return { ...(raw as Omit<LogEvent, "kind">), kind: "log" }
}

export function createStageRuntime(request: PipelineStartRequest | null = null): StageRuntime[] {
  const start = request?.fromStage ? PIPELINE_STAGES.indexOf(request.fromStage) : 0
  const end = request?.toStage ? PIPELINE_STAGES.indexOf(request.toStage) : PIPELINE_STAGES.length - 1
  return PIPELINE_STAGES.map((name, offset) => ({
    name,
    index: offset + 1,
    status: request && (offset < start || offset > end) ? "not-selected" : "pending",
    percent: 0,
    operation: "等待开始",
    elapsedSeconds: 0,
  }))
}

export function appendPipelineLog(
  snapshot: PipelineSnapshot,
  entry: Omit<PipelineLogEntry, "id">,
  limit = MAX_PIPELINE_LOGS,
): PipelineSnapshot {
  const nextId = (snapshot.logs.at(-1)?.id ?? 0) + 1
  const logs = [...snapshot.logs, { ...entry, id: nextId }]
  return { ...snapshot, logs: logs.slice(-Math.max(1, limit)) }
}

export function applyStructuredEvent(
  snapshot: PipelineSnapshot,
  event: StructuredPipelineEvent,
): PipelineSnapshot {
  if (event.kind === "log") {
    return appendPipelineLog(snapshot, {
      timestamp: event.timestamp,
      level: event.level.toUpperCase(),
      message: event.message,
      stream: "structured",
      stage: event.stage ?? snapshot.currentStage,
    })
  }

  const stageOffset = PIPELINE_STAGES.indexOf(event.stage)
  const stages = snapshot.stages.map((stage, offset) => {
    if (offset !== stageOffset) return stage
    if (event.kind === "progress") {
      return {
        ...stage,
        percent: Math.max(stage.percent, Math.min(100, Math.max(0, event.percent))),
        operation: event.operation || stage.operation,
      }
    }
    return {
      ...stage,
      status: event.status,
      percent: event.status === "complete" || event.status === "skipped" ? 100 : stage.percent,
      operation: event.operation || stage.operation,
      elapsedSeconds: Math.max(stage.elapsedSeconds, event.elapsed_seconds),
    }
  })
  const stage = stages[stageOffset]
  return {
    ...snapshot,
    stages,
    currentStage: event.stage,
    currentStageIndex: stageOffset + 1,
    stagePercent: stage?.percent ?? snapshot.stagePercent,
    operation: event.operation || snapshot.operation,
    command: event.command || snapshot.command,
    error: event.kind === "stage"
      && (event.status === "failed" || event.status === "interrupted")
      && event.error
      ? event.error
      : snapshot.error,
  }
}
