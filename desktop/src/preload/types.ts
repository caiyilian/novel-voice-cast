export type InputKind = "novel" | "labels"

export const PIPELINE_STAGES = [
  "parse",
  "gender",
  "emotion",
  "performance",
  "tts",
  "splice",
  "bgm-segment",
  "bgm-label",
  "bgm-generate",
  "bgm-mix",
  "illustration-plan",
  "illustrations",
  "video",
] as const

export type PipelineStage = (typeof PIPELINE_STAGES)[number]
export type PipelineStatus =
  | "idle"
  | "starting"
  | "running"
  | "stopping"
  | "interrupted"
  | "failed"
  | "completed"

export type StageRuntimeStatus =
  | "pending"
  | "not-selected"
  | "running"
  | "complete"
  | "skipped"
  | "failed"
  | "interrupted"

export interface StageRuntime {
  name: PipelineStage
  index: number
  status: StageRuntimeStatus
  percent: number
  operation: string
  elapsedSeconds: number
}

export interface PipelineLogEntry {
  id: number
  timestamp: string
  level: string
  message: string
  stream: "stdout" | "stderr" | "structured"
  stage: PipelineStage | null
}

export type ManifestStatus = "not-read" | "valid" | "missing" | "invalid" | "stale"

export interface PipelineArtifact {
  path: string
  exists: boolean
}

export interface OpenDirectoryResult {
  ok: boolean
  error: string | null
}

export interface PipelineStartRequest {
  novelPath: string
  labelsPath: string
  fromStage?: PipelineStage
  toStage?: PipelineStage
}

export interface PipelineSnapshot {
  status: PipelineStatus
  pid: number | null
  command: string
  projectRoot: string
  outputDirectory: string
  manifestPath: string
  logPath: string
  startedAt: string | null
  finishedAt: string | null
  exitCode: number | null
  error: string | null
  request: PipelineStartRequest | null
  currentStage: PipelineStage | null
  currentStageIndex: number | null
  stagePercent: number
  operation: string
  stages: StageRuntime[]
  logs: PipelineLogEntry[]
  totalElapsedSeconds: number
  manifestStatus: ManifestStatus
  manifestMessage: string
  artifacts: PipelineArtifact[]
  outputDirectoryAvailable: boolean
}

export type PipelineEvent =
  | { type: "state"; state: PipelineSnapshot }
  | { type: "output"; stream: "stdout" | "stderr"; line: string; timestamp: string }

export interface SelectedTextFile {
  path: string
  name: string
  size: number
  modifiedAt: string
}

export type TextFileSelection =
  | { ok: true; file: SelectedTextFile }
  | { ok: false; error: string }

export interface NovelVoiceCastAPI {
  platform: string
  versions: Readonly<{
    electron: string
    chrome: string
  }>
  pickTextFile: (inputKind: InputKind) => Promise<TextFileSelection | null>
  acceptDroppedTextFile: (file: File, inputKind: InputKind) => Promise<TextFileSelection>
  getPipelineState: () => Promise<PipelineSnapshot>
  startPipeline: (request: PipelineStartRequest) => Promise<PipelineSnapshot>
  stopPipeline: () => Promise<PipelineSnapshot>
  openOutputDirectory: () => Promise<OpenDirectoryResult>
  onPipelineEvent: (callback: (event: PipelineEvent) => void) => () => void
}
