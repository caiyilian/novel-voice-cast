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
  onPipelineEvent: (callback: (event: PipelineEvent) => void) => () => void
}
