import { describe, expect, it } from "vitest"
import type { PipelineSnapshot } from "../src/preload/types"
import {
  appendPipelineLog,
  applyStructuredEvent,
  createStageRuntime,
  parseStructuredLine,
} from "../src/main/pipeline-events"

function snapshot(): PipelineSnapshot {
  return {
    status: "running",
    pid: 42,
    command: "python run_full.py",
    projectRoot: "E:/project",
    outputDirectory: "E:/project/output",
    manifestPath: "E:/project/output/run_full_manifest.json",
    logPath: "E:/project/logs/desktop.log",
    startedAt: "2026-07-30T00:00:00.000Z",
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
    totalElapsedSeconds: 0,
    manifestStatus: "not-read",
    manifestMessage: "尚未读取 manifest",
    artifacts: [],
    outputDirectoryAvailable: false,
  }
}

describe("structured pipeline events", () => {
  it("parses valid UTF-8 events and rejects malformed markers", () => {
    const event = parseStructuredLine(
      '[PROGRESS] {"version":1,"timestamp":"2026-07-30T00:00:00Z","stage":"emotion","current":3,"total":4,"percent":75,"status":"running","operation":"正在标注赫萝"}',
    )
    expect(event).toMatchObject({ kind: "progress", stage: "emotion", percent: 75, operation: "正在标注赫萝" })
    expect(parseStructuredLine("[PROGRESS] {not json}")).toBeNull()
    expect(parseStructuredLine('[STAGE] {"version":2}')).toBeNull()
    expect(parseStructuredLine("ordinary output")).toBeNull()
  })

  it("keeps progress monotonic and updates the 13-stage runtime", () => {
    const running = parseStructuredLine(
      '[STAGE] {"version":1,"timestamp":"2026-07-30T00:00:00Z","stage":"tts","index":5,"total":13,"status":"running","elapsed_seconds":0,"operation":"生成语音"}',
    )!
    const eighty = parseStructuredLine(
      '[PROGRESS] {"version":1,"timestamp":"2026-07-30T00:00:01Z","stage":"tts","current":8,"total":10,"percent":80,"status":"running","operation":"8/10"}',
    )!
    const stale = parseStructuredLine(
      '[PROGRESS] {"version":1,"timestamp":"2026-07-30T00:00:02Z","stage":"tts","current":2,"total":10,"percent":20,"status":"running","operation":"重读断点"}',
    )!
    const complete = parseStructuredLine(
      '[STAGE] {"version":1,"timestamp":"2026-07-30T00:00:03Z","stage":"tts","index":5,"total":13,"status":"complete","elapsed_seconds":3,"operation":"生成语音完成"}',
    )!
    const result = [running, eighty, stale, complete].reduce(applyStructuredEvent, snapshot())

    expect(result.stages).toHaveLength(13)
    expect(result.stages[4]).toMatchObject({ status: "complete", percent: 100, elapsedSeconds: 3 })
    expect(result.currentStage).toBe("tts")
    expect(result.stagePercent).toBe(100)
  })

  it("marks stages outside a requested range and bounds logs", () => {
    const state = snapshot()
    state.stages = createStageRuntime({
      novelPath: "novel.txt",
      labelsPath: "labels.txt",
      fromStage: "tts",
      toStage: "splice",
    })
    expect(state.stages.slice(0, 4).every((stage) => stage.status === "not-selected")).toBe(true)
    expect(state.stages[4]!.status).toBe("pending")
    expect(state.stages[5]!.status).toBe("pending")
    expect(state.stages[6]!.status).toBe("not-selected")

    const result = Array.from({ length: 6 }, (_, index) => index).reduce(
      (current, index) => appendPipelineLog(current, {
        timestamp: `2026-07-30T00:00:0${index}Z`,
        level: "INFO",
        message: `line-${index}`,
        stream: "stdout",
        stage: null,
      }, 3),
      state,
    )
    expect(result.logs.map((entry) => entry.message)).toEqual(["line-3", "line-4", "line-5"])
  })
})
