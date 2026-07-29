// @vitest-environment jsdom
import { render } from "solid-js/web"
import { afterEach, describe, expect, it, vi } from "vitest"
import type {
  NovelVoiceCastAPI,
  PipelineSnapshot,
  SelectedTextFile,
} from "../src/preload/types"
import { App, stages } from "../src/renderer/App"

let dispose: (() => void) | undefined

afterEach(() => {
  dispose?.()
  dispose = undefined
  document.body.replaceChildren()
})

function file(name: string): SelectedTextFile {
  return {
    path: `E:/测试 输入/${name}`,
    name,
    size: 2048,
    modifiedAt: "2026-07-29T00:00:00.000Z",
  }
}

const idlePipeline: PipelineSnapshot = {
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
}

describe("desktop input shell", () => {
  it("shows all stages and only enables start after both pickers succeed", async () => {
    const api: NovelVoiceCastAPI = {
      platform: "win32",
      versions: { electron: "43.2.0", chrome: "142" },
      pickTextFile: vi.fn(async (kind) => ({
        ok: true as const,
        file: file(kind === "novel" ? "novel.txt" : "labels.txt"),
      })),
      acceptDroppedTextFile: vi.fn(),
      getPipelineState: vi.fn(async () => idlePipeline),
      startPipeline: vi.fn(async (request) => ({
        ...idlePipeline,
        status: "running" as const,
        pid: 1234,
        request,
      })),
      stopPipeline: vi.fn(async () => ({ ...idlePipeline, status: "stopping" as const })),
      onPipelineEvent: vi.fn(() => () => undefined),
    }
    Object.defineProperty(window, "novelVoiceCast", { value: api, configurable: true })
    const container = document.createElement("div")
    document.body.append(container)
    dispose = render(() => <App />, container)

    expect(stages).toHaveLength(13)
    const start = Array.from(container.querySelectorAll("button")).find(
      (button) => button.textContent?.includes("开始完整流程"),
    )
    expect(start?.disabled).toBe(true)

    container.querySelector<HTMLButtonElement>('[aria-label="选择小说原文"]')?.click()
    await new Promise((resolve) => setTimeout(resolve, 0))
    expect(start?.disabled).toBe(true)
    expect(container.textContent).toContain("novel.txt")

    container.querySelector<HTMLButtonElement>('[aria-label="选择角色标注"]')?.click()
    await new Promise((resolve) => setTimeout(resolve, 0))
    expect(start?.disabled).toBe(false)
    expect(container.textContent).toContain("labels.txt")
    expect(api.pickTextFile).toHaveBeenCalledTimes(2)

    start?.click()
    await new Promise((resolve) => setTimeout(resolve, 0))
    expect(api.startPipeline).toHaveBeenCalledWith({
      novelPath: "E:/测试 输入/novel.txt",
      labelsPath: "E:/测试 输入/labels.txt",
    })
    const stop = Array.from(container.querySelectorAll("button")).find(
      (button) => button.textContent?.includes("停止并保留断点"),
    )
    expect(stop).toBeTruthy()
    stop?.click()
    await new Promise((resolve) => setTimeout(resolve, 0))
    expect(api.stopPipeline).toHaveBeenCalledOnce()
    expect(container.textContent).toContain("正在停止")
  })
})
