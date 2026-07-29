import { describe, expect, it } from "vitest"
import {
  applySelection,
  clearSelection,
  emptyInputState,
  formatFileSize,
  inputsReady,
} from "../src/renderer/input-state"

const novel = {
  path: "E:/小说/novel.txt",
  name: "novel.txt",
  size: 2048,
  modifiedAt: "2026-07-29T00:00:00.000Z",
}
const labels = { ...novel, path: "E:/小说/labels.txt", name: "labels.txt" }

describe("input gate state", () => {
  it("only becomes ready when both validated files are present", () => {
    let state = emptyInputState()
    expect(inputsReady(state)).toBe(false)
    state = applySelection(state, "novel", { ok: true, file: novel })
    expect(inputsReady(state)).toBe(false)
    state = applySelection(state, "labels", { ok: true, file: labels })
    expect(inputsReady(state)).toBe(true)
    state = clearSelection(state, "novel")
    expect(inputsReady(state)).toBe(false)
  })

  it("stores validation errors without destroying the other valid slot", () => {
    let state = applySelection(emptyInputState(), "novel", { ok: true, file: novel })
    state = applySelection(state, "labels", { ok: false, error: "请选择 .txt 文本文件" })

    expect(state.novel.file).toEqual(novel)
    expect(state.labels.file).toBeNull()
    expect(state.labels.error).toContain(".txt")
  })

  it("keeps state unchanged when a native picker is cancelled and formats sizes", () => {
    const state = emptyInputState()
    expect(applySelection(state, "novel", null)).toBe(state)
    expect(formatFileSize(512)).toBe("512 B")
    expect(formatFileSize(2048)).toBe("2.0 KB")
    expect(formatFileSize(2 * 1024 * 1024)).toBe("2.0 MB")
  })
})
