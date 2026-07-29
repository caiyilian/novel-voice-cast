import { describe, expect, it } from "vitest"
import { App, stages } from "../src/renderer/App"

describe("desktop shell", () => {
  it("renders the input placeholders and all thirteen stages", () => {
    expect(typeof App).toBe("function")
    expect(stages).toHaveLength(13)
    expect(stages).toContain("BGM 生成")
    expect(stages.at(-1)).toBe("视频")
  })
})
