import { readFile } from "node:fs/promises"
import { fileURLToPath } from "node:url"
import { describe, expect, it } from "vitest"

const fixture = fileURLToPath(new URL("../fixtures/", import.meta.url))

describe("desktop quick fixture", () => {
  it("keeps a 10-20 line UTF-8 novel paired with every explicit dialogue label", async () => {
    const novel = await readFile(`${fixture}novel.txt`, "utf8")
    const labels = (await readFile(`${fixture}labels.txt`, "utf8"))
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter(Boolean)
    const lines = novel.replace(/\r?\n$/, "").split(/\r?\n/)
    const dialogues = [...novel.matchAll(/「[^」]*」/g)]

    expect(lines.length).toBeGreaterThanOrEqual(10)
    expect(lines.length).toBeLessThanOrEqual(20)
    expect(dialogues).toHaveLength(labels.length)
    expect(labels).toEqual(["罗伦斯", "赫萝", "骑士", "罗伦斯"])
  })
})
