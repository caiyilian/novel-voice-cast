import { mkdtemp, mkdir, rm, writeFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { afterEach, describe, expect, it } from "vitest"
import { isInputKind, validateTextFile } from "../src/main/text-file"

const temporaryDirectories: string[] = []

afterEach(async () => {
  await Promise.all(temporaryDirectories.splice(0).map((path) => rm(path, { recursive: true, force: true })))
})

async function temporaryDirectory(): Promise<string> {
  const path = await mkdtemp(join(tmpdir(), "nvc-desktop-"))
  temporaryDirectories.push(path)
  return path
}

describe("text input validation", () => {
  it("accepts a readable Chinese .txt path and returns canonical metadata", async () => {
    const directory = await temporaryDirectory()
    const path = join(directory, "中文 小说.txt")
    await writeFile(path, "第一章\n旁白。", "utf8")

    const result = await validateTextFile(path)

    expect(result.ok).toBe(true)
    if (!result.ok) return
    expect(result.file.path).toBe(path)
    expect(result.file.name).toBe("中文 小说.txt")
    expect(result.file.size).toBeGreaterThan(0)
  })

  it("rejects wrong extensions, directories, missing files, and invalid values", async () => {
    const directory = await temporaryDirectory()
    const markdown = join(directory, "novel.md")
    const nested = join(directory, "folder.txt")
    await writeFile(markdown, "text", "utf8")
    await mkdir(nested)

    await expect(validateTextFile(markdown)).resolves.toMatchObject({ ok: false, error: "请选择 .txt 文本文件" })
    await expect(validateTextFile(nested)).resolves.toMatchObject({ ok: false, error: "所选路径不是普通文件" })
    await expect(validateTextFile(join(directory, "missing.txt"))).resolves.toMatchObject({ ok: false, error: "文件不存在或已被移动" })
    await expect(validateTextFile(null)).resolves.toMatchObject({ ok: false })
  })

  it("only accepts the two declared input kinds", () => {
    expect(isInputKind("novel")).toBe(true)
    expect(isInputKind("labels")).toBe(true)
    expect(isInputKind("config")).toBe(false)
  })
})
