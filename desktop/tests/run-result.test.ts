import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join, resolve } from "node:path"
import { afterEach, describe, expect, it } from "vitest"
import { readRunResult, requireOpenableDirectory } from "../src/main/run-result"

const temporaryDirectories: string[] = []

afterEach(async () => {
  await Promise.all(temporaryDirectories.splice(0).map((path) => rm(path, { recursive: true, force: true })))
})

async function fixture() {
  const root = await mkdtemp(join(tmpdir(), "nvc-result-"))
  temporaryDirectories.push(root)
  const output = join(root, "output")
  const manifestPath = join(output, "run_full_manifest.json")
  await mkdir(output)
  const artifact = join(output, "fixture.mp3")
  await writeFile(artifact, "audio")
  const options = {
    manifestPath,
    outputDirectory: output,
    projectRoot: root,
    expectedStartedAt: "2026-07-30T00:00:00.000Z",
    request: {
      novelPath: join(root, "novel.txt"),
      labelsPath: join(root, "labels.txt"),
      fromStage: "parse" as const,
      toStage: "parse" as const,
    },
    now: new Date("2026-07-30T00:00:10.000Z"),
  }
  return { root, output, manifestPath, artifact, options }
}

function manifest(root: string, artifact: string, status = "complete") {
  return {
    version: 1,
    root,
    selected_stages: ["parse"],
    run_started_at: "2026-07-30T00:00:01.000Z",
    run_finished_at: "2026-07-30T00:00:06.000Z",
    run_status: status,
    ...(status === "failed" ? { run_error: "模型失败" } : {}),
    stages: {
      parse: {
        status: status === "complete" ? "complete" : status,
        elapsed_seconds: 4.25,
        artifacts: [artifact, artifact],
      },
    },
  }
}

describe("run manifest result", () => {
  it("accepts the current complete manifest and de-duplicates artifacts", async () => {
    const value = await fixture()
    await writeFile(value.manifestPath, JSON.stringify(manifest(value.root, value.artifact)), "utf8")

    const result = await readRunResult(value.options)

    expect(result.manifestStatus).toBe("valid")
    expect(result.runStatus).toBe("complete")
    expect(result.totalElapsedSeconds).toBe(5)
    expect(result.stages).toEqual([{
      name: "parse",
      status: "complete",
      elapsedSeconds: 4.25,
      artifacts: [value.artifact, value.artifact],
    }])
    expect(result.artifacts.filter((artifact) => artifact.path === resolve(value.artifact))).toHaveLength(1)
    expect(result.artifacts.find((artifact) => artifact.path.endsWith("full_volume_bgm.mp3"))?.exists).toBe(false)
  })

  it.each(["failed", "interrupted"])("keeps a current %s manifest valid without calling it complete", async (status) => {
    const value = await fixture()
    await writeFile(value.manifestPath, JSON.stringify(manifest(value.root, value.artifact, status)), "utf8")

    const result = await readRunResult(value.options)

    expect(result.manifestStatus).toBe("valid")
    expect(result.runStatus).toBe(status)
    expect(result.stages[0]?.status).toBe(status)
  })

  it("rejects stale, corrupt, and missing manifests with clear states", async () => {
    const value = await fixture()
    const stale = manifest(value.root, value.artifact)
    stale.run_started_at = "2026-07-29T00:00:00.000Z"
    stale.run_finished_at = "2026-07-29T00:00:06.000Z"
    await writeFile(value.manifestPath, JSON.stringify(stale), "utf8")
    expect((await readRunResult(value.options)).manifestStatus).toBe("stale")

    await writeFile(value.manifestPath, "{broken", "utf8")
    expect((await readRunResult(value.options)).manifestStatus).toBe("invalid")

    await rm(value.manifestPath)
    expect((await readRunResult(value.options)).manifestStatus).toBe("missing")
  })

  it("only accepts a real output directory", async () => {
    const value = await fixture()
    await expect(requireOpenableDirectory(value.output)).resolves.toBe(resolve(value.output))
    await expect(requireOpenableDirectory(join(value.root, "missing"))).rejects.toThrow("不存在")
  })
})
