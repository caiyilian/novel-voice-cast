import { existsSync, readFileSync, statSync } from "node:fs"
import { dirname, isAbsolute, resolve } from "node:path"
import { parse } from "yaml"

export interface ProjectPaths {
  root: string
  python: string
  runFull: string
  config: string
  output: string
  logs: string
}

function isFile(path: string): boolean {
  try {
    return statSync(path).isFile()
  } catch {
    return false
  }
}

function candidateRoot(start: string): string | null {
  let current = resolve(start)
  for (let depth = 0; depth < 7; depth += 1) {
    if (isFile(resolve(current, "scripts", "run_full.py"))) return current
    const parent = dirname(current)
    if (parent === current) break
    current = parent
  }
  return null
}

function configuredOutputDirectory(configPath: string, root: string): string {
  try {
    const value = parse(readFileSync(configPath, "utf8")) as unknown
    if (
      value
      && typeof value === "object"
      && "output" in value
      && value.output
      && typeof value.output === "object"
      && "dir" in value.output
      && typeof value.output.dir === "string"
      && value.output.dir.trim()
    ) {
      return isAbsolute(value.output.dir) ? resolve(value.output.dir) : resolve(root, value.output.dir)
    }
  } catch (error) {
    throw new Error(`无法读取配置中的输出目录：${error instanceof Error ? error.message : String(error)}`)
  }
  return resolve(root, "output")
}

export function resolveProjectPaths(
  explicitRoot?: string,
  configOverride?: string,
  pythonOverride?: string,
): ProjectPaths {
  const candidates = [
    explicitRoot,
    process.env.NOVEL_VOICE_CAST_ROOT,
    process.cwd(),
    resolve(__dirname, "../../.."),
  ].filter((value): value is string => Boolean(value))
  const root = candidates.map(candidateRoot).find((value): value is string => Boolean(value))
  if (!root) {
    throw new Error(
      "找不到 Novel Voice Cast 项目根目录；请设置 NOVEL_VOICE_CAST_ROOT 后重启桌面应用",
    )
  }

  const python = pythonOverride
    ? resolve(pythonOverride)
    : resolve(root, process.platform === "win32" ? ".venv/Scripts/python.exe" : ".venv/bin/python")
  const runFull = resolve(root, "scripts/run_full.py")
  const config = configOverride
    ? isAbsolute(configOverride)
      ? configOverride
      : resolve(root, configOverride)
    : resolve(root, "config/config.yaml")
  const missing = [python, runFull, config].filter((path) => !isFile(path))
  if (missing.length > 0) {
    throw new Error(`项目运行时不完整，缺少：${missing.join("，")}`)
  }
  return {
    root,
    python,
    runFull,
    config,
    output: configuredOutputDirectory(config, root),
    logs: resolve(root, "logs"),
  }
}

export function projectMarkersExist(root: string): boolean {
  return existsSync(resolve(root, "scripts/run_full.py")) && existsSync(resolve(root, "config"))
}
