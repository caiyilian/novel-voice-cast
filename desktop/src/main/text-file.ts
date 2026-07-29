import { constants } from "node:fs"
import { access, lstat, realpath } from "node:fs/promises"
import { basename, extname } from "node:path"
import type {
  SelectedTextFile,
  TextFileSelection,
} from "../preload/types"

export const INPUT_KINDS = ["novel", "labels"] as const

export function isInputKind(value: unknown): value is (typeof INPUT_KINDS)[number] {
  return typeof value === "string" && INPUT_KINDS.includes(value as (typeof INPUT_KINDS)[number])
}

export async function validateTextFile(filePath: unknown): Promise<TextFileSelection> {
  if (typeof filePath !== "string" || !filePath.trim() || filePath.includes("\0")) {
    return { ok: false, error: "没有取得有效的文件路径" }
  }
  if (extname(filePath).toLowerCase() !== ".txt") {
    return { ok: false, error: "请选择 .txt 文本文件" }
  }

  try {
    const information = await lstat(filePath)
    if (!information.isFile()) {
      return { ok: false, error: "所选路径不是普通文件" }
    }
    await access(filePath, constants.R_OK)
    const canonicalPath = await realpath(filePath)
    const file: SelectedTextFile = {
      path: canonicalPath,
      name: basename(canonicalPath),
      size: information.size,
      modifiedAt: information.mtime.toISOString(),
    }
    return { ok: true, file }
  } catch (error) {
    const code = error instanceof Error && "code" in error ? String(error.code) : ""
    if (code === "ENOENT") return { ok: false, error: "文件不存在或已被移动" }
    if (code === "EACCES" || code === "EPERM") {
      return { ok: false, error: "文件不可读，请检查访问权限" }
    }
    return { ok: false, error: "无法读取所选文件" }
  }
}
