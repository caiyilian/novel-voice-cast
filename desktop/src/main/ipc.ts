import { dialog, ipcMain, shell } from "electron"
import type { InputKind, TextFileSelection } from "../preload/types"
import type { PipelineController } from "./pipeline-controller"
import { isInputKind, validateTextFile } from "./text-file"

function assertInputKind(value: unknown): asserts value is InputKind {
  if (!isInputKind(value)) throw new TypeError("Invalid input kind")
}

export function registerIpcHandlers(controller: PipelineController): void {
  ipcMain.handle("dialog:pick-text-file", async (_event, inputKind: unknown) => {
    assertInputKind(inputKind)
    const result = await dialog.showOpenDialog({
      title: inputKind === "novel" ? "选择小说原文" : "选择角色标注",
      properties: ["openFile"],
      filters: [{ name: "UTF-8 文本", extensions: ["txt"] }],
    })
    if (result.canceled || result.filePaths.length === 0) return null
    return validateTextFile(result.filePaths[0])
  })

  ipcMain.handle(
    "file:validate-text-file",
    async (_event, filePath: unknown, inputKind: unknown): Promise<TextFileSelection> => {
      assertInputKind(inputKind)
      return validateTextFile(filePath)
    },
  )

  ipcMain.handle("pipeline:get-state", () => controller.getState())
  ipcMain.handle("pipeline:start", (_event, request) => controller.start(request))
  ipcMain.handle("pipeline:stop", () => controller.stop())
  ipcMain.handle("output:open-directory", async () => {
    try {
      const directory = await controller.getOpenableOutputDirectory()
      const error = await shell.openPath(directory)
      return { ok: !error, error: error || null }
    } catch (error) {
      return { ok: false, error: error instanceof Error ? error.message : String(error) }
    }
  })
}
