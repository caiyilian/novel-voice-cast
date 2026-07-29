import { dialog, ipcMain } from "electron"
import type { InputKind, TextFileSelection } from "../preload/types"
import { isInputKind, validateTextFile } from "./text-file"

function assertInputKind(value: unknown): asserts value is InputKind {
  if (!isInputKind(value)) throw new TypeError("Invalid input kind")
}

export function registerIpcHandlers(): void {
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
}
