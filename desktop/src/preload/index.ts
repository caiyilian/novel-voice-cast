import { contextBridge, ipcRenderer, webUtils } from "electron"
import type { InputKind, NovelVoiceCastAPI } from "./types"

const api: NovelVoiceCastAPI = Object.freeze({
  platform: process.platform,
  versions: Object.freeze({
    electron: process.versions.electron,
    chrome: process.versions.chrome,
  }),
  pickTextFile: (inputKind: InputKind) => ipcRenderer.invoke("dialog:pick-text-file", inputKind),
  acceptDroppedTextFile: (file: File, inputKind: InputKind) =>
    ipcRenderer.invoke("file:validate-text-file", webUtils.getPathForFile(file), inputKind),
})

contextBridge.exposeInMainWorld("novelVoiceCast", api)
