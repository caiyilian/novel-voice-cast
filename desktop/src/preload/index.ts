import { contextBridge, ipcRenderer, webUtils } from "electron"
import type { InputKind, NovelVoiceCastAPI, PipelineEvent, PipelineStartRequest } from "./types"

const api: NovelVoiceCastAPI = Object.freeze({
  platform: process.platform,
  versions: Object.freeze({
    electron: process.versions.electron,
    chrome: process.versions.chrome,
  }),
  pickTextFile: (inputKind: InputKind) => ipcRenderer.invoke("dialog:pick-text-file", inputKind),
  acceptDroppedTextFile: (file: File, inputKind: InputKind) =>
    ipcRenderer.invoke("file:validate-text-file", webUtils.getPathForFile(file), inputKind),
  getPipelineState: () => ipcRenderer.invoke("pipeline:get-state"),
  startPipeline: (request: PipelineStartRequest) => ipcRenderer.invoke("pipeline:start", request),
  stopPipeline: () => ipcRenderer.invoke("pipeline:stop"),
  onPipelineEvent: (callback: (event: PipelineEvent) => void) => {
    const handler = (_event: Electron.IpcRendererEvent, pipelineEvent: PipelineEvent) => {
      callback(pipelineEvent)
    }
    ipcRenderer.on("pipeline:event", handler)
    return () => ipcRenderer.removeListener("pipeline:event", handler)
  },
})

contextBridge.exposeInMainWorld("novelVoiceCast", api)
