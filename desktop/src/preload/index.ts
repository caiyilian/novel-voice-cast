import { contextBridge } from "electron"
import type { NovelVoiceCastAPI } from "./types"

const api: NovelVoiceCastAPI = Object.freeze({
  platform: process.platform,
  versions: Object.freeze({
    electron: process.versions.electron,
    chrome: process.versions.chrome,
  }),
})

contextBridge.exposeInMainWorld("novelVoiceCast", api)
