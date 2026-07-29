import { join } from "node:path"
import { app, BrowserWindow, session } from "electron"
import { registerIpcHandlers } from "./ipc"
import { allowedRendererUrl, createWindowOptions } from "./window-options"

let mainWindow: BrowserWindow | null = null

export function createMainWindow(): BrowserWindow {
  const preload = join(__dirname, "../preload/index.js")
  const rendererFile = join(__dirname, "../renderer/index.html")
  const window = new BrowserWindow(createWindowOptions(preload))

  window.webContents.setWindowOpenHandler(() => ({ action: "deny" }))
  window.webContents.on("will-navigate", (event, url) => {
    if (!allowedRendererUrl(url, process.env.ELECTRON_RENDERER_URL, rendererFile)) {
      event.preventDefault()
    }
  })
  window.once("ready-to-show", () => window.show())

  if (process.env.ELECTRON_RENDERER_URL) {
    void window.loadURL(process.env.ELECTRON_RENDERER_URL)
  } else {
    void window.loadFile(rendererFile)
  }

  window.on("closed", () => {
    if (mainWindow === window) mainWindow = null
  })
  mainWindow = window
  return window
}

app.whenReady().then(() => {
  registerIpcHandlers()
  session.defaultSession.setPermissionRequestHandler((_webContents, _permission, callback) => {
    callback(false)
  })
  createMainWindow()

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createMainWindow()
  })
})

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit()
})
