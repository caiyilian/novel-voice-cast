import { pathToFileURL } from "node:url"
import { describe, expect, it } from "vitest"
import {
  allowedRendererUrl,
  createWindowOptions,
  secureWebPreferences,
} from "../src/main/window-options"

describe("secure BrowserWindow options", () => {
  it("keeps renderer isolated and sandboxed", () => {
    const preferences = secureWebPreferences("C:/app/preload.js")

    expect(preferences).toMatchObject({
      preload: "C:/app/preload.js",
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
      webSecurity: true,
    })
  })

  it("creates a hidden window until its renderer is ready", () => {
    const options = createWindowOptions("C:/app/preload.js")

    expect(options.show).toBe(false)
    expect(options.webPreferences).toEqual(secureWebPreferences("C:/app/preload.js"))
    expect(options.minWidth).toBeGreaterThanOrEqual(900)
  })

  it("only allows the exact development origin or packaged renderer file", () => {
    const rendererFile = "C:/app/out/renderer/index.html"

    expect(allowedRendererUrl("http://localhost:5173/dashboard", "http://localhost:5173", rendererFile)).toBe(true)
    expect(allowedRendererUrl("http://localhost:5173.evil.test", "http://localhost:5173", rendererFile)).toBe(false)
    expect(allowedRendererUrl("https://example.com", undefined, rendererFile)).toBe(false)
    expect(allowedRendererUrl(pathToFileURL(rendererFile).toString(), undefined, rendererFile)).toBe(true)
    expect(allowedRendererUrl(pathToFileURL("C:/other/file.html").toString(), undefined, rendererFile)).toBe(false)
  })
})
