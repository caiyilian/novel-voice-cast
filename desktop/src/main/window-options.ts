import { pathToFileURL } from "node:url"
import type { BrowserWindowConstructorOptions, WebPreferences } from "electron"

export function allowedRendererUrl(
  url: string,
  developmentUrl: string | undefined,
  rendererFile: string,
): boolean {
  try {
    const target = new URL(url)
    if (developmentUrl) return target.origin === new URL(developmentUrl).origin
    const expected = pathToFileURL(rendererFile)
    return target.protocol === "file:" && target.pathname === expected.pathname
  } catch {
    return false
  }
}

export function secureWebPreferences(preload: string): WebPreferences {
  return {
    preload,
    contextIsolation: true,
    nodeIntegration: false,
    sandbox: true,
    webSecurity: true,
  }
}

export function createWindowOptions(preload: string): BrowserWindowConstructorOptions {
  return {
    width: 1180,
    height: 780,
    minWidth: 920,
    minHeight: 620,
    show: false,
    backgroundColor: "#0b1220",
    autoHideMenuBar: true,
    title: "Novel Voice Cast",
    webPreferences: secureWebPreferences(preload),
  }
}
