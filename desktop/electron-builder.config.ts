import type { Configuration } from "electron-builder"

const config: Configuration = {
  appId: "com.caiyilian.novelvoicecast",
  productName: "Novel Voice Cast",
  artifactName: "novel-voice-cast-desktop-${version}-${arch}.${ext}",
  asar: true,
  electronDist: "node_modules/electron/dist",
  directories: {
    output: "release",
  },
  files: ["out/**/*", "package.json"],
  win: {
    target: ["nsis"],
  },
  nsis: {
    oneClick: true,
    perMachine: false,
    allowElevation: true,
    createDesktopShortcut: true,
    createStartMenuShortcut: true,
  },
}

export default config
