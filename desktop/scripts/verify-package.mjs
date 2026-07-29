import { readdir, stat } from "node:fs/promises"
import { fileURLToPath } from "node:url"

const desktop = fileURLToPath(new URL("../", import.meta.url))
const release = `${desktop}release`
const unpacked = `${release}/win-unpacked/Novel Voice Cast.exe`
const entries = await readdir(release)
const installerName = entries.find((name) => (
  /^novel-voice-cast-desktop-.*-x64\.exe$/i.test(name)
))
if (!installerName) throw new Error("未找到 electron-builder 生成的 x64 NSIS 安装器")

for (const path of [unpacked, `${release}/${installerName}`]) {
  const info = await stat(path)
  if (!info.isFile() || info.size < 1_000_000) {
    throw new Error(`打包产物不存在或异常过小：${path}`)
  }
}

console.log(`PACKAGE_OK unpacked=${unpacked}`)
console.log(`PACKAGE_OK installer=${release}/${installerName}`)
