import { readFile } from "node:fs/promises"
import { fileURLToPath } from "node:url"

const packagePath = fileURLToPath(new URL("../package.json", import.meta.url))
const value = JSON.parse(await readFile(packagePath, "utf8"))
const required = [
  ["electron", value.devDependencies?.electron],
  ["electron-vite", value.devDependencies?.["electron-vite"]],
  ["electron-builder", value.devDependencies?.["electron-builder"]],
  ["solid-js", value.dependencies?.["solid-js"]],
  ["tailwindcss", value.devDependencies?.tailwindcss],
]
for (const [name, version] of required) {
  if (typeof version !== "string" || !/^\d+\.\d+\.\d+$/.test(version)) {
    throw new Error(`${name} 必须锁定为精确的稳定版本，实际为 ${String(version)}`)
  }
}
const nodeMajor = Number(process.versions.node.split(".")[0])
if (nodeMajor < 24) throw new Error(`需要 Node.js >= 24，实际为 ${process.versions.node}`)

console.log(`VERSIONS_OK node=${process.versions.node} ${required.map(([name, version]) => `${name}=${version}`).join(" ")}`)
