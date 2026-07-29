import { spawn } from "node:child_process"
import { stat } from "node:fs/promises"
import { fileURLToPath } from "node:url"

if (process.platform !== "win32") {
  console.log("SMOKE_SKIPPED platform is not win32")
  process.exit(0)
}

const executable = fileURLToPath(new URL("../release/win-unpacked/Novel Voice Cast.exe", import.meta.url))
await stat(executable)
const child = spawn(executable, [], {
  cwd: fileURLToPath(new URL("../../", import.meta.url)),
  windowsHide: true,
  stdio: "ignore",
})
let exitCode
child.once("exit", (code) => { exitCode = code })
await new Promise((resolve) => setTimeout(resolve, 4_000))
if (exitCode !== undefined) {
  throw new Error(`打包后的应用过早退出，exit=${exitCode}`)
}
child.kill()
console.log(`SMOKE_OK pid=${child.pid}`)
