import { readFile } from "node:fs/promises"
import { fileURLToPath } from "node:url"

const fixtureDirectory = fileURLToPath(new URL("../fixtures/", import.meta.url))
const novel = await readFile(`${fixtureDirectory}novel.txt`, "utf8")
const labels = (await readFile(`${fixtureDirectory}labels.txt`, "utf8"))
  .split(/\r?\n/)
  .map((line) => line.trim())
  .filter(Boolean)
const lines = novel.replace(/\r?\n$/, "").split(/\r?\n/)
const dialogueCount = [...novel.matchAll(/「[^」]*」/g)].length
const narrativeCount = lines.filter((line) => line.trim() && !line.includes("「") && !line.startsWith("第一章")).length

if (lines.length < 10 || lines.length > 20) {
  throw new Error(`桌面测试小说必须为 10～20 行，实际为 ${lines.length} 行`)
}
if (dialogueCount !== labels.length) {
  throw new Error(`样例对话与 labels 数量不匹配：${dialogueCount} != ${labels.length}`)
}
if (!novel.includes("第一章") || narrativeCount === 0) {
  throw new Error("样例必须同时覆盖章节、旁白和显式对话")
}

console.log(`FIXTURE_OK lines=${lines.length} dialogues=${dialogueCount} labels=${labels.length}`)
