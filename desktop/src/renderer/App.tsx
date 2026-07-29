import { createEffect, createSignal, For, onCleanup, onMount, Show } from "solid-js"
import {
  PIPELINE_STAGES,
  type InputKind,
  type PipelineSnapshot,
  type PipelineStage,
  type TextFileSelection,
} from "../preload/types"
import {
  applySelection,
  clearSelection,
  emptyInputState,
  formatFileSize,
  inputsReady,
} from "./input-state"

export const stages = [
  "解析", "性别", "情绪", "表演", "语音", "拼接", "BGM 分割",
  "BGM 标注", "BGM 生成", "BGM 混音", "插图规划", "插图生成", "视频",
]

const inputCopy: Record<InputKind, { title: string; hint: string }> = {
  novel: { title: "小说原文", hint: "novel.txt · UTF-8 小说正文" },
  labels: { title: "角色标注", hint: "labels.txt · 对话角色逐行标注" },
}

const idlePipeline: PipelineSnapshot = {
  status: "idle",
  pid: null,
  command: "",
  projectRoot: "",
  outputDirectory: "",
  manifestPath: "",
  logPath: "",
  startedAt: null,
  finishedAt: null,
  exitCode: null,
  error: null,
  request: null,
  currentStage: null,
  currentStageIndex: null,
  stagePercent: 0,
  operation: "等待开始",
  stages: PIPELINE_STAGES.map((name, index) => ({
    name,
    index: index + 1,
    status: "pending" as const,
    percent: 0,
    operation: "等待开始",
    elapsedSeconds: 0,
  })),
  logs: [],
}

const stageLabels: Record<PipelineStage, string> = Object.fromEntries(
  PIPELINE_STAGES.map((stage, index) => [stage, `${index + 1}. ${stages[index]}`]),
) as Record<PipelineStage, string>

const stageStatusLabels = {
  pending: "等待",
  "not-selected": "未选择",
  running: "运行中",
  complete: "完成",
  skipped: "跳过",
  failed: "失败",
  interrupted: "已停止",
} as const

function stageTone(status: keyof typeof stageStatusLabels): string {
  if (status === "complete") return "border-emerald-400/35 bg-emerald-400/[0.07] text-emerald-200"
  if (status === "running") return "border-amber-300/60 bg-amber-300/[0.09] text-amber-100"
  if (status === "failed") return "border-rose-400/50 bg-rose-400/[0.08] text-rose-200"
  if (status === "interrupted") return "border-orange-400/45 bg-orange-400/[0.07] text-orange-200"
  return "border-white/10 bg-white/[0.025] text-slate-400"
}

export function App() {
  const [inputs, setInputs] = createSignal(emptyInputState())
  const [dragging, setDragging] = createSignal<InputKind | null>(null)
  const [pipeline, setPipeline] = createSignal(idlePipeline)
  const [fromStage, setFromStage] = createSignal<PipelineStage | "">("")
  const [runError, setRunError] = createSignal("")
  const [autoScroll, setAutoScroll] = createSignal(true)
  const [clearedThroughId, setClearedThroughId] = createSignal(0)
  let logPanel: HTMLDivElement | undefined
  const busy = () => ["starting", "running", "stopping"].includes(pipeline().status)
  const visibleLogs = () => pipeline().logs.filter((entry) => entry.id > clearedThroughId())

  createEffect(() => {
    visibleLogs().length
    if (autoScroll()) queueMicrotask(() => {
      if (logPanel) logPanel.scrollTop = logPanel.scrollHeight
    })
  })

  onMount(() => {
    const unsubscribe = window.novelVoiceCast.onPipelineEvent((event) => {
      if (event.type === "state") setPipeline(event.state)
    })
    void window.novelVoiceCast.getPipelineState().then(setPipeline).catch((error: unknown) => {
      setRunError(error instanceof Error ? error.message : "无法读取流水线状态")
    })
    onCleanup(unsubscribe)
  })

  const storeSelection = (inputKind: InputKind, selection: TextFileSelection | null) => {
    setInputs((current) => applySelection(current, inputKind, selection))
  }

  const pick = async (inputKind: InputKind) => {
    if (busy()) return
    try {
      storeSelection(inputKind, await window.novelVoiceCast.pickTextFile(inputKind))
    } catch (error) {
      storeSelection(inputKind, {
        ok: false,
        error: error instanceof Error ? error.message : "文件选择失败",
      })
    }
  }

  const drop = async (event: DragEvent, inputKind: InputKind) => {
    event.preventDefault()
    setDragging(null)
    if (busy()) return
    const file = event.dataTransfer?.files.item(0)
    if (!file) return
    try {
      storeSelection(
        inputKind,
        await window.novelVoiceCast.acceptDroppedTextFile(file, inputKind),
      )
    } catch (error) {
      storeSelection(inputKind, {
        ok: false,
        error: error instanceof Error ? error.message : "拖放文件读取失败",
      })
    }
  }

  const start = async () => {
    const current = inputs()
    if (!current.novel.file || !current.labels.file || busy()) return
    setRunError("")
    try {
      const stage = fromStage()
      setPipeline(await window.novelVoiceCast.startPipeline({
        novelPath: current.novel.file.path,
        labelsPath: current.labels.file.path,
        ...(stage ? { fromStage: stage } : {}),
      }))
    } catch (error) {
      setRunError(error instanceof Error ? error.message : "流水线启动失败")
    }
  }

  const stop = async () => {
    setRunError("")
    try {
      setPipeline(await window.novelVoiceCast.stopPipeline())
    } catch (error) {
      setRunError(error instanceof Error ? error.message : "停止请求失败")
    }
  }

  const startLabel = () => {
    if (pipeline().status === "interrupted" || pipeline().status === "failed") return "继续运行"
    if (pipeline().status === "completed") return "重新运行"
    return "开始完整流程"
  }

  return (
    <main class="min-h-screen bg-slate-950 px-6 py-8 text-slate-100 lg:px-10">
      <section class="mx-auto max-w-6xl overflow-hidden rounded-3xl border border-white/10 bg-slate-900/85 shadow-2xl shadow-black/40">
        <header class="border-b border-white/10 px-8 py-7">
          <p class="text-xs font-semibold tracking-[0.28em] text-amber-300">NOVEL VOICE CAST</p>
          <h1 class="mt-2 text-3xl font-semibold tracking-tight">一键生成完整有声插图视频</h1>
          <p class="mt-3 max-w-3xl text-sm leading-6 text-slate-400">
            先选择小说原文和角色标注。两份文件都通过检查后，才能启动 13 阶段流水线。
          </p>
        </header>

        <div class="p-8">
          <section class="grid gap-5 md:grid-cols-2">
            <For each={["novel", "labels"] as InputKind[]}>
              {(inputKind) => {
                const slot = () => inputs()[inputKind]
                const copy = inputCopy[inputKind]
                return (
                  <article
                    class={`relative min-h-56 rounded-2xl border p-6 transition ${
                      dragging() === inputKind
                        ? "border-amber-300 bg-amber-300/10"
                        : slot().file
                          ? "border-emerald-400/50 bg-emerald-400/[0.06]"
                          : "border-dashed border-slate-600 bg-slate-950/45"
                    }`}
                    onDragOver={(event) => {
                      event.preventDefault()
                      if (!busy()) setDragging(inputKind)
                    }}
                    onDragLeave={() => setDragging(null)}
                    onDrop={(event) => void drop(event, inputKind)}
                  >
                    <div class="flex items-start justify-between gap-4">
                      <div>
                        <p class="text-xs font-medium uppercase tracking-[0.18em] text-slate-500">
                          必填输入
                        </p>
                        <h2 class="mt-2 text-lg font-medium">{copy.title}</h2>
                      </div>
                      <button
                        type="button"
                        disabled={busy()}
                        aria-label={`选择${copy.title}`}
                        class="grid size-11 shrink-0 place-items-center rounded-full border border-white/15 bg-white/[0.06] text-2xl text-amber-200 transition enabled:hover:border-amber-300 enabled:hover:bg-amber-300/10 disabled:opacity-40"
                        onClick={() => void pick(inputKind)}
                      >
                        +
                      </button>
                    </div>

                    <Show
                      when={slot().file}
                      fallback={
                        <div class="mt-9 text-center">
                          <p class="text-sm text-slate-300">拖入文件，或点击右上角“+”选择</p>
                          <p class="mt-2 text-xs text-slate-500">{copy.hint}</p>
                        </div>
                      }
                    >
                      {(file) => (
                        <div class="mt-6 rounded-xl border border-white/10 bg-black/20 p-4">
                          <div class="flex items-start justify-between gap-4">
                            <div class="min-w-0">
                              <p class="truncate text-sm font-medium text-emerald-200" title={file().name}>
                                {file().name}
                              </p>
                              <p class="mt-1 text-xs text-slate-500">
                                {formatFileSize(file().size)} · 已通过检查
                              </p>
                            </div>
                            <button
                              type="button"
                              disabled={busy()}
                              class="text-xs text-slate-400 underline decoration-slate-600 underline-offset-4 enabled:hover:text-white disabled:opacity-40"
                              onClick={() => setInputs((current) => clearSelection(current, inputKind))}
                            >
                              清除
                            </button>
                          </div>
                          <p class="mt-3 break-all font-mono text-[11px] leading-5 text-slate-500" title={file().path}>
                            {file().path}
                          </p>
                        </div>
                      )}
                    </Show>

                    <Show when={slot().error}>
                      <p role="alert" class="mt-4 rounded-lg bg-rose-400/10 px-3 py-2 text-xs text-rose-300">
                        {slot().error}
                      </p>
                    </Show>
                  </article>
                )
              }}
            </For>
          </section>

          <div class="mt-7 flex flex-col gap-4 rounded-2xl border border-white/10 bg-slate-950/45 p-5 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <p class="text-sm font-medium">
                {busy()
                  ? `流水线状态：${pipeline().status}`
                  : inputsReady(inputs())
                    ? "输入已就绪"
                    : "等待两份有效输入"}
              </p>
              <p class="mt-1 text-xs text-slate-500">
                运行期间不会改写原小说、角色标注或 config.yaml。
              </p>
            </div>
            <div class="flex flex-col gap-3 sm:flex-row sm:items-center">
              <label class="text-xs text-slate-400">
                断点起点
                <select
                  aria-label="断点续跑起点"
                  disabled={busy()}
                  value={fromStage()}
                  onChange={(event) => setFromStage(event.currentTarget.value as PipelineStage | "")}
                  class="ml-2 rounded-lg border border-white/10 bg-slate-900 px-3 py-2 text-xs text-slate-200"
                >
                  <option value="">自动检查全部缓存（推荐）</option>
                  <For each={PIPELINE_STAGES}>
                    {(stage) => <option value={stage}>{stageLabels[stage]}</option>}
                  </For>
                </select>
              </label>
              <Show
                when={busy()}
                fallback={
                  <button
                    type="button"
                    disabled={!inputsReady(inputs())}
                    class="rounded-xl bg-amber-300 px-6 py-3 text-sm font-semibold text-slate-950 transition enabled:hover:bg-amber-200 disabled:cursor-not-allowed disabled:bg-slate-700 disabled:text-slate-400"
                    onClick={() => void start()}
                  >
                    {startLabel()}
                  </button>
                }
              >
                <button
                  type="button"
                  disabled={pipeline().status === "stopping"}
                  class="rounded-xl bg-rose-400 px-6 py-3 text-sm font-semibold text-slate-950 transition enabled:hover:bg-rose-300 disabled:cursor-wait disabled:bg-slate-700 disabled:text-slate-400"
                  onClick={() => void stop()}
                >
                  {pipeline().status === "stopping" ? "正在停止…" : "停止并保留断点"}
                </button>
              </Show>
            </div>
          </div>

          <Show when={runError() || pipeline().error}>
            <p role="alert" class="mt-4 rounded-xl bg-rose-400/10 px-4 py-3 text-sm text-rose-300">
              {runError() || pipeline().error}
            </p>
          </Show>
          <Show when={pipeline().command}>
            <div class="mt-4 rounded-xl border border-white/10 bg-black/20 p-4">
              <p class="text-xs font-medium text-slate-400">当前命令</p>
              <p class="mt-2 break-all font-mono text-[11px] leading-5 text-slate-500">
                {pipeline().command}
              </p>
            </div>
          </Show>

          <Show when={pipeline().currentStage}>
            {(currentStage) => (
              <section class="mt-5 rounded-2xl border border-amber-300/25 bg-gradient-to-br from-amber-300/[0.08] to-slate-950/20 p-5">
                <div class="flex flex-wrap items-end justify-between gap-3">
                  <div>
                    <p class="text-xs font-semibold tracking-[0.18em] text-amber-300">当前阶段</p>
                    <h2 class="mt-2 text-lg font-medium">
                      {stageLabels[currentStage()]} · {pipeline().stagePercent.toFixed(1)}%
                    </h2>
                    <p class="mt-1 text-sm text-slate-400">{pipeline().operation || "正在处理"}</p>
                  </div>
                  <p class="font-mono text-xs text-slate-500">
                    {pipeline().currentStageIndex}/13
                  </p>
                </div>
                <div class="mt-4 h-2.5 overflow-hidden rounded-full bg-slate-800" aria-label="当前阶段进度">
                  <div
                    class="h-full rounded-full bg-gradient-to-r from-amber-400 to-yellow-200 transition-[width] duration-300"
                    style={{ width: `${pipeline().stagePercent}%` }}
                  />
                </div>
              </section>
            )}
          </Show>

          <section class="mt-5 rounded-2xl border border-white/10 bg-slate-950/45 p-5">
            <div class="flex flex-wrap items-center justify-between gap-3">
              <div>
                <h2 class="text-sm font-medium text-slate-200">实时日志</h2>
                <p class="mt-1 text-xs text-slate-500">内存仅保留最近 800 条；完整日志继续写入磁盘。</p>
              </div>
              <div class="flex gap-2">
                <button
                  type="button"
                  aria-pressed={autoScroll()}
                  class={`rounded-lg border px-3 py-2 text-xs transition ${
                    autoScroll() ? "border-amber-300/40 text-amber-200" : "border-white/10 text-slate-400"
                  }`}
                  onClick={() => setAutoScroll((value) => !value)}
                >
                  自动滚动：{autoScroll() ? "开" : "关"}
                </button>
                <button
                  type="button"
                  class="rounded-lg border border-white/10 px-3 py-2 text-xs text-slate-400 hover:text-white"
                  onClick={() => setClearedThroughId(pipeline().logs.at(-1)?.id ?? 0)}
                >
                  清空显示
                </button>
              </div>
            </div>
            <div
              ref={logPanel}
              class="mt-4 h-72 overflow-auto rounded-xl border border-white/[0.06] bg-black/35 p-3 font-mono text-[11px] leading-5"
            >
              <Show
                when={visibleLogs().length > 0}
                fallback={<p class="text-slate-600">等待流水线输出……</p>}
              >
                <For each={visibleLogs()}>
                  {(entry) => (
                    <p class={entry.level === "ERROR" ? "text-rose-300" : entry.level === "WARNING" ? "text-amber-200" : "text-slate-400"}>
                      <span class="text-slate-600">{entry.timestamp.slice(11, 19)}</span>
                      <span class="mx-2 text-slate-600">{entry.level}</span>
                      <Show when={entry.stage}><span class="mr-2 text-sky-300">[{entry.stage}]</span></Show>
                      {entry.message}
                    </p>
                  )}
                </For>
              </Show>
            </div>
          </section>
        </div>

        <footer class="border-t border-white/10 px-8 py-6">
          <ol class="grid grid-cols-2 gap-2 text-xs sm:grid-cols-4 lg:grid-cols-7">
            <For each={pipeline().stages}>
              {(stage) => (
                <li class={`rounded-xl border px-3 py-3 transition ${stageTone(stage.status)}`}>
                  <div class="flex items-center justify-between gap-2">
                    <span><span class="mr-2 opacity-50">{stage.index}</span>{stages[stage.index - 1]}</span>
                    <span class="text-[10px] opacity-75">{stageStatusLabels[stage.status]}</span>
                  </div>
                  <div class="mt-2 h-1 overflow-hidden rounded-full bg-black/25">
                    <div class="h-full bg-current opacity-70" style={{ width: `${stage.percent}%` }} />
                  </div>
                  <p class="mt-1 text-[10px] opacity-60">{stage.percent.toFixed(0)}%</p>
                </li>
              )}
            </For>
          </ol>
        </footer>
      </section>
    </main>
  )
}
