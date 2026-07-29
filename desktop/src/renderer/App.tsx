import { createSignal, For, Show } from "solid-js"
import type { InputKind, TextFileSelection } from "../preload/types"
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

export function App() {
  const [inputs, setInputs] = createSignal(emptyInputState())
  const [dragging, setDragging] = createSignal<InputKind | null>(null)

  const storeSelection = (inputKind: InputKind, selection: TextFileSelection | null) => {
    setInputs((current) => applySelection(current, inputKind, selection))
  }

  const pick = async (inputKind: InputKind) => {
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
                      setDragging(inputKind)
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
                        aria-label={`选择${copy.title}`}
                        class="grid size-11 shrink-0 place-items-center rounded-full border border-white/15 bg-white/[0.06] text-2xl text-amber-200 transition hover:border-amber-300 hover:bg-amber-300/10"
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
                              <p class="mt-1 text-xs text-slate-500">{formatFileSize(file().size)} · 已通过检查</p>
                            </div>
                            <button
                              type="button"
                              class="text-xs text-slate-400 underline decoration-slate-600 underline-offset-4 hover:text-white"
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
              <p class="text-sm font-medium">{inputsReady(inputs()) ? "输入已就绪" : "等待两份有效输入"}</p>
              <p class="mt-1 text-xs text-slate-500">运行期间不会改写原小说、角色标注或 config.yaml。</p>
            </div>
            <button
              type="button"
              disabled={!inputsReady(inputs())}
              class="rounded-xl bg-amber-300 px-6 py-3 text-sm font-semibold text-slate-950 transition enabled:hover:bg-amber-200 disabled:cursor-not-allowed disabled:bg-slate-700 disabled:text-slate-400"
            >
              开始完整流程
            </button>
          </div>
        </div>

        <footer class="border-t border-white/10 px-8 py-6">
          <ol class="grid grid-cols-2 gap-2 text-xs text-slate-500 sm:grid-cols-4 lg:grid-cols-7">
            <For each={stages}>
              {(stage, index) => (
                <li class="rounded-lg bg-white/[0.035] px-3 py-2">
                  <span class="mr-2 text-slate-600">{index() + 1}</span>{stage}
                </li>
              )}
            </For>
          </ol>
        </footer>
      </section>
    </main>
  )
}
