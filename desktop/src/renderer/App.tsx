export const stages = [
  "解析",
  "性别",
  "情绪",
  "表演",
  "语音",
  "拼接",
  "BGM 分割",
  "BGM 标注",
  "BGM 生成",
  "BGM 混音",
  "插图规划",
  "插图生成",
  "视频",
]

export function App() {
  return (
    <main class="min-h-screen bg-slate-950 px-8 py-10 text-slate-100">
      <section class="mx-auto max-w-6xl overflow-hidden rounded-3xl border border-white/10 bg-slate-900/85 shadow-2xl shadow-black/40">
        <header class="border-b border-white/10 px-8 py-7">
          <p class="text-xs font-semibold tracking-[0.28em] text-amber-300">NOVEL VOICE CAST</p>
          <h1 class="mt-2 text-3xl font-semibold tracking-tight">把小说变成完整的有声插图视频</h1>
          <p class="mt-3 max-w-3xl text-sm leading-6 text-slate-400">
            桌面控制台已就绪。下一步将接入小说与角色标注选择、13 阶段实时进度、停止与断点继续。
          </p>
        </header>

        <div class="grid gap-6 p-8 lg:grid-cols-[1.35fr_0.65fr]">
          <section class="rounded-2xl border border-dashed border-slate-600 bg-slate-950/45 p-6">
            <h2 class="text-lg font-medium">运行输入</h2>
            <div class="mt-5 grid gap-4 sm:grid-cols-2">
              <div class="rounded-xl border border-white/10 bg-white/5 p-5">
                <p class="text-sm font-medium">小说原文</p>
                <p class="mt-2 text-xs text-slate-500">等待选择 novel.txt</p>
              </div>
              <div class="rounded-xl border border-white/10 bg-white/5 p-5">
                <p class="text-sm font-medium">角色标注</p>
                <p class="mt-2 text-xs text-slate-500">等待选择 labels.txt</p>
              </div>
            </div>
          </section>

          <aside class="rounded-2xl border border-white/10 bg-slate-950/45 p-6">
            <p class="text-sm font-medium">安全运行环境</p>
            <dl class="mt-4 space-y-3 text-xs text-slate-400">
              <div class="flex justify-between gap-4"><dt>渲染进程隔离</dt><dd class="text-emerald-300">已启用</dd></div>
              <div class="flex justify-between gap-4"><dt>Node 集成</dt><dd class="text-emerald-300">已禁用</dd></div>
              <div class="flex justify-between gap-4"><dt>Electron</dt><dd>{window.novelVoiceCast?.versions.electron ?? "测试环境"}</dd></div>
            </dl>
          </aside>
        </div>

        <footer class="border-t border-white/10 px-8 py-6">
          <ol class="grid grid-cols-2 gap-2 text-xs text-slate-500 sm:grid-cols-4 lg:grid-cols-7">
            {stages.map((stage, index) => (
              <li class="rounded-lg bg-white/[0.035] px-3 py-2">
                <span class="mr-2 text-slate-600">{index + 1}</span>{stage}
              </li>
            ))}
          </ol>
        </footer>
      </section>
    </main>
  )
}
