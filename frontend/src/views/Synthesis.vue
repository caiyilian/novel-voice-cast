<template>
  <div>
    <n-page-header @back="$router.push(`/projects/${id}`)" title="Synthesis" />

    <n-card title="Synthesis Control">
      <n-space vertical>
        <n-button type="primary" :loading="synthesizing" @click="startSynthesis">
          {{ synthesizing ? 'Synthesizing...' : 'Start Synthesis' }}
        </n-button>

        <n-progress
          v-if="synthesisStatus.status === 'synthesizing'"
          type="line"
          :percentage="progress"
          :indicator-placement="'inside'"
          status="info"
        />

        <n-alert v-if="synthesisStatus.status === 'done'" type="success">
          Synthesis complete! <n-button text @click="downloadOutput">Download</n-button>
        </n-alert>

        <n-alert v-if="synthesisStatus.status === 'failed'" type="error">
          Synthesis failed: {{ synthesisStatus.message }}
        </n-alert>

        <n-text v-if="synthesisStatus.status === 'idle'">Ready to synthesize</n-text>
      </n-space>
    </n-card>

    <n-card title="Status" v-if="synthesisStatus.status !== 'idle'">
      <n-descriptions bordered :column="2">
        <n-descriptions-item label="Status">{{ synthesisStatus.status }}</n-descriptions-item>
        <n-descriptions-item label="Progress">{{ synthesisStatus.current }}/{{ synthesisStatus.total }}</n-descriptions-item>
        <n-descriptions-item label="Message">{{ synthesisStatus.message }}</n-descriptions-item>
      </n-descriptions>
    </n-card>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useRoute } from 'vue-router'
import { NPageHeader, NCard, NSpace, NButton, NProgress, NAlert, NText } from 'naive-ui'
import { synthesisApi } from '../api'

const route = useRoute()
const id = computed(() => route.params.id)

const synthesizing = ref(false)
const synthesisStatus = ref({ status: 'idle', current: 0, total: 0, message: '' })

const progress = computed(() => {
  if (synthesisStatus.value.total === 0) return 0
  return Math.round((synthesisStatus.value.current / synthesisStatus.value.total) * 100)
})

let pollInterval = null

onMounted(async () => {
  const res = await synthesisApi.status(id.value)
  synthesisStatus.value = res.data
})

const startSynthesis = async () => {
  synthesizing.value = true
  try {
    await synthesisApi.start(id.value, {})
    pollStatus()
  } catch (e) {
    console.error(e)
    synthesizing.value = false
  }
}

const pollStatus = () => {
  pollInterval = setInterval(async () => {
    const res = await synthesisApi.status(id.value)
    synthesisStatus.value = res.data
    if (res.data.status !== 'synthesizing') {
      clearInterval(pollInterval)
      synthesizing.value = false
    }
  }, 1000)
}

const downloadOutput = async () => {
  const res = await synthesisApi.output(id.value)
  const url = window.URL.createObjectURL(new Blob([res.data]))
  const a = document.createElement('a')
  a.href = url
  a.download = 'synthesis.wav'
  a.click()
  window.URL.revokeObjectURL(url)
}

onUnmounted(() => {
  if (pollInterval) clearInterval(pollInterval)
})
</script>
