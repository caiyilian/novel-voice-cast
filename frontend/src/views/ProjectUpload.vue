<template>
  <div>
    <n-page-header @back="$router.push('/projects')" title="Upload Novel" />

    <n-card title="Upload Novel Text">
      <n-upload
        :max="1"
        accept=".txt"
        :custom-request="handleUpload"
        :show-file-list="false"
      >
        <n-upload-dragger>
          <n-text depth="3" style="font-size: 16px">
            Click or drag a .txt file here
          </n-text>
          <n-text depth="3" style="font-size: 12px; display: block; margin-top: 8px">
            Upload a labeled novel text file (with 「」 dialogue markers)
          </n-text>
        </n-upload-dragger>
      </n-upload>
    </n-card>

    <n-card v-if="result" title="Parsing Results" style="margin-top: 16px">
      <n-space vertical>
        <n-statistic label="Characters" :value="result.characters" />
        <n-statistic label="Dialogues" :value="result.dialogues" />
        <n-statistic label="Chapters" :value="result.chapters" />
        <n-statistic label="Method" :value="result.chapter_method" />

        <n-button type="primary" @click="goToCharacters">
          Enter Character Management
        </n-button>
      </n-space>
    </n-card>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { NPageHeader, NCard, NUpload, NUploadDragger, NText, NStatistic, NSpace, NButton } from 'naive-ui'
import { projectApi } from '../api'

const route = useRoute()
const router = useRouter()
const id = computed(() => route.params.id)

const result = ref(null)
const uploading = ref(false)

const handleUpload = async ({ file }) => {
  uploading.value = true
  try {
    const res = await projectApi.upload(id.value, file.file)
    result.value = res.data
  } catch (e) {
    console.error(e)
  } finally {
    uploading.value = false
  }
}

const goToCharacters = () => {
  router.push(`/projects/${id.value}/characters`)
}
</script>
