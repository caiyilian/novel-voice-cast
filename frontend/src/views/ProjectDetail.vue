<template>
  <div>
    <n-page-header @back="$router.push('/projects')" :title="project?.name || 'Loading...'">
      <template #extra>
        <n-space>
          <n-button @click="$router.push(`/projects/${id}/upload`)">Upload Novel</n-button>
          <n-button @click="$router.push(`/projects/${id}/characters`)">Characters</n-button>
          <n-button @click="$router.push(`/projects/${id}/synthesis`)">Synthesis</n-button>
        </n-space>
      </template>
    </n-page-header>

    <n-spin :show="store.loading">
      <n-card v-if="project" :title="`Project: ${project.name}`">
        <n-descriptions bordered :column="2">
          <n-descriptions-item label="Status">{{ project.status }}</n-descriptions-item>
          <n-descriptions-item label="Characters">{{ project.character_count }}</n-descriptions-item>
          <n-descriptions-item label="Dialogues">{{ project.dialogue_count }}</n-descriptions-item>
          <n-descriptions-item label="Chapters">{{ project.chapter_count }}</n-descriptions-item>
        </n-descriptions>
      </n-card>
    </n-spin>
  </div>
</template>

<script setup>
import { computed, onMounted } from 'vue'
import { useRoute } from 'vue-router'
import { NPageHeader, NButton, NSpace, NSpin, NCard, NDescriptions, NDescriptionsItem } from 'naive-ui'
import { useProjectStore } from '../stores/project'

const route = useRoute()
const store = useProjectStore()
const id = computed(() => route.params.id)
const project = computed(() => store.currentProject)

onMounted(() => store.fetchProject(id.value))
</script>
