<template>
  <div>
    <n-card title="Projects">
      <template #header-extra>
        <n-button type="primary" @click="showCreate = true">New Project</n-button>
      </template>

      <n-spin :show="store.loading">
        <n-empty v-if="!store.projects.length" description="No projects yet. Click 'New Project' to create one." />
        <n-list v-else bordered>
          <n-list-item v-for="p in store.projects" :key="p.id">
            <n-thing
              :title="p.name"
              :description="`Status: ${p.status} | Created: ${formatDate(p.created_at)}`"
              @click="goToProject(p.id)"
              style="cursor: pointer"
            />
          </n-list-item>
        </n-list>
      </n-spin>
    </n-card>

    <n-modal v-model:show="showCreate" preset="dialog" title="New Project">
      <n-space vertical>
        <n-input v-model:value="newName" placeholder="Enter project name" />
        <n-button type="primary" @click="createProject" :disabled="!newName.trim()">
          Create
        </n-button>
      </n-space>
    </n-modal>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { NCard, NButton, NSpin, NEmpty, NList, NListItem, NThing, NModal, NInput, NSpace } from 'naive-ui'
import { useProjectStore } from '../stores/project'

const router = useRouter()
const store = useProjectStore()
const showCreate = ref(false)
const newName = ref('')

onMounted(() => store.fetchProjects())

const formatDate = (dateStr) => {
  if (!dateStr) return ''
  return new Date(dateStr).toLocaleDateString()
}

const goToProject = (id) => {
  router.push(`/projects/${id}`)
}

const createProject = async () => {
  if (newName.value.trim()) {
    const project = await store.createProject(newName.value.trim())
    showCreate.value = false
    newName.value = ''
    router.push(`/projects/${project.id}/upload`)
  }
}
</script>
