<template>
  <div>
    <n-page-header @back="$router.push(`/projects/${id}`)" title="Characters" />

    <n-spin :show="store.loading">
      <n-empty v-if="!store.characters.length" description="No characters found" />
      <n-grid :cols="3" :x-gap="12" :y-gap="12" v-else>
        <n-grid-item v-for="c in store.characters" :key="c.id">
          <n-card :title="c.name" size="small">
            <n-space vertical>
              <n-text>Gender: {{ c.gender }}</n-text>
              <n-text>Dialogues: {{ c.dialogue_count }}</n-text>
              <n-button size="small" @click="viewDialogues(c.name)">View Dialogues</n-button>
            </n-space>
          </n-card>
        </n-grid-item>
      </n-grid>
    </n-spin>

    <n-modal v-model:show="showDialogues" preset="card" :title="`${selectedCharacter} Dialogues`" style="width: 600px">
      <n-list bordered>
        <n-list-item v-for="d in dialogues" :key="d.id">
          <n-thing :title="`[${d.speaker}]`" :description="d.text" />
        </n-list-item>
      </n-list>
    </n-modal>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { useRoute } from 'vue-router'
import { NPageHeader, NSpin, NEmpty, NGrid, NGridItem, NCard, NSpace, NText, NButton, NModal, NList, NListItem, NThing } from 'naive-ui'
import { useProjectStore } from '../stores/project'
import { characterApi } from '../api'

const route = useRoute()
const store = useProjectStore()
const id = computed(() => route.params.id)

const showDialogues = ref(false)
const selectedCharacter = ref('')
const dialogues = ref([])

onMounted(() => store.fetchCharacters(id.value))

const viewDialogues = async (name) => {
  selectedCharacter.value = name
  showDialogues.value = true
  const res = await characterApi.dialogues(id.value, name)
  dialogues.value = res.data.dialogues
}
</script>
