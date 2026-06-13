<template>
  <n-config-provider :theme-overrides="themeOverrides">
    <n-layout has-sider style="height: 100vh">
      <n-layout-sider bordered :width="200">
        <n-menu :options="menuOptions" :value="currentRoute" @update:value="handleMenuClick" />
      </n-layout-sider>
      <n-layout>
        <n-layout-header bordered style="padding: 12px 24px">
          <n-h2 style="margin: 0">Novel Voice Cast</n-h2>
        </n-layout-header>
        <n-layout-content style="padding: 24px">
          <router-view />
        </n-layout-content>
      </n-layout>
    </n-layout>
  </n-config-provider>
</template>

<script setup>
import { computed } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { NConfigProvider, NLayout, NLayoutSider, NLayoutHeader, NLayoutContent, NMenu, NH2 } from 'naive-ui'

const router = useRouter()
const route = useRoute()

const themeOverrides = {
  common: {
    primaryColor: '#18a058'
  }
}

const menuOptions = [
  {
    label: 'Projects',
    key: '/projects',
    icon: () => '📁'
  },
  {
    label: 'Characters',
    key: '/characters',
    icon: () => '🎭'
  },
  {
    label: 'Synthesis',
    key: '/synthesis',
    icon: () => '🔊'
  }
]

const currentRoute = computed(() => route.path)

const handleMenuClick = (key) => {
  if (key === '/projects') {
    router.push('/projects')
  } else if (key === '/characters' && route.params.id) {
    router.push(`/projects/${route.params.id}/characters`)
  } else if (key === '/synthesis' && route.params.id) {
    router.push(`/projects/${route.params.id}/synthesis`)
  }
}
</script>

<style>
body {
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}
</style>
