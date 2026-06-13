import { createRouter, createWebHistory } from 'vue-router'

const routes = [
  {
    path: '/',
    redirect: '/projects'
  },
  {
    path: '/projects',
    name: 'Projects',
    component: () => import('../views/ProjectList.vue')
  },
  {
    path: '/projects/:id',
    name: 'ProjectDetail',
    component: () => import('../views/ProjectDetail.vue')
  },
  {
    path: '/projects/:id/upload',
    name: 'ProjectUpload',
    component: () => import('../views/ProjectUpload.vue')
  },
  {
    path: '/projects/:id/characters',
    name: 'Characters',
    component: () => import('../views/Characters.vue')
  },
  {
    path: '/projects/:id/synthesis',
    name: 'Synthesis',
    component: () => import('../views/Synthesis.vue')
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

export default router
