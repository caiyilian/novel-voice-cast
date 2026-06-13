import { defineStore } from 'pinia'
import { projectApi, characterApi, chapterApi } from '../api'

export const useProjectStore = defineStore('project', {
  state: () => ({
    projects: [],
    currentProject: null,
    characters: [],
    chapters: [],
    loading: false,
    error: null,
  }),

  actions: {
    async fetchProjects() {
      this.loading = true
      try {
        const res = await projectApi.list()
        this.projects = res.data
      } catch (e) {
        this.error = e.message
      } finally {
        this.loading = false
      }
    },

    async fetchProject(id) {
      this.loading = true
      try {
        const res = await projectApi.get(id)
        this.currentProject = res.data
      } catch (e) {
        this.error = e.message
      } finally {
        this.loading = false
      }
    },

    async createProject(name) {
      const res = await projectApi.create(name)
      this.projects.push(res.data)
      return res.data
    },

    async fetchCharacters(projectId) {
      this.loading = true
      try {
        const res = await characterApi.list(projectId)
        this.characters = res.data.characters
      } catch (e) {
        this.error = e.message
      } finally {
        this.loading = false
      }
    },

    async fetchChapters(projectId) {
      this.loading = true
      try {
        const res = await chapterApi.list(projectId)
        this.chapters = res.data.chapters
      } catch (e) {
        this.error = e.message
      } finally {
        this.loading = false
      }
    },
  },
})
