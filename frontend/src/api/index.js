import axios from 'axios'

const api = axios.create({
  baseURL: 'http://localhost:8000',
  timeout: 30000,
})

// Project APIs
export const projectApi = {
  list: () => api.get('/api/project'),
  create: (name) => api.post('/api/project', { name }),
  get: (id) => api.get(`/api/project/${id}`),
  upload: (id, file) => {
    const formData = new FormData()
    formData.append('file', file)
    return api.post(`/api/project/${id}/upload`, formData)
  },
}

// Character APIs
export const characterApi = {
  list: (projectId) => api.get(`/api/project/${projectId}/characters`),
  get: (projectId, name) => api.get(`/api/project/${projectId}/characters/${name}`),
  update: (projectId, name, data) => api.patch(`/api/project/${projectId}/characters/${name}`, data),
  dialogues: (projectId, name, limit) => {
    const params = limit ? { limit } : {}
    return api.get(`/api/project/${projectId}/characters/${name}/dialogues`, { params })
  },
  identifyGenders: (projectId, file) => {
    const formData = new FormData()
    formData.append('file', file)
    return api.post(`/api/project/${projectId}/characters/identify-genders`, formData)
  },
}

// Audio APIs
export const audioApi = {
  upload: (projectId, characterName, file) => {
    const formData = new FormData()
    formData.append('file', file)
    return api.post(`/api/project/${projectId}/audio/upload?character_name=${characterName}`, formData)
  },
  getInfo: (audioId) => api.get(`/api/audio/${audioId}`),
  download: (audioId) => api.get(`/api/audio/${audioId}/download`, { responseType: 'blob' }),
  delete: (projectId, audioId) => api.delete(`/api/project/${projectId}/audio/${audioId}`),
}

// Preset APIs
export const presetApi = {
  list: (projectId) => api.get(`/api/project/${projectId}/presets`),
  download: (projectId, name) => api.get(`/api/project/${projectId}/presets/${name}/download`, { responseType: 'blob' }),
  audioStatus: (projectId) => api.get(`/api/project/${projectId}/audio-status`),
}

// Emotion APIs
export const emotionApi = {
  analyze: (projectId, file, limit) => {
    const formData = new FormData()
    formData.append('file', file)
    return api.post(`/api/project/${projectId}/emotions/analyze?limit=${limit || ''}`, formData)
  },
  status: (projectId) => api.get(`/api/project/${projectId}/emotions/status`),
}

// Preview API
export const previewApi = {
  preview: (projectId, text, characterName) =>
    api.post(`/api/project/${projectId}/preview`, { text, character_name: characterName }),
}

// Synthesis APIs
export const synthesisApi = {
  start: (projectId, characterVoiceMap) =>
    api.post(`/api/project/${projectId}/synthesize`, { character_voice_map: characterVoiceMap }),
  status: (projectId) => api.get(`/api/project/${projectId}/synthesis/status`),
  cancel: (projectId) => api.delete(`/api/project/${projectId}/synthesis`),
  output: (projectId) => api.get(`/api/project/${projectId}/output`, { responseType: 'blob' }),
}

// Chapter APIs
export const chapterApi = {
  list: (projectId) => api.get(`/api/project/${projectId}/chapters`),
}

// WebSocket
export const createWebSocket = (projectId) => {
  return new WebSocket(`ws://localhost:8000/ws/${projectId}`)
}

export default api
