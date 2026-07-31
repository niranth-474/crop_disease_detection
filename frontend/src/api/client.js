import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api'

const apiClient = axios.create({
  baseURL: API_BASE_URL,
})

export const predictImage = async (file) => {
  const formData = new FormData()
  formData.append('file', file)
  return apiClient.post('/predict', formData)
}

export default apiClient
