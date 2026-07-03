import axios from 'axios';

const API_BASE = 'http://127.0.0.1:5050/api';

export const api = {
  ping: () => axios.get(`${API_BASE}/ping`),
  
  upload: (file, application) => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('application', application);
    return axios.post(`${API_BASE}/upload`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    });
  },
  
  getResults: (runId) => axios.get(`${API_BASE}/results/${runId}`),
  
  getFigure: (runId, filename) => `${API_BASE}/fig/${runId}/${filename}`,
  
  getGraph: (runId, filename) => `${API_BASE}/graph/${runId}/${filename}`
};