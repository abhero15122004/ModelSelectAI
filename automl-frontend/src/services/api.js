import axios from "axios";

// ✅ Use your local backend URL
const API_BASE = "http://localhost:5050/api";

export const api = {
  ping: () => axios.get(`${API_BASE}/ping`),

  upload: async (file, application, onProgress) => {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("application", application);

    const response = await axios.post(`${API_BASE}/upload`, formData, {
      headers: { "Content-Type": "multipart/form-data" },
      maxBodyLength: Infinity,
      onUploadProgress: (progressEvent) => {
        if (progressEvent.total) {
          const percent = (progressEvent.loaded / progressEvent.total) * 100;
          if (onProgress) onProgress(percent);
        }
      },
    });
    return response.data;
  },

  getResults: (runId) => axios.get(`${API_BASE}/results/${runId}`),

  getFigure: (runId, filename) => `${API_BASE}/fig/${runId}/${filename}`,

  getGraph: (runId, filename) => `${API_BASE}/graph/${runId}/${filename}`,
};