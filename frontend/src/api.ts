import axios from 'axios';

const API_BASE = 'http://localhost:8000';

export interface StartJobRequest {
    faculty_file: string;
    student_file: string;
    textbook_file: string;
}

export const api = {
    uploadFile: async (file: File) => {
        const formData = new FormData();
        formData.append('file', file);
        const res = await axios.post(`${API_BASE}/upload`, formData, {
            headers: { 'Content-Type': 'multipart/form-data' }
        });
        return res.data; // { filename: ... }
    },

    startJob: async (req: StartJobRequest) => {
        const res = await axios.post(`${API_BASE}/job/start`, req);
        return res.data; // { job_id: ... }
    },

    getResults: async (jobId: string) => {
        const res = await axios.get(`${API_BASE}/results/${jobId}`);
        return res.data;
    }
};
