
import axios from 'axios';

const API_base = 'http://localhost:8001/api';

export const api = {
    // === Questions ===
    getQuestions: async () => {
        const res = await axios.get(`${API_base}/questions`);
        return res.data;
    },

    addQuestion: async (data: { question: string; answer: string; marks: number }) => {
        const res = await axios.post(`${API_base}/questions`, data);
        return res.data;
    },

    // === Grading ===
    gradeAnswer: async (data: {
        facultyQuestionId: number;
        studentAnswerText: string;
        allowRag2: boolean;
    }) => {
        const res = await axios.post(`${API_base}/grade`, data);
        return res.data;
    },

    // Legacy support (optional)
    uploadFile: async (file: File) => {
        const formData = new FormData();
        formData.append('file', file);
        const res = await axios.post('http://localhost:8001/upload', formData, {
            headers: { 'Content-Type': 'multipart/form-data' }
        });
        return res.data;
    },

    // Bulk Uploads
    uploadQuestionsCsv: async (file: File) => {
        const formData = new FormData();
        formData.append('file', file);
        const res = await axios.post(`${API_base}/upload/questions`, formData, {
            headers: { 'Content-Type': 'multipart/form-data' }
        });
        return res.data;
    },

    batchGradeCsv: async (file: File) => {
        const formData = new FormData();
        formData.append('file', file);
        const res = await axios.post(`${API_base}/upload/batch-grade`, formData, {
            headers: { 'Content-Type': 'multipart/form-data' }
        });
        return res.data;
    }
};

export interface Question {
    id: number;
    question: string;
    answer: string;
    marks: number;
}
