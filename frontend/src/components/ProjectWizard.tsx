
import { useState } from 'react';
import { Upload, CheckCircle, FileText, Play } from 'lucide-react';
import { api } from '../api';

interface StartJobRequest {
    faculty_file: string;
    student_file: string;
    textbook_file: string;
}

interface Props {
    onStart: (req: StartJobRequest) => void;
}

export function ProjectWizard({ onStart }: Props) {
    const [faculty, setFaculty] = useState<File | null>(null);
    const [student, setStudent] = useState<File | null>(null);
    const [textbook, setTextbook] = useState<File | null>(null);
    const [uploading, setUploading] = useState(false);

    const handleStart = async () => {
        if (!faculty || !student || !textbook) return;
        setUploading(true);
        try {
            const fRes = await api.uploadFile(faculty);
            const sRes = await api.uploadFile(student);
            const tRes = await api.uploadFile(textbook);

            onStart({
                faculty_file: fRes.filename,
                student_file: sRes.filename,
                textbook_file: tRes.filename
            });
        } catch (e) {
            alert("Upload failed: " + e);
            setUploading(false);
        }
    };

    const FileRow = ({ label, file, setFile, accept }: any) => (
        <div className="flex items-center justify-between p-4 bg-white border rounded-lg shadow-sm">
            <div className="flex items-center gap-3">
                <div className="p-2 bg-blue-50 text-blue-600 rounded-full">
                    {file ? <CheckCircle size={20} className="text-green-500" /> : <FileText size={20} />}
                </div>
                <div>
                    <h3 className="font-medium text-slate-900">{label}</h3>
                    <p className="text-sm text-slate-500">{file ? file.name : "No file selected"}</p>
                </div>
            </div>
            <label className="cursor-pointer px-4 py-2 text-sm font-medium text-blue-600 hover:bg-blue-50 rounded-md transition border border-blue-200">
                Choose File
                <input
                    type="file"
                    className="hidden"
                    accept={accept}
                    onChange={(e) => setFile(e.target.files?.[0] || null)}
                />
            </label>
        </div>
    );

    return (
        <div className="max-w-2xl mx-auto py-12 px-4">
            <h1 className="text-3xl font-bold text-slate-900 mb-2">Create New Grading Job</h1>
            <p className="text-slate-500 mb-8">Upload the required files to initialize the AI grader.</p>

            <div className="space-y-4 mb-8">
                <FileRow label="Faculty Answer Key (CSV)" file={faculty} setFile={setFaculty} accept=".csv" />
                <FileRow label="Student Answers (CSV)" file={student} setFile={setStudent} accept=".csv" />
                <FileRow label="Textbook Source (PDF)" file={textbook} setFile={setTextbook} accept=".pdf" />
            </div>

            <button
                onClick={handleStart}
                disabled={!faculty || !student || !textbook || uploading}
                className="w-full flex items-center justify-center gap-2 py-3 px-6 bg-blue-600 hover:bg-blue-700 text-white font-semibold rounded-lg shadow-lg disabled:opacity-50 disabled:cursor-not-allowed transition-all transform hover:scale-[1.01]"
            >
                {uploading ? "Uploading..." : <><Play size={18} /> Start Grading Engine</>}
            </button>
        </div>
    );
}
