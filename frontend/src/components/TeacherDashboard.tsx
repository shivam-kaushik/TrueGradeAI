
import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { api, type Question } from '../api';
import { Plus, BookOpen, Trash2, Save, FileText, Upload } from 'lucide-react';
import clsx from 'clsx';

function UploadCsv({ onSuccess }: { onSuccess: () => void }) {
    const [file, setFile] = useState<File | null>(null);
    const mutation = useMutation({
        mutationFn: api.uploadQuestionsCsv,
        onSuccess: () => {
            alert("Questions imported successfully!");
            onSuccess();
        },
        onError: () => alert("Import failed.")
    });

    return (
        <div className="bg-white border rounded-xl shadow-sm p-8 max-w-2xl mx-auto text-center">
            <div className="mb-6">
                <div className="w-16 h-16 bg-blue-50 text-blue-600 rounded-full flex items-center justify-center mx-auto mb-4">
                    <Upload size={32} />
                </div>
                <h2 className="text-xl font-bold text-slate-900">Import Questions via CSV</h2>
                <p className="text-slate-500 mt-2">Upload your `faculty_key.csv` file to bulk add questions.</p>
            </div>

            <div className="border-2 border-dashed border-slate-300 rounded-xl p-8 mb-6 hover:bg-slate-50 transition-colors">
                <input
                    type="file"
                    accept=".csv"
                    onChange={e => setFile(e.target.files?.[0] || null)}
                    className="block w-full text-sm text-slate-500
                      file:mr-4 file:py-2 file:px-4
                      file:rounded-full file:border-0
                      file:text-sm file:font-semibold
                      file:bg-blue-50 file:text-blue-700
                      hover:file:bg-blue-100
                    "
                />
            </div>

            <button
                onClick={() => file && mutation.mutate(file)}
                disabled={!file || mutation.isPending}
                className="w-full bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-lg font-bold shadow-md transition-all disabled:opacity-50"
            >
                {mutation.isPending ? "Importing..." : "Upload Questions"}
            </button>
        </div>
    )
}

export function TeacherDashboard() {
    const [activeTab, setActiveTab] = useState<'list' | 'add' | 'import'>('list');
    const queryClient = useQueryClient();

    const { data: questions, isLoading } = useQuery({
        queryKey: ['questions'],
        queryFn: api.getQuestions
    });

    return (
        <div className="max-w-5xl mx-auto py-12 px-4">
            <header className="mb-10">
                <h1 className="text-3xl font-bold text-slate-900">Teacher Dashboard</h1>
                <p className="text-slate-500">Manage curriculum and answer keys.</p>
            </header>

            <div className="flex gap-4 mb-8 border-b border-slate-200">
                <TabButton active={activeTab === 'list'} onClick={() => setActiveTab('list')} icon={<BookOpen size={18} />}>
                    Question Bank
                </TabButton>
                <TabButton active={activeTab === 'add'} onClick={() => setActiveTab('add')} icon={<Plus size={18} />}>
                    Add New Question
                </TabButton>
                <TabButton active={activeTab === 'import'} onClick={() => setActiveTab('import')} icon={<Upload size={18} />}>
                    Bulk Import (CSV)
                </TabButton>
            </div>

            {activeTab === 'list' && (
                <div className="space-y-4 animate-in fade-in slide-in-from-bottom-2 duration-300">
                    {isLoading ? (
                        <div className="text-center py-10 text-slate-400">Loading questions...</div>
                    ) : (
                        questions?.map((q: Question) => (
                            <QuestionCard key={q.id} question={q} />
                        ))
                    )}
                    {questions?.length === 0 && (
                        <div className="text-center py-20 bg-slate-50 rounded-xl border border-dashed border-slate-300">
                            <p className="text-slate-400">No questions added yet.</p>
                            <button onClick={() => setActiveTab('add')} className="text-blue-600 font-medium mt-2 hover:underline">Add your first question</button>
                        </div>
                    )}
                </div>
            )}

            {activeTab === 'add' && <AddQuestionForm onSuccess={() => {
                queryClient.invalidateQueries({ queryKey: ['questions'] });
                setActiveTab('list');
            }} />}

            {activeTab === 'import' && <UploadCsv onSuccess={() => {
                queryClient.invalidateQueries({ queryKey: ['questions'] });
                setActiveTab('list');
            }} />}
        </div>
    );
}

function TabButton({ children, active, onClick, icon }: any) {
    return (
        <button
            onClick={onClick}
            className={clsx(
                "flex items-center gap-2 px-4 py-3 font-medium text-sm transition-all border-b-2 -mb-[2px]",
                active ? "border-blue-600 text-blue-600 bg-blue-50/50" : "border-transparent text-slate-500 hover:text-slate-700 hover:bg-slate-50"
            )}
        >
            {icon}
            {children}
        </button>
    )
}

function QuestionCard({ question }: { question: Question }) {
    const [expanded, setExpanded] = useState(false);

    return (
        <div className="bg-white border rounded-lg shadow-sm hover:shadow-md transition-shadow overflow-hidden">
            <div className="p-5 flex justify-between items-start">
                <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1">
                        <h3 className="font-semibold text-lg text-slate-900">{question.question}</h3>
                        <span className="bg-slate-100 text-slate-600 px-2 py-0.5 rounded text-xs font-mono font-medium">
                            {question.marks} marks
                        </span>
                    </div>
                    <button
                        onClick={() => setExpanded(!expanded)}
                        className="text-sm text-blue-600 hover:text-blue-700 font-medium"
                    >
                        {expanded ? "Hide Answer Key" : "View Answer Key"}
                    </button>
                </div>
                <button className="text-slate-400 hover:text-red-500 transition-colors p-2">
                    <Trash2 size={18} />
                </button>
            </div>
            {expanded && (
                <div className="bg-slate-50 p-5 border-t text-sm font-mono text-slate-700 leading-relaxed">
                    <strong className="block text-xs uppercase text-slate-400 mb-2 font-sans tracking-wider">Faculty Answer Key</strong>
                    {question.answer}
                </div>
            )}
        </div>
    )
}

function AddQuestionForm({ onSuccess }: { onSuccess: () => void }) {
    const [marks, setMarks] = useState(5.0);
    const [qText, setQText] = useState("");
    const [aText, setAText] = useState("");

    const mutation = useMutation({
        mutationFn: api.addQuestion,
        onSuccess: () => {
            onSuccess();
        },
        onError: () => alert("Failed to add question")
    });

    return (
        <div className="bg-white border rounded-xl shadow-sm p-8 max-w-3xl animate-in fade-in zoom-in-95 duration-300">
            <h2 className="text-xl font-bold mb-6 flex items-center gap-2">
                <FileText className="text-blue-500" />
                New Question Details
            </h2>
            <div className="space-y-6">
                <div>
                    <label className="block text-sm font-medium text-slate-700 mb-2">Question Text</label>
                    <input
                        className="w-full p-3 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-all"
                        placeholder="e.g. Explain the impact of the Industrial Revolution."
                        value={qText}
                        onChange={e => setQText(e.target.value)}
                    />
                </div>

                <div className="grid grid-cols-2 gap-6">
                    <div>
                        <label className="block text-sm font-medium text-slate-700 mb-2">Max Marks</label>
                        <input
                            type="number"
                            step="0.5"
                            className="w-full p-3 border rounded-lg focus:ring-2 focus:ring-blue-500 outline-none"
                            value={marks}
                            onChange={e => setMarks(parseFloat(e.target.value))}
                        />
                    </div>
                </div>

                <div>
                    <label className="block text-sm font-medium text-slate-700 mb-2">
                        Official Answer Key
                        <span className="ml-2 text-xs font-normal text-slate-400">The AI will use this to generate key points.</span>
                    </label>
                    <textarea
                        className="w-full p-3 border rounded-lg h-40 focus:ring-2 focus:ring-blue-500 outline-none resize-y font-mono text-sm"
                        placeholder="Enter the ideal answer here..."
                        value={aText}
                        onChange={e => setAText(e.target.value)}
                    />
                </div>

                <div className="flex justify-end gap-3 pt-4">
                    <button
                        onClick={() => mutation.mutate({ question: qText, answer: aText, marks })}
                        disabled={mutation.isPending || !qText || !aText}
                        className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-2.5 rounded-lg font-medium shadow-md hover:shadow-lg transition-all flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                        {mutation.isPending ? "Saving..." : <><Save size={18} /> Save Question</>}
                    </button>
                </div>
            </div>
        </div>
    )
}
