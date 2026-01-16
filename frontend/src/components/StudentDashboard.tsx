
import { useState } from 'react';
import { useQuery, useMutation } from '@tanstack/react-query';
import { api, type Question } from '../api';
import { useNavigate } from 'react-router-dom';
import { GraduationCap, BookOpen, Send, Loader2, Sparkles, FileText } from 'lucide-react';
import clsx from 'clsx';

function BatchGrade({ onSuccess }: { onSuccess: (results: any[]) => void }) {
    const [file, setFile] = useState<File | null>(null);
    const mutation = useMutation({
        mutationFn: api.batchGradeCsv,
        onSuccess: (data) => {
            onSuccess(data);
        },
        onError: () => alert("Batch grading failed.")
    });

    return (
        <div className="bg-white border rounded-xl shadow-sm p-8 max-w-2xl mx-auto text-center animate-in fade-in slide-in-from-bottom-4 duration-500">
            <div className="mb-6">
                <div className="w-16 h-16 bg-emerald-50 text-emerald-600 rounded-full flex items-center justify-center mx-auto mb-4">
                    <FileText size={32} />
                </div>
                <h2 className="text-xl font-bold text-slate-900">Batch Grade Student Answers</h2>
                <p className="text-slate-500 mt-2">Upload `student_answers.csv` to grade multiple answers at once.</p>
            </div>

            <div className="border-2 border-dashed border-emerald-200 bg-emerald-50/30 rounded-xl p-8 mb-6 hover:bg-emerald-50 transition-colors">
                <input
                    type="file"
                    accept=".csv"
                    onChange={e => setFile(e.target.files?.[0] || null)}
                    className="block w-full text-sm text-slate-500
                      file:mr-4 file:py-2 file:px-4
                      file:rounded-full file:border-0
                      file:text-sm file:font-semibold
                      file:bg-emerald-100 file:text-emerald-700
                      hover:file:bg-emerald-200
                    "
                />
            </div>

            <button
                onClick={() => file && mutation.mutate(file)}
                disabled={!file || mutation.isPending}
                className="w-full bg-emerald-600 hover:bg-emerald-700 text-white px-6 py-3 rounded-lg font-bold shadow-md transition-all disabled:opacity-50"
            >
                {mutation.isPending ? "Processing..." : "Start Batch Grading"}
            </button>
        </div>
    )
}

export function StudentDashboard() {
    const navigate = useNavigate();
    const [selectedQ, setSelectedQ] = useState<number | null>(null);
    const [answer, setAnswer] = useState("");
    const [ragEnabled, setRagEnabled] = useState(true);
    const [mode, setMode] = useState<'single' | 'batch'>('single');

    const { data: questions } = useQuery({
        queryKey: ['questions'],
        queryFn: api.getQuestions
    });

    const mutation = useMutation({
        mutationFn: api.gradeAnswer,
        onSuccess: (data) => {
            // Navigate to results page with the data in state
            navigate('/results/latest', { state: { result: data, question: questions?.find(q => q.id === selectedQ) } });
        },
        onError: () => alert("Grading failed. Please try again.")
    });

    const activeQuestion = questions?.find((q: Question) => q.id === selectedQ);

    return (
        <div className="max-w-6xl mx-auto py-12 px-4">
            <header className="mb-10 flex items-center justify-between">
                <div className="flex items-center gap-3">
                    <div className="p-3 bg-emerald-100 text-emerald-600 rounded-xl">
                        <GraduationCap size={32} />
                    </div>
                    <div>
                        <h1 className="text-3xl font-bold text-slate-900">Student Portal</h1>
                        <p className="text-slate-500">Submit answers for instant AI feedback.</p>
                    </div>
                </div>
                <div className="bg-slate-100 p-1 rounded-lg flex text-sm font-medium">
                    <button
                        onClick={() => setMode('single')}
                        className={clsx("px-4 py-2 rounded-md transition-all", mode === 'single' ? "bg-white text-emerald-700 shadow-sm" : "text-slate-500 hover:text-slate-700")}
                    >
                        Single Answer
                    </button>
                    <button
                        onClick={() => setMode('batch')}
                        className={clsx("px-4 py-2 rounded-md transition-all", mode === 'batch' ? "bg-white text-emerald-700 shadow-sm" : "text-slate-500 hover:text-slate-700")}
                    >
                        Batch Grading
                    </button>
                </div>
            </header>

            {mode === 'batch' ? (
                <BatchGrade onSuccess={(results) => {
                    // Navigate to a special batch results page or just show simple list for now
                    console.log(results);
                    navigate('/results/batch', { state: { results } }); // We need to handle this route or updated ResultsPage
                }} />
            ) : (
                <div className="grid md:grid-cols-3 gap-8">
                    {/* Question List */}
                    <div className="md:col-span-1 space-y-4">
                        <h2 className="font-semibold text-slate-900 flex items-center gap-2">
                            <BookOpen size={18} />
                            Available Questions
                        </h2>
                        <div className="space-y-2">
                            {questions?.map((q: Question) => (
                                <button
                                    key={q.id}
                                    onClick={() => setSelectedQ(q.id)}
                                    className={clsx(
                                        "w-full text-left p-3 rounded-lg text-sm transition-all border",
                                        selectedQ === q.id
                                            ? "bg-blue-50 border-blue-200 text-blue-700 shadow-sm"
                                            : "bg-white border-transparent hover:bg-white hover:border-slate-200 text-slate-600"
                                    )}
                                >
                                    <span className="block font-medium truncate">{q.question}</span>
                                    <span className="text-xs opacity-70">{q.marks} marks</span>
                                </button>
                            ))}
                        </div>
                    </div>

                    {/* Answer Area */}
                    <div className="md:col-span-2">
                        {activeQuestion ? (
                            <div className="bg-white border rounded-xl shadow-sm p-6 animate-in fade-in slide-in-from-right-4 duration-300">
                                <div className="mb-6">
                                    <span className="text-xs font-bold text-blue-600 uppercase tracking-wider bg-blue-50 px-2 py-1 rounded">Question</span>
                                    <h3 className="text-xl font-bold text-slate-900 mt-2">{activeQuestion.question}</h3>
                                    <div className="text-slate-400 text-sm mt-1">Maximum Score: {activeQuestion.marks}</div>
                                </div>

                                <div className="mb-6">
                                    <label className="block text-sm font-medium text-slate-700 mb-2">Your Answer</label>
                                    <textarea
                                        className="w-full p-4 border rounded-lg h-60 focus:ring-2 focus:ring-blue-500 outline-none resize-none text-slate-800 leading-relaxed"
                                        placeholder="Type your detailed answer here..."
                                        value={answer}
                                        onChange={e => setAnswer(e.target.value)}
                                    />
                                    <div className="flex justify-between mt-2 text-xs text-slate-400">
                                        <span>{answer.length} characters</span>
                                        {ragEnabled && <span className="flex items-center gap-1 text-emerald-600"><Sparkles size={12} /> AI Verification Active</span>}
                                    </div>
                                </div>

                                <div className="flex items-center justify-between border-t pt-6">
                                    <label className="flex items-center gap-2 cursor-pointer select-none">
                                        <input
                                            type="checkbox"
                                            checked={ragEnabled}
                                            onChange={e => setRagEnabled(e.target.checked)}
                                            className="w-4 h-4 rounded border-slate-300 text-blue-600 focus:ring-blue-500"
                                        />
                                        <span className="text-sm text-slate-600">Enable Textbook Verification (RAG2)</span>
                                    </label>

                                    <button
                                        onClick={() => mutation.mutate({
                                            facultyQuestionId: activeQuestion.id,
                                            studentAnswerText: answer,
                                            allowRag2: ragEnabled
                                        })}
                                        disabled={mutation.isPending || !answer.trim()}
                                        className="bg-emerald-600 hover:bg-emerald-700 text-white px-8 py-3 rounded-lg font-bold shadow-lg hover:shadow-xl transition-all flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
                                    >
                                        {mutation.isPending ? (
                                            <><Loader2 size={18} className="animate-spin" /> Grading...</>
                                        ) : (
                                            <><Send size={18} /> Submit Answer</>
                                        )}
                                    </button>
                                </div>
                            </div>
                        ) : (
                            <div className="h-full flex flex-col items-center justify-center text-slate-400 border-2 border-dashed rounded-xl p-10 bg-slate-50/50">
                                <BookOpen size={48} className="mb-4 opacity-20" />
                                <p>Select a question from the list to begin.</p>
                            </div>
                        )}
                    </div>
                </div>
            )}
        </div>
    );
}
