

import { useLocation, Navigate, Link } from 'react-router-dom';
import { CheckCircle, AlertCircle, AlertTriangle, BookOpen, Clock, ArrowLeft } from 'lucide-react';
import clsx from 'clsx';

export function ResultsPage() {
    const location = useLocation();
    const result = location.state?.result;
    const question = location.state?.question;

    if (!result) {
        return <Navigate to="/student" />;
    }

    const percentage = (result.score / result.maxMarks) * 100;
    const scoreColor = percentage >= 80 ? "text-emerald-600" : percentage >= 50 ? "text-yellow-600" : "text-red-600";
    const barColor = percentage >= 80 ? "bg-emerald-500" : percentage >= 50 ? "bg-yellow-500" : "bg-red-500";

    return (
        <div className="max-w-4xl mx-auto py-12 px-4">
            <Link to="/student" className="inline-flex items-center gap-2 text-slate-500 hover:text-blue-600 mb-6 transition-colors">
                <ArrowLeft size={16} /> Back to Dashboard
            </Link>

            <header className="mb-8">
                <h1 className="text-3xl font-bold text-slate-900 mb-2">Grading Results</h1>
                <p className="text-slate-500 text-lg">{question?.question || "Question Evaluation"}</p>
            </header>

            <div className="grid md:grid-cols-3 gap-6 mb-8">
                {/* Score Card */}
                <div className="bg-white p-6 rounded-xl border shadow-sm flex flex-col items-center justify-center text-center">
                    <div className="text-sm font-medium text-slate-400 uppercase tracking-widest mb-2">Final Score</div>
                    <div className={clsx("text-6xl font-black mb-2", scoreColor)}>
                        {result.score} <span className="text-2xl font-medium text-slate-300">/ {result.maxMarks}</span>
                    </div>
                    <div className="w-full bg-slate-100 h-2 rounded-full overflow-hidden">
                        <div className={clsx("h-full rounded-full transition-all duration-1000", barColor)} style={{ width: `${percentage}%` }} />
                    </div>
                </div>

                {/* Metrics */}
                <div className="bg-white p-6 rounded-xl border shadow-sm space-y-4">
                    <MetricRow label="AI Confidence" value={`${(result.confidence * 100).toFixed(0)}%`} icon={<CheckCircle size={16} />} color="text-blue-600" />
                    <MetricRow label="Answer Novelty" value={result.novelty.toFixed(2)} icon={<BookOpen size={16} />} color="text-purple-600" />
                    <MetricRow
                        label="RAG2 Verification"
                        value={result.usedRag2 ? "Triggered" : "Not Needed"}
                        icon={<AlertTriangle size={16} />}
                        color={result.usedRag2 ? "text-orange-600" : "text-slate-400"}
                    />
                    <MetricRow label="Faculty Match" value={result.facultyMatch?.similarity?.toFixed(2) || "N/A"} icon={<Clock size={16} />} color="text-slate-600" />
                </div>
            </div>

            <div className="grid md:grid-cols-2 gap-8">
                {/* Key Point Breakdown */}
                <div className="space-y-4">
                    <h2 className="font-semibold text-slate-900 pb-2 border-b">Key Point Analysis</h2>
                    {result.keyPointBreakdown.map((kp: any, idx: number) => (
                        <div key={idx} className="bg-white p-4 rounded-lg border shadow-sm">
                            <div className="flex justify-between items-start mb-2">
                                <span className={clsx(
                                    "px-2 py-0.5 rounded text-xs font-bold uppercase",
                                    kp.status === "FULL" ? "bg-emerald-100 text-emerald-700" :
                                        kp.status === "PARTIAL" ? "bg-yellow-100 text-yellow-700" : "bg-red-100 text-red-700"
                                )}>
                                    {kp.status} Match
                                </span>
                                <span className="font-mono font-medium text-slate-600">+{kp.awarded} / {kp.weight}</span>
                            </div>
                            <p className="text-sm text-slate-600">{kp.reason}</p>
                        </div>
                    ))}
                </div>

                {/* Deductions */}
                <div className="space-y-4">
                    <h2 className="font-semibold text-slate-900 pb-2 border-b">Deductions & Improvements</h2>
                    {result.deductions.length > 0 ? (
                        result.deductions.map((d: any, idx: number) => (
                            <div key={idx} className="bg-red-50 p-4 rounded-lg border border-red-100">
                                <div className="flex items-start gap-3">
                                    <AlertCircle className="text-red-500 shrink-0 mt-0.5" size={18} />
                                    <div>
                                        <div className="text-red-700 font-medium mb-1">-{d.lost} marks</div>
                                        <p className="text-sm text-red-600 mb-2">{d.reason}</p>
                                        <div className="text-xs bg-white/50 p-2 rounded text-red-500 italic">
                                            Missing: "{d.keyPoint}"
                                        </div>
                                    </div>
                                </div>
                            </div>
                        ))
                    ) : (
                        <div className="bg-emerald-50 p-8 rounded-lg border border-emerald-100 text-center">
                            <CheckCircle size={48} className="text-emerald-500 mx-auto mb-4" />
                            <h3 className="text-emerald-700 font-bold text-lg">Perfect Answer!</h3>
                            <p className="text-emerald-600">No deductions identified.</p>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}

function MetricRow({ label, value, icon, color }: any) {
    return (
        <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 text-slate-500 text-sm">
                {icon}
                {label}
            </div>
            <div className={clsx("font-bold font-mono", color)}>{value}</div>
        </div>
    )
}
