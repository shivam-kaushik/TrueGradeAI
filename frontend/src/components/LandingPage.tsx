

import { Link } from 'react-router-dom';
import { Brain, GraduationCap, School } from 'lucide-react';

export function LandingPage() {
    return (
        <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 text-white flex flex-col items-center justify-center p-4">
            <div className="text-center mb-12">
                <div className="flex justify-center mb-6">
                    <div className="p-4 bg-blue-600 rounded-2xl shadow-xl shadow-blue-500/20">
                        <Brain size={48} className="text-white" />
                    </div>
                </div>
                <h1 className="text-5xl font-extrabold tracking-tight mb-4 bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-indigo-400">
                    TrueGradeAI
                </h1>
                <p className="text-xl text-slate-400 max-w-lg mx-auto">
                    The autonomous grading platform that thinks like a professor.
                </p>
            </div>

            <div className="grid md:grid-cols-2 gap-8 w-full max-w-4xl">
                <RoleCard
                    to="/teacher"
                    title="Teacher Dashboard"
                    icon={<School size={32} />}
                    desc="Manage questions, set answer keys, and configure strictness."
                    color="bg-indigo-600 hover:bg-indigo-500"
                />
                <RoleCard
                    to="/student"
                    title="Student Portal"
                    icon={<GraduationCap size={32} />}
                    desc="Submit answers and get instant, explainable feedback."
                    color="bg-emerald-600 hover:bg-emerald-500"
                />
            </div>
        </div>
    );
}

function RoleCard({ to, title, icon, desc, color }: any) {
    return (
        <Link to={to} className={`block p-8 rounded-2xl border border-white/10 ${color} transition-all transform hover:scale-[1.02] hover:shadow-2xl`}>
            <div className="bg-white/20 w-fit p-3 rounded-lg mb-4 text-white">
                {icon}
            </div>
            <h2 className="text-2xl font-bold mb-2 text-white">{title}</h2>
            <p className="text-white/80">{desc}</p>
        </Link>
    )
}
