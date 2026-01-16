import { useState, useEffect, useRef } from 'react';
import { ProjectWizard } from './components/ProjectWizard';
import { api, type StartJobRequest } from './api';
import { Activity, CheckCircle, ChevronDown, ChevronRight, AlertCircle } from 'lucide-react';

function App() {
  const [view, setView] = useState<'wizard' | 'console' | 'results'>('wizard');
  const [jobId, setJobId] = useState<string | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const [results, setResults] = useState<any[]>([]);

  const logEndRef = useRef<HTMLDivElement>(null);

  const startJob = async (req: StartJobRequest) => {
    try {
      const res = await api.startJob(req);
      setJobId(res.job_id);
      setView('console');
    } catch (e) {
      alert("Failed to start job");
    }
  };

  useEffect(() => {
    if (view === 'console' && jobId) {
      const evtSource = new EventSource(`http://localhost:8000/job/${jobId}/stream`);

      evtSource.onmessage = (e) => {
        if (e.data.startsWith("[DONE]")) {
          evtSource.close();
          loadResults(jobId);
        } else {
          setLogs(prev => [...prev, e.data]);
        }
      };

      return () => evtSource.close();
    }
  }, [view, jobId]);

  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  const loadResults = async (id: string) => {
    // Small delay to ensure file write
    setTimeout(async () => {
      try {
        const data = await api.getResults(id);
        setResults(data);
        setView('results');
      } catch (e) {
        alert("Error loading results");
      }
    }, 1000);
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 font-sans">
      <nav className="bg-white border-b px-6 py-4 flex items-center gap-3 sticky top-0 z-10">
        <div className="bg-blue-600 text-white p-1.5 rounded-lg">
          <Activity size={24} />
        </div>
        <h1 className="text-xl font-bold tracking-tight">TrueGradeAI</h1>
      </nav>

      {view === 'wizard' && <ProjectWizard onStart={startJob} />}

      {view === 'console' && (
        <div className="max-w-4xl mx-auto py-12 px-4">
          <div className="bg-slate-900 rounded-xl shadow-2xl overflow-hidden border border-slate-800">
            <div className="bg-slate-800 px-4 py-3 border-b border-slate-700 flex items-center justify-between">
              <span className="text-slate-400 font-mono text-sm">Output Console</span>
              <div className="flex gap-2">
                <div className="w-3 h-3 rounded-full bg-red-500/20 border border-red-500/50"></div>
                <div className="w-3 h-3 rounded-full bg-yellow-500/20 border border-yellow-500/50"></div>
                <div className="w-3 h-3 rounded-full bg-green-500/20 border border-green-500/50"></div>
              </div>
            </div>
            <div className="p-6 h-[500px] overflow-y-auto font-mono text-sm space-y-2">
              {logs.map((log, i) => (
                <div key={i} className="text-slate-300 border-l-2 border-slate-700 pl-3 py-0.5 animate-in fade-in duration-300">
                  <span className="text-blue-500 mr-2">➜</span>
                  {log}
                </div>
              ))}
              {logs.length === 0 && <div className="text-slate-500 italic">Initializing worker process...</div>}
              <div ref={logEndRef} />
            </div>
          </div>
        </div>
      )}

      {view === 'results' && (
        <div className="max-w-7xl mx-auto py-8 px-4">
          <div className="flex justify-between items-center mb-6">
            <h2 className="text-2xl font-bold">Grading Results</h2>
            <button onClick={() => window.location.reload()} className="text-sm bg-white border px-3 py-1.5 rounded hover:bg-slate-50">New Job</button>
          </div>

          <div className="grid gap-4">
            {results.map((r, i) => (
              <ResultCard key={i} data={r} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function ResultCard({ data }: { data: any }) {
  const [open, setOpen] = useState(false);
  // Determine color based on score
  const pct = (data.score / data.max_marks);
  const scoreColor = pct > 0.8 ? "text-green-600" : pct > 0.5 ? "text-yellow-600" : "text-red-600";

  return (
    <div className="bg-white rounded-lg border shadow-sm overflow-hidden">
      <div
        className="p-4 flex items-center justify-between cursor-pointer hover:bg-slate-50"
        onClick={() => setOpen(!open)}
      >
        <div className="flex items-center gap-4">
          <div className={`text-xl font-bold w-16 text-center ${scoreColor}`}>
            {data.score} <span className="text-slate-400 text-sm font-normal">/ {data.max_marks}</span>
          </div>
          <div>
            <h4 className="font-medium text-slate-900">{data.student_id}</h4>
            <p className="text-sm text-slate-500 line-clamp-1">{data.question_no}</p>
          </div>
        </div>
        <div className="text-slate-400">
          {open ? <ChevronDown /> : <ChevronRight />}
        </div>
      </div>

      {open && (
        <div className="p-4 bg-slate-50 border-t space-y-4">
          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-white p-3 rounded border">
              <h5 className="text-xs font-semibold text-slate-500 uppercase mb-2">Confidence Metrics</h5>
              <div className="space-y-1 text-sm">
                <div className="flex justify-between">
                  <span>Faculty Match:</span>
                  <span className="font-mono">{data.faculty_match_sim}</span>
                </div>
                <div className="flex justify-between">
                  <span>Novelty Ratio:</span>
                  <span className="font-mono text-orange-600">{data.notes?.novelty?.toFixed(2)}</span>
                </div>
              </div>
            </div>
            <div className="bg-white p-3 rounded border">
              <h5 className="text-xs font-semibold text-slate-500 uppercase mb-2">Deductions</h5>
              {data.deductions?.length > 0 ? (
                <ul className="space-y-2">
                  {data.deductions.map((d: any, idx: number) => (
                    <li key={idx} className="text-sm text-red-600 flex gap-2">
                      <AlertCircle size={16} className="shrink-0 mt-0.5" />
                      <span>
                        <span className="font-semibold">{d.key_point}</span>: {d.reason}
                      </span>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="text-sm text-green-600 flex items-center gap-2"><CheckCircle size={16} /> Perfect Score</p>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default App;
