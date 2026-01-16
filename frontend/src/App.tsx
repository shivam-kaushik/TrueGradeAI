
import { BrowserRouter, Routes, Route, Link } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { LandingPage } from './components/LandingPage';
import { Activity } from 'lucide-react';

const queryClient = new QueryClient();

// Placeholder components until Phase 6
import { TeacherDashboard } from './components/TeacherDashboard';
import { StudentDashboard } from './components/StudentDashboard';
import { ResultsPage } from './components/ResultsPage';

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <div className="min-h-screen bg-slate-50 font-sans text-slate-900">
          {/* Global Nav (visible on subpages, hidden on landing) */}
          <Routes>
            <Route path="/" element={null} />
            <Route path="*" element={<Navbar />} />
          </Routes>

          <Routes>
            <Route path="/" element={<LandingPage />} />
            <Route path="/teacher" element={<TeacherDashboard />} />
            <Route path="/student" element={<StudentDashboard />} />
            <Route path="/results/:id" element={<ResultsPage />} />
          </Routes>
        </div>
      </BrowserRouter>
    </QueryClientProvider>
  );
}

function Navbar() {
  return (
    <nav className="bg-white border-b px-6 py-4 flex items-center justify-between sticky top-0 z-10">
      <Link to="/" className="flex items-center gap-2 hover:opacity-80 transition">
        <div className="bg-blue-600 text-white p-1.5 rounded-lg">
          <Activity size={20} />
        </div>
        <h1 className="text-xl font-bold tracking-tight">TrueGradeAI</h1>
      </Link>
      <div className="flex gap-4 text-sm font-medium text-slate-500">
        <Link to="/teacher" className="hover:text-blue-600">Teacher</Link>
        <Link to="/student" className="hover:text-blue-600">Student</Link>
      </div>
    </nav>
  )
}

export default App;
