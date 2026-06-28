import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import Home from './pages/Home'
import MapView from './pages/MapView'

const qc = new QueryClient()

function Nav() {
  return (
    <nav className="flex items-center justify-between px-6 py-3 border-b"
         style={{ background: 'var(--bg-card)', borderColor: 'var(--border)' }}>
      <span className="font-bold text-base tracking-tight"
            style={{ color: 'var(--text-primary)' }}>
        <span style={{ color: 'var(--accent-green)' }}>Bhu</span>Me
        <span className="ml-2 text-xs font-normal"
              style={{ color: 'var(--text-muted)' }}>boundary correction</span>
      </span>
      <div className="flex gap-6 text-sm" style={{ color: 'var(--text-muted)' }}>
        <NavLink to="/"
          className={({ isActive }) => isActive ? 'font-semibold' : ''}
          style={({ isActive }) => ({ color: isActive ? 'var(--text-primary)' : undefined })}>
          Villages
        </NavLink>
        <a href="https://hiring.bhume.in/task/" target="_blank" rel="noreferrer">Task</a>
        <a href="https://github.com" target="_blank" rel="noreferrer">GitHub</a>
      </div>
    </nav>
  )
}

export default function App() {
  return (
    <QueryClientProvider client={qc}>
      <BrowserRouter>
        <div style={{ height: '100vh', display: 'flex', flexDirection: 'column' }}>
          <Nav />
          <div style={{ flex: 1, overflow: 'hidden' }}>
            <Routes>
              <Route path="/" element={<Home />} />
              <Route path="/map/:slug" element={<MapView />} />
            </Routes>
          </div>
        </div>
      </BrowserRouter>
    </QueryClientProvider>
  )
}
