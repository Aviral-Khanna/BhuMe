import { useQuery } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import { MapPin, Layers, Activity } from 'lucide-react'
import { fetchVillages, runPipeline, type VillageMeta } from '../lib/api'
import { useState } from 'react'

function StatChip({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="flex flex-col">
      <span className="text-2xl font-bold" style={{ color: 'var(--text-primary)' }}>
        {typeof value === 'number' ? value.toLocaleString() : value}
      </span>
      <span className="text-xs uppercase tracking-widest mt-0.5"
            style={{ color: 'var(--text-muted)' }}>{label}</span>
    </div>
  )
}

function VillageCard({ v }: { v: VillageMeta }) {
  const navigate = useNavigate()
  const [running, setRunning] = useState(false)
  const [ran, setRan] = useState(false)

  const parts   = v.slug.split('_')
  const name    = parts.slice(1, -2).join(' ')
    .replace(/\b\w/g, c => c.toUpperCase())
  const district = parts.at(-1)?.replace(/\b\w/g, c => c.toUpperCase()) ?? ''

  async function handleRun(e: React.MouseEvent) {
    e.stopPropagation()
    setRunning(true)
    await runPipeline(v.slug)
    setRunning(false)
    setRan(true)
  }

  return (
    <div
      onClick={() => navigate(`/map/${v.slug}`)}
      className="rounded-lg p-5 cursor-pointer transition-all"
      style={{
        background: 'var(--bg-card)',
        border: '1px solid var(--border)',
      }}
      onMouseEnter={e => (e.currentTarget.style.borderColor = 'var(--accent-blue)')}
      onMouseLeave={e => (e.currentTarget.style.borderColor = 'var(--border)')}
    >
      {/* header */}
      <div className="flex items-start justify-between mb-4">
        <div>
          <div className="flex items-center gap-2 mb-0.5">
            <span className="w-2 h-2 rounded-full"
                  style={{ background: 'var(--accent-red)' }} />
            <h2 className="text-base font-semibold capitalize">{name}</h2>
          </div>
          <p className="text-xs uppercase tracking-widest"
             style={{ color: 'var(--text-muted)' }}>{district}</p>
        </div>
        {v.has_predictions && (
          <span className="text-xs px-2 py-0.5 rounded"
                style={{ background: '#1a3a1a', color: 'var(--accent-green)', border: '1px solid #2d5a2d' }}>
            predictions ready
          </span>
        )}
      </div>

      {/* stats row */}
      <div className="flex gap-6 mb-5">
        <StatChip label="plots"      value={v.n_plots} />
        <StatChip label="imagery"    value={v.has_imagery    ? 'yes' : '—'} />
        <StatChip label="boundaries" value={v.has_boundaries ? 'yes' : '—'} />
        <StatChip label="truths"     value={v.has_example_truths ? 'yes' : '—'} />
      </div>

      {/* file list */}
      <div className="flex flex-col gap-1 mb-4 text-xs font-mono"
           style={{ color: 'var(--text-muted)' }}>
        <div className="flex items-center gap-2">
          <Layers size={12} /> input.geojson
        </div>
        <div className="flex items-center gap-2">
          <Layers size={12} /> imagery.tif
        </div>
        <div className="flex items-center gap-2">
          <Layers size={12} />
          boundaries.tif
          <span className="px-1.5 py-0 rounded text-[10px] uppercase tracking-wide"
                style={{ background: '#3a2a0a', color: 'var(--accent-orange)', border: '1px solid #5a3f10' }}>
            rough
          </span>
        </div>
        {v.has_example_truths && (
          <div className="flex items-center gap-2">
            <Layers size={12} /> example_truths.geojson
          </div>
        )}
      </div>

      {/* actions */}
      <div className="flex gap-2">
        <button
          onClick={handleRun}
          disabled={running}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded text-xs font-medium transition-opacity"
          style={{
            background: running ? 'var(--bg-card2)' : 'linear-gradient(135deg,var(--cta-from),var(--cta-to))',
            color: 'white',
            border: 'none',
            cursor: running ? 'wait' : 'pointer',
            opacity: running ? 0.7 : 1,
          }}
        >
          <Activity size={12} />
          {running ? 'Running…' : ran ? 'Re-run' : 'Run pipeline'}
        </button>
        <button
          onClick={e => { e.stopPropagation(); navigate(`/map/${v.slug}`) }}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded text-xs font-medium"
          style={{
            background: 'var(--bg-card2)',
            color: 'var(--text-muted)',
            border: '1px solid var(--border)',
            cursor: 'pointer',
          }}
        >
          <MapPin size={12} /> Open map
        </button>
      </div>
    </div>
  )
}

export default function Home() {
  const { data, isLoading, error } = useQuery({
    queryKey: ['villages'],
    queryFn: fetchVillages,
    refetchInterval: 5000,
  })

  return (
    <div style={{ height: '100%', overflowY: 'auto', padding: '40px 48px' }}>
      {/* hero */}
      <div className="mb-10">
        <p className="text-xs uppercase tracking-widest mb-2"
           style={{ color: 'var(--text-muted)' }}>
          Maharashtra · cadastral boundary correction
        </p>
        <h1 className="text-3xl font-bold mb-3" style={{ color: 'var(--text-primary)' }}>
          The boundary on the map<br />
          <span style={{ color: 'var(--accent-blue)' }}>isn't where the land is.</span>
        </h1>
        <p style={{ color: 'var(--text-muted)', maxWidth: 520 }}>
          Cross-correlation alignment of official cadastral plots against
          pre-detected field edges, with calibrated per-plot confidence scores.
          Targets Gold–Platinum tier on the BhuMe scoring rubric.
        </p>
      </div>

      {/* village cards */}
      {isLoading && (
        <p style={{ color: 'var(--text-muted)' }}>Loading villages…</p>
      )}
      {error && (
        <p style={{ color: 'var(--accent-red)' }}>
          Backend not reachable — start the server with{' '}
          <code>uv run uvicorn backend.main:app --reload</code>
        </p>
      )}
      {data && (
        <div className="grid gap-6" style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(380px, 1fr))' }}>
          {data.map(v => <VillageCard key={v.slug} v={v} />)}
        </div>
      )}
    </div>
  )
}
