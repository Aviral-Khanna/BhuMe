import { useState } from 'react'
import { X, Zap, AlertTriangle } from 'lucide-react'
import { useMutation } from '@tanstack/react-query'
import { explainPlot, type ExplainResponse } from '../../lib/api'

import type { Feature } from 'geojson'

interface Props {
  slug: string
  feature: Feature | null
  onClose: () => void
}

function Row({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex justify-between items-start py-1.5"
         style={{ borderBottom: '1px solid var(--border)' }}>
      <span style={{ color: 'var(--text-muted)', minWidth: 120 }}>{label}</span>
      <span className="text-right" style={{ color: 'var(--text-primary)' }}>{value ?? '—'}</span>
    </div>
  )
}

function ConfidenceBar({ value }: { value: number }) {
  const pct = Math.round(value * 100)
  const color = pct >= 70 ? 'var(--accent-green)'
              : pct >= 40 ? 'var(--accent-orange)'
              : 'var(--accent-red)'
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 rounded-full h-1.5" style={{ background: 'var(--bg-hover)' }}>
        <div className="h-1.5 rounded-full" style={{ width: `${pct}%`, background: color }} />
      </div>
      <span className="text-xs font-mono" style={{ color }}>{pct}%</span>
    </div>
  )
}

export default function PlotPanel({ slug, feature, onClose }: Props) {
  const [explanation, setExplanation] = useState<string | null>(null)
  const explainMut = useMutation({
    mutationFn: ({ pn }: { pn: string }) => explainPlot(slug, pn),
    onSuccess: (data: ExplainResponse) => setExplanation(data.explanation),
  })

  if (!feature) return null
  const p = feature.properties ?? {}
  const isFlagged = p.status === 'flagged'

  return (
    <div
      style={{
        position: 'absolute',
        top: 12,
        right: 12,
        width: 320,
        background: 'var(--bg-card)',
        border: '1px solid var(--border)',
        borderRadius: 8,
        zIndex: 100,
        boxShadow: '0 8px 32px #0006',
        overflow: 'hidden',
      }}
    >
      {/* header */}
      <div className="flex items-center justify-between px-4 py-3"
           style={{ borderBottom: '1px solid var(--border)', background: 'var(--bg-card2)' }}>
        <div className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full"
                style={{ background: isFlagged ? 'var(--accent-orange)' : 'var(--accent-green)' }} />
          <span className="font-semibold">Plot {p.plot_number}</span>
          {isFlagged
            ? <span className="flex items-center gap-0.5 text-xs"
                    style={{ color: 'var(--accent-orange)' }}>
                <AlertTriangle size={10} /> flagged
              </span>
            : <span className="text-xs" style={{ color: 'var(--accent-green)' }}>corrected</span>
          }
        </div>
        <button onClick={onClose} style={{ color: 'var(--text-muted)', background: 'none', border: 'none', cursor: 'pointer' }}>
          <X size={16} />
        </button>
      </div>

      <div className="px-4 py-3 text-xs flex flex-col gap-0">
        {!isFlagged && p.confidence != null && (
          <div className="mb-3">
            <div className="flex justify-between mb-1">
              <span style={{ color: 'var(--text-muted)' }}>Confidence</span>
            </div>
            <ConfidenceBar value={p.confidence} />
          </div>
        )}

        <Row label="Status" value={
          <span style={{ color: isFlagged ? 'var(--accent-orange)' : 'var(--accent-green)' }}>
            {p.status}
          </span>
        } />
        {p.confidence != null && <Row label="Confidence" value={Number(p.confidence).toFixed(3)} />}
        <Row label="Village"     value={p.village} />
        <Row label="Map area"    value={p.map_area_sqm ? `${Number(p.map_area_sqm).toLocaleString()} m²` : null} />
        <Row label="Rec. area"   value={p.recorded_area_sqm ? `${Number(p.recorded_area_sqm).toLocaleString()} m²` : null} />
        {p.map_area_sqm && p.recorded_area_sqm && (
          <Row label="Area ratio" value={
            <span style={{ color: Math.abs(p.map_area_sqm/p.recorded_area_sqm - 1) < 0.5 ? 'var(--accent-green)' : 'var(--accent-orange)' }}>
              {(p.map_area_sqm / p.recorded_area_sqm).toFixed(2)}×
            </span>
          } />
        )}

        {p.method_note && (
          <div className="mt-2 p-2 rounded text-[11px] font-mono leading-relaxed"
               style={{ background: 'var(--bg-card2)', color: 'var(--text-muted)', wordBreak: 'break-all' }}>
            {p.method_note}
          </div>
        )}

        {/* AI explanation */}
        <div className="mt-3">
          {explanation ? (
            <div className="p-2.5 rounded text-xs leading-relaxed"
                 style={{ background: '#0d2137', border: '1px solid #1a3a5c', color: 'var(--text-primary)' }}>
              <div className="flex items-center gap-1 mb-1 text-[11px]"
                   style={{ color: 'var(--accent-blue)' }}>
                <Zap size={10} /> Gemini explanation
              </div>
              {explanation}
            </div>
          ) : (
            <button
              onClick={() => explainMut.mutate({ pn: p.plot_number })}
              disabled={explainMut.isPending}
              className="w-full flex items-center justify-center gap-1.5 py-1.5 rounded text-xs"
              style={{
                background: 'var(--bg-card2)',
                border: '1px solid var(--border)',
                color: explainMut.isPending ? 'var(--text-dim)' : 'var(--accent-blue)',
                cursor: explainMut.isPending ? 'wait' : 'pointer',
              }}
            >
              <Zap size={11} />
              {explainMut.isPending ? 'Asking Gemini…' : 'Explain with AI'}
            </button>
          )}
        </div>
      </div>
    </div>
  )
}
