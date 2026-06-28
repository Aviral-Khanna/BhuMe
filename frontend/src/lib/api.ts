/**
 * Typed API client for the BhuMe FastAPI backend.
 */
import type { FeatureCollection } from 'geojson'

const BASE = '/api'

export interface VillageMeta {
  slug: string
  n_plots: number
  has_imagery: boolean
  has_boundaries: boolean
  has_example_truths: boolean
  has_predictions: boolean
}

export interface Scorecard {
  median_iou_pred: number | null
  median_iou_official: number | null
  improvement: number | null
  spearman: number | null
  auc: number | null
  n_corrected: number
  n_flagged: number
}

export interface JobStatus {
  status: 'not_started' | 'running' | 'done' | 'error'
  progress: number
  total: number
  error?: string | null
  scorecard?: Scorecard | null
}

export interface ExplainResponse {
  plot_number: string
  explanation: string
  status: string
  confidence: number
}

// ── village list ──────────────────────────────────────────────────────────────

export async function fetchVillages(): Promise<VillageMeta[]> {
  const res = await fetch(`${BASE}/villages/`)
  if (!res.ok) throw new Error(`fetchVillages: ${res.status}`)
  return res.json()
}

// ── GeoJSON ───────────────────────────────────────────────────────────────────

export async function fetchPlots(slug: string): Promise<FeatureCollection> {
  const res = await fetch(`${BASE}/villages/${slug}/plots`)
  if (!res.ok) throw new Error(`fetchPlots: ${res.status}`)
  return res.json()
}

export async function fetchPredictions(slug: string): Promise<FeatureCollection> {
  const res = await fetch(`${BASE}/villages/${slug}/predictions`)
  if (!res.ok) throw new Error(`fetchPredictions: ${res.status}`)
  return res.json()
}

export async function fetchTruths(slug: string): Promise<FeatureCollection> {
  const res = await fetch(`${BASE}/villages/${slug}/truths`)
  if (!res.ok) throw new Error(`fetchTruths: ${res.status}`)
  return res.json()
}

// ── pipeline ──────────────────────────────────────────────────────────────────

export async function runPipeline(slug: string): Promise<{ job_id: string; status: string }> {
  const res = await fetch(`${BASE}/villages/${slug}/run`, { method: 'POST' })
  if (!res.ok) throw new Error(`runPipeline: ${res.status}`)
  return res.json()
}

export async function fetchJobStatus(slug: string): Promise<JobStatus> {
  const res = await fetch(`${BASE}/villages/${slug}/status`)
  if (!res.ok) throw new Error(`fetchJobStatus: ${res.status}`)
  return res.json()
}

// ── AI explanation ────────────────────────────────────────────────────────────

export async function explainPlot(slug: string, plotNumber: string): Promise<ExplainResponse> {
  const res = await fetch(`${BASE}/explain/`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ slug, plot_number: plotNumber }),
  })
  if (!res.ok) throw new Error(`explainPlot: ${res.status}`)
  return res.json()
}

// ── WebSocket progress ────────────────────────────────────────────────────────

export function connectProgress(
  slug: string,
  onMessage: (state: JobStatus) => void,
): WebSocket {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws'
  const ws = new WebSocket(`${proto}://${location.host}/api/villages/${slug}/ws`)
  ws.onmessage = (e) => onMessage(JSON.parse(e.data))
  return ws
}
