import { useParams, useNavigate } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { useRef, useEffect, useState } from 'react'
import maplibregl from 'maplibre-gl'
import 'maplibre-gl/dist/maplibre-gl.css'
import type { Feature, FeatureCollection, Polygon, MultiPolygon } from 'geojson'
import { ArrowLeft, Layers, Activity, CheckCircle, AlertCircle } from 'lucide-react'
import {
  fetchPlots, fetchPredictions, fetchTruths,
  runPipeline, fetchJobStatus, type JobStatus,
} from '../lib/api'
import PlotPanel from '../components/PlotPanel/PlotPanel'

const OSMT = 'https://tile.openstreetmap.org/{z}/{x}/{y}.png'

export default function MapView() {
  const { slug } = useParams<{ slug: string }>()
  const navigate = useNavigate()
  const mapDiv  = useRef<HTMLDivElement>(null)
  const mapRef  = useRef<maplibregl.Map | null>(null)

  const [selectedFeature, setSelectedFeature] = useState<Feature | null>(null)
  const [layers, setLayers] = useState({ official: true, corrected: true, truths: false })
  const [jobStatus, setJobStatus] = useState<JobStatus | null>(null)
  const [running, setRunning] = useState(false)

  const { data: plots }       = useQuery<FeatureCollection>({ queryKey: ['plots', slug],       queryFn: () => fetchPlots(slug!),       enabled: !!slug })
  const { data: predictions } = useQuery<FeatureCollection>({ queryKey: ['predictions', slug], queryFn: () => fetchPredictions(slug!), enabled: !!slug })
  const { data: truths }      = useQuery<FeatureCollection>({ queryKey: ['truths', slug],      queryFn: () => fetchTruths(slug!),      enabled: !!slug, retry: false })

  // ── initialise map ──────────────────────────────────────────────────────────
  useEffect(() => {
    if (!mapDiv.current || mapRef.current) return
    const map = new maplibregl.Map({
      container: mapDiv.current,
      style: {
        version: 8,
        sources: { osm: { type: 'raster', tiles: [OSMT], tileSize: 256, attribution: '© OpenStreetMap' } },
        layers:  [{ id: 'osm', type: 'raster', source: 'osm', paint: { 'raster-opacity': 0.4 } }],
      },
      center: [74.03, 20.24],
      zoom: 12,
    })
    map.addControl(new maplibregl.NavigationControl(), 'top-left')
    mapRef.current = map
    return () => { map.remove(); mapRef.current = null }
  }, [])

  // ── add / update sources + layers ────────────────────────────────────────────
  useEffect(() => {
    const map = mapRef.current
    if (!map || !map.isStyleLoaded()) return

    function addOrUpdate(id: string, geoJson: FeatureCollection | undefined, color: string, opacity: number, width: number) {
      if (!geoJson) return
      if (map!.getSource(id)) {
        (map!.getSource(id) as maplibregl.GeoJSONSource).setData(geoJson)
      } else {
        map!.addSource(id, { type: 'geojson', data: geoJson })
        map!.addLayer({ id: `${id}-fill`, type: 'fill',   source: id, paint: { 'fill-color': color, 'fill-opacity': 0.08 } })
        map!.addLayer({ id: `${id}-line`, type: 'line',   source: id, paint: { 'line-color': color, 'line-width': width, 'line-opacity': opacity } })
      }
      map!.setLayoutProperty(`${id}-fill`, 'visibility', 'visible')
      map!.setLayoutProperty(`${id}-line`, 'visibility', 'visible')
    }

    if (plots)       addOrUpdate('official',    plots,       '#8b949e', 0.6, 1)
    if (predictions) addOrUpdate('predictions', predictions, '#3fb950', 1,   1.5)
    if (truths)      addOrUpdate('truths',      truths,      '#58a6ff', 1,   2)

    // Fit to village bounds
    if (plots && plots.features.length) {
      const coords = plots.features.flatMap((f) => {
        const g = f.geometry as Polygon | MultiPolygon
        if (g.type === 'Polygon') return g.coordinates[0]
        return g.coordinates.flatMap((r: number[][][]) => r[0])
      })
      const lons = coords.map((c: number[]) => c[0]), lats = coords.map((c: number[]) => c[1])
      map.fitBounds([[Math.min(...lons), Math.min(...lats)], [Math.max(...lons), Math.max(...lats)]], { padding: 40 })
    }
  }, [plots, predictions, truths])

  // ── layer visibility toggles ─────────────────────────────────────────────────
  useEffect(() => {
    const map = mapRef.current
    if (!map || !map.isStyleLoaded()) return
    const vis = (v: boolean) => v ? 'visible' : 'none'
    for (const sfx of ['-fill', '-line']) {
      if (map.getLayer('official' + sfx))    map.setLayoutProperty('official' + sfx,    'visibility', vis(layers.official))
      if (map.getLayer('predictions' + sfx)) map.setLayoutProperty('predictions' + sfx, 'visibility', vis(layers.corrected))
      if (map.getLayer('truths' + sfx))      map.setLayoutProperty('truths' + sfx,      'visibility', vis(layers.truths))
    }
  }, [layers])

  // ── click handler ────────────────────────────────────────────────────────────
  useEffect(() => {
    const map = mapRef.current
    if (!map) return
    const handler = (e: maplibregl.MapMouseEvent) => {
      const features = map.queryRenderedFeatures(e.point, {
        layers: ['predictions-fill', 'official-fill'],
      })
      setSelectedFeature((features[0] as unknown as Feature) ?? null)
    }
    map.on('click', handler)
    map.on('mouseenter', 'predictions-fill', () => { map.getCanvas().style.cursor = 'pointer' })
    map.on('mouseleave', 'predictions-fill', () => { map.getCanvas().style.cursor = '' })
    return () => { map.off('click', handler) }
  }, [predictions])

  // ── pipeline run ─────────────────────────────────────────────────────────────
  async function handleRun() {
    if (!slug) return
    setRunning(true)
    await runPipeline(slug)
    const poll = setInterval(async () => {
      const s = await fetchJobStatus(slug)
      setJobStatus(s)
      if (s.status === 'done' || s.status === 'error') {
        clearInterval(poll)
        setRunning(false)
      }
    }, 800)
  }

  const pct = jobStatus && jobStatus.total ? Math.round(100 * jobStatus.progress / jobStatus.total) : 0

  return (
    <div style={{ height: '100%', position: 'relative', display: 'flex', flexDirection: 'column' }}>
      {/* toolbar */}
      <div className="flex items-center gap-4 px-4 py-2 text-sm"
           style={{ background: 'var(--bg-card)', borderBottom: '1px solid var(--border)', zIndex: 10 }}>
        <button onClick={() => navigate('/')}
                className="flex items-center gap-1.5"
                style={{ color: 'var(--text-muted)', background: 'none', border: 'none', cursor: 'pointer' }}>
          <ArrowLeft size={14} /> Back
        </button>
        <span style={{ color: 'var(--text-dim)' }}>|</span>
        <span className="font-medium" style={{ color: 'var(--text-primary)' }}>
          {slug?.split('_').slice(1, -2).join(' ').replace(/\b\w/g, c => c.toUpperCase())}
        </span>

        {/* layer toggles */}
        <div className="flex items-center gap-2 ml-4">
          <Layers size={13} style={{ color: 'var(--text-muted)' }} />
          {[
            { key: 'official',  label: 'Official',  color: '#8b949e' },
            { key: 'corrected', label: 'Corrected', color: '#3fb950' },
            { key: 'truths',    label: 'Truths',    color: '#58a6ff' },
          ].map(({ key, label, color }) => (
            <button
              key={key}
              onClick={() => setLayers(l => ({ ...l, [key]: !l[key as keyof typeof l] }))}
              className="flex items-center gap-1 px-2 py-0.5 rounded text-xs"
              style={{
                background: layers[key as keyof typeof layers] ? 'var(--bg-card2)' : 'transparent',
                border: `1px solid ${layers[key as keyof typeof layers] ? color : 'var(--border)'}`,
                color: layers[key as keyof typeof layers] ? color : 'var(--text-dim)',
                cursor: 'pointer',
              }}
            >
              <span className="w-2 h-2 rounded-full" style={{ background: color }} />
              {label}
            </button>
          ))}
        </div>

        {/* run button */}
        <div className="ml-auto flex items-center gap-3">
          {jobStatus?.status === 'done' && jobStatus.scorecard && (
            <span className="text-xs flex items-center gap-1" style={{ color: 'var(--accent-green)' }}>
              <CheckCircle size={12} />
              IoU {jobStatus.scorecard.median_iou_pred?.toFixed(3)} · Spearman {jobStatus.scorecard.spearman?.toFixed(2) ?? '—'}
            </span>
          )}
          {jobStatus?.status === 'error' && (
            <span className="text-xs flex items-center gap-1" style={{ color: 'var(--accent-red)' }}>
              <AlertCircle size={12} /> {jobStatus.error}
            </span>
          )}
          {running && (
            <div className="flex items-center gap-2 text-xs" style={{ color: 'var(--text-muted)' }}>
              <div className="w-24 h-1.5 rounded-full" style={{ background: 'var(--bg-hover)' }}>
                <div className="h-1.5 rounded-full" style={{ width: `${pct}%`, background: 'var(--cta-from)' }} />
              </div>
              {pct}%
            </div>
          )}
          <button
            onClick={handleRun}
            disabled={running}
            className="flex items-center gap-1.5 px-3 py-1 rounded text-xs font-medium"
            style={{
              background: running ? 'var(--bg-card2)' : 'linear-gradient(135deg,var(--cta-from),var(--cta-to))',
              color: 'white', border: 'none', cursor: running ? 'wait' : 'pointer',
            }}
          >
            <Activity size={12} />
            {running ? 'Running…' : 'Run pipeline'}
          </button>
        </div>
      </div>

      {/* map */}
      <div style={{ flex: 1, position: 'relative' }}>
        <div ref={mapDiv} style={{ width: '100%', height: '100%' }} />
        {slug && (
          <PlotPanel
            slug={slug}
            feature={selectedFeature}
            onClose={() => setSelectedFeature(null)}
          />
        )}
      </div>
    </div>
  )
}
