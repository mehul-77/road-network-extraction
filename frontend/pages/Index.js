import { useState, useEffect, useCallback, useRef } from "react"
import Head from "next/head"
import dynamic from "next/dynamic"
import { MetricCard, Spinner, Btn, SectionLabel, ErrorBox, ReadinessBadge, TypeBar } from "../components/UI"

const MapPanel = dynamic(() => import("../components/MapPanel"), { ssr: false })
const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"

const PRESETS = [
    { key: "surat", label: "Surat", lat: 21.1702, lon: 72.8311 },
    { key: "mumbai", label: "Mumbai", lat: 19.0760, lon: 72.8777 },
    { key: "delhi", label: "Delhi", lat: 28.6139, lon: 77.2090 },
    { key: "bangalore", label: "Bengaluru", lat: 12.9716, lon: 77.5946 },
    { key: "hyderabad", label: "Hyderabad", lat: 17.3850, lon: 78.4867 },
    { key: "pune", label: "Pune", lat: 18.5204, lon: 73.8567 },
    { key: "london", label: "London", lat: 51.5074, lon: -0.1278 },
    { key: "singapore", label: "Singapore", lat: 1.3521, lon: 103.8198 },
]

// ── SIDEBAR SECTION WRAPPER ─────────────────────────────────────────────────

function Section({ title, children }) {
    return (
        <div style={{ borderBottom: "1px solid var(--border)", paddingBottom: 18, marginBottom: 18 }}>
            <SectionLabel>{title}</SectionLabel>
            {children}
        </div>
    )
}

// ── IMAGE RESULT PANEL ──────────────────────────────────────────────────────

function CVResult({ cv, osm }) {
    const [view, setView] = useState("overlay")
    const imgs = cv?.images || {}
    const m = cv?.metrics || {}

    const views = [
        { key: "original", label: "Original" },
        { key: "mask", label: "Road Mask" },
        { key: "skeleton", label: "Centerlines" },
        { key: "overlay", label: "Overlay" },
    ]

    return (
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            {/* Image viewer */}
            <div style={{ background: "var(--bg2)", borderRadius: 10, overflow: "hidden", border: "1px solid var(--border)" }}>
                <div style={{ display: "flex", borderBottom: "1px solid var(--border)" }}>
                    {views.map(v => (
                        <button key={v.key} onClick={() => setView(v.key)} style={{
                            flex: 1, padding: "8px 4px",
                            background: view === v.key ? "rgba(0,212,170,.1)" : "transparent",
                            color: view === v.key ? "var(--accent)" : "var(--muted)",
                            border: "none", fontFamily: "monospace", fontSize: 10,
                            borderBottom: view === v.key ? "2px solid var(--accent)" : "2px solid transparent",
                            letterSpacing: "0.05em", textTransform: "uppercase"
                        }}>{v.label}</button>
                    ))}
                </div>
                {imgs[view] && (
                    <img src={`data:image/png;base64,${imgs[view]}`}
                        style={{ width: "100%", display: "block" }} alt={view} />
                )}
            </div>

            {/* CV Metrics */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
                <MetricCard label="Road Area" value={`${m.road_area_percent}%`} />
                <MetricCard label="Segments" value={m.num_segments} />
                <MetricCard label="Centerline" value={`${m.road_length_pixels}px`} sub="skeleton length" />
                <MetricCard label="Density" value={m.road_density} sub="px/px ratio" />
            </div>

            {/* OSM Comparison */}
            {osm?.summary && (
                <div style={{ background: "rgba(91,143,255,.06)", border: "1px solid rgba(91,143,255,.2)", borderRadius: 10, padding: 14 }}>
                    <p style={{ fontFamily: "monospace", fontSize: 10, color: "var(--accent2)", letterSpacing: "0.12em", marginBottom: 10, textTransform: "uppercase" }}>
                        OSM Ground Truth (800m radius)
                    </p>
                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
                        <MetricCard label="OSM Roads" value={osm.summary.total_roads} accent="var(--accent2)" />
                        <MetricCard label="OSM Length" value={`${osm.summary.total_length_km}km`} accent="var(--accent2)" />
                        <MetricCard label="Density" value={`${osm.summary.road_density_km_km2}`} sub="km/km²" accent="var(--accent2)" />
                        <MetricCard label="Connectivity" value={`${osm.summary.connectivity_index}%`} accent="var(--accent2)" />
                    </div>
                    <p style={{ fontSize: 10, color: "var(--muted)", marginTop: 10, fontFamily: "monospace", lineHeight: 1.6 }}>
                        CV detected {m.road_area_percent}% road coverage.<br />
                        OSM has {osm.summary.total_length_km} km of roads in this area.
                    </p>
                </div>
            )}
        </div>
    )
}

// ── MAIN PAGE ───────────────────────────────────────────────────────────────

export default function Home() {
    const [tab, setTab] = useState("city")   // "city" | "satellite"

    // City analysis state
    const [lat, setLat] = useState(21.1702)
    const [lon, setLon] = useState(72.8311)
    const [radius, setRadius] = useState(1500)
    const [latInput, setLatInput] = useState("21.1702")
    const [lonInput, setLonInput] = useState("72.8311")
    const [osmData, setOsmData] = useState(null)
    const [osmLoading, setOsmLoading] = useState(false)
    const [osmError, setOsmError] = useState(null)
    const [activeLayer, setActiveLayer] = useState("roads")

    // Satellite state
    const [satFile, setSatFile] = useState(null)
    const [satLat, setSatLat] = useState("")
    const [satLon, setSatLon] = useState("")
    const [cvData, setCvData] = useState(null)
    const [cvLoading, setCvLoading] = useState(false)
    const [cvError, setCvError] = useState(null)
    const dropRef = useRef()

    // ── Analyze city ─────────────────────────────────────────────────────────
    const analyzeCity = useCallback(async (la, lo, r) => {
        setOsmLoading(true); setOsmError(null); setOsmData(null)
        try {
            const res = await fetch(`${API}/analyze?lat=${la}&lon=${lo}&radius=${r}`)
            if (!res.ok) { const e = await res.json(); throw new Error(e.detail || "Failed") }
            setOsmData(await res.json())
        } catch (e) { setOsmError(e.message) }
        finally { setOsmLoading(false) }
    }, [])

    useEffect(() => { analyzeCity(21.1702, 72.8311, 1500) }, [])

    function selectPreset(p) {
        setLat(p.lat); setLon(p.lon)
        setLatInput(String(p.lat)); setLonInput(String(p.lon))
        analyzeCity(p.lat, p.lon, radius)
    }

    function searchCity() {
        const la = parseFloat(latInput), lo = parseFloat(lonInput)
        if (isNaN(la) || isNaN(lo)) return
        setLat(la); setLon(lo); analyzeCity(la, lo, radius)
    }

    // ── Analyze satellite image ───────────────────────────────────────────────
    async function analyzeSatellite() {
        if (!satFile) return
        setCvLoading(true); setCvError(null); setCvData(null)
        try {
            const fd = new FormData()
            fd.append("file", satFile)
            const la = parseFloat(satLat), lo = parseFloat(satLon)
            const url = `${API}/cv/detect${!isNaN(la) && !isNaN(lo) ? `?lat=${la}&lon=${lo}` : ""}`
            const res = await fetch(url, { method: "POST", body: fd })
            if (!res.ok) { const e = await res.json(); throw new Error(e.detail || "CV failed") }
            setCvData(await res.json())
        } catch (e) { setCvError(e.message) }
        finally { setCvLoading(false) }
    }

    // ── Downloads ─────────────────────────────────────────────────────────────
    async function downloadPDF() {
        const res = await fetch(`${API}/report?lat=${lat}&lon=${lon}&radius=${radius}`)
        const blob = await res.blob()
        const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "its_report.pdf"; a.click()
    }
    async function downloadGeoJSON() {
        const res = await fetch(`${API}/geojson?lat=${lat}&lon=${lon}&radius=${radius}`)
        const blob = await res.blob()
        const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "roads.geojson"; a.click()
    }

    const s = osmData?.metrics?.summary
    const td = osmData?.metrics?.type_distribution || []

    return (
        <>
            <Head>
                <title>ITS Road Network Dashboard</title>
                <meta name="description" content="Intelligent Transportation System road network analysis using OpenStreetMap and satellite imagery" />
                <meta name="viewport" content="width=device-width, initial-scale=1" />
            </Head>

            <div style={{ display: "flex", height: "100vh", overflow: "hidden" }}>

                {/* ── SIDEBAR ──────────────────────────────────────────────────── */}
                <aside style={{
                    width: 320, flexShrink: 0,
                    background: "var(--bg2)",
                    borderRight: "1px solid var(--border)",
                    display: "flex", flexDirection: "column",
                    overflow: "hidden",
                }}>

                    {/* Logo */}
                    <div style={{ padding: "18px 20px 14px", borderBottom: "1px solid var(--border)" }}>
                        <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
                            <div style={{ width: 8, height: 8, borderRadius: "50%", background: "var(--accent)", boxShadow: "0 0 8px var(--accent)", animation: "pulse 2s infinite" }} />
                            <span style={{ fontFamily: "monospace", fontSize: 9, color: "var(--accent)", letterSpacing: "0.2em" }}>ITS DASHBOARD · NIT SURAT</span>
                        </div>
                        <h1 style={{ fontFamily: "'Space Mono', monospace", fontSize: 16, fontWeight: 700, color: "#f0f0ff", lineHeight: 1.3 }}>
                            Road Network<br /><span style={{ color: "var(--accent)" }}>Analysis</span>
                        </h1>
                    </div>

                    {/* Tab switcher */}
                    <div style={{ display: "flex", borderBottom: "1px solid var(--border)" }}>
                        {[["city", "🗺 City Analysis"], ["satellite", "🛰 Satellite Image"]].map(([key, label]) => (
                            <button key={key} onClick={() => setTab(key)} style={{
                                flex: 1, padding: "10px 8px", border: "none",
                                background: tab === key ? "rgba(0,212,170,.08)" : "transparent",
                                color: tab === key ? "var(--accent)" : "var(--muted)",
                                fontFamily: "monospace", fontSize: 10, letterSpacing: "0.06em",
                                borderBottom: tab === key ? "2px solid var(--accent)" : "2px solid transparent",
                                textTransform: "uppercase"
                            }}>{label}</button>
                        ))}
                    </div>

                    {/* Scrollable content */}
                    <div style={{ flex: 1, overflowY: "auto", padding: "16px 20px" }}>

                        {/* ── CITY TAB ── */}
                        {tab === "city" && (
                            <div style={{ display: "flex", flexDirection: "column", gap: 0 }}>

                                <Section title="Quick Select">
                                    <div style={{ display: "flex", flexWrap: "wrap", gap: 5 }}>
                                        {PRESETS.map(p => (
                                            <button key={p.key} onClick={() => selectPreset(p)} style={{
                                                padding: "4px 10px", borderRadius: 6,
                                                border: `1px solid ${lat === p.lat ? "rgba(0,212,170,.4)" : "var(--border)"}`,
                                                background: lat === p.lat ? "rgba(0,212,170,.1)" : "var(--bg3)",
                                                color: lat === p.lat ? "var(--accent)" : "var(--muted)",
                                                fontSize: 11, fontFamily: "monospace",
                                            }}>{p.label}</button>
                                        ))}
                                    </div>
                                </Section>

                                <Section title="Custom Location">
                                    <div style={{ display: "flex", gap: 6, marginBottom: 8 }}>
                                        {[["Latitude", latInput, setLatInput], ["Longitude", lonInput, setLonInput]].map(([ph, val, set]) => (
                                            <input key={ph} value={val} onChange={e => set(e.target.value)}
                                                placeholder={ph} style={{
                                                    flex: 1, padding: "8px 10px", background: "var(--bg3)",
                                                    border: "1px solid var(--border)", borderRadius: 7, color: "var(--text)",
                                                    fontFamily: "monospace", fontSize: 11
                                                }} />
                                        ))}
                                    </div>
                                    <div style={{ marginBottom: 10 }}>
                                        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                                            <span style={{ fontSize: 10, color: "var(--muted)", fontFamily: "monospace" }}>Radius</span>
                                            <span style={{ fontSize: 10, color: "var(--accent)", fontFamily: "monospace" }}>{radius}m</span>
                                        </div>
                                        <input type="range" min="500" max="5000" step="250" value={radius}
                                            onChange={e => setRadius(+e.target.value)}
                                            style={{ width: "100%", accentColor: "var(--accent)" }} />
                                    </div>
                                    <Btn onClick={searchCity} disabled={osmLoading} loading={osmLoading} style={{ width: "100%", justifyContent: "center" }}>
                                        ▶ Analyze Area
                                    </Btn>
                                    <ErrorBox message={osmError} />
                                </Section>

                                {s && (
                                    <>
                                        <Section title="ITS Readiness">
                                            <ReadinessBadge its={s.its_readiness} />
                                        </Section>

                                        <Section title="Key Metrics">
                                            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 7 }}>
                                                <MetricCard label="Total Roads" value={s.total_roads} />
                                                <MetricCard label="Total Length" value={`${s.total_length_km}km`} />
                                                <MetricCard label="Road Density" value={s.road_density_km_km2} sub="km/km²" />
                                                <MetricCard label="Intersections" value={s.intersection_count} />
                                                <MetricCard label="Dead Ends" value={s.dead_end_count} accent="var(--danger)" />
                                                <MetricCard label="Connectivity" value={`${s.connectivity_index}%`}
                                                    accent={s.connectivity_index > 65 ? "var(--accent)" : s.connectivity_index > 40 ? "var(--warn)" : "var(--danger)"} />
                                            </div>
                                        </Section>

                                        <Section title="Road Hierarchy">
                                            {td.slice(0, 8).map(t => <TypeBar key={t.type} item={t} />)}
                                        </Section>

                                        <Section title="Map Layer">
                                            <div style={{ display: "flex", gap: 5 }}>
                                                {["roads", "heatmap", "intersections"].map(l => (
                                                    <button key={l} onClick={() => setActiveLayer(l)} style={{
                                                        flex: 1, padding: "6px 4px", borderRadius: 6,
                                                        border: `1px solid ${activeLayer === l ? "var(--accent)" : "var(--border)"}`,
                                                        background: activeLayer === l ? "rgba(0,212,170,.1)" : "var(--bg3)",
                                                        color: activeLayer === l ? "var(--accent)" : "var(--muted)",
                                                        fontSize: 9, fontFamily: "monospace", textTransform: "uppercase", letterSpacing: "0.06em"
                                                    }}>{l}</button>
                                                ))}
                                            </div>
                                        </Section>

                                        <div style={{ display: "flex", gap: 7, marginBottom: 16 }}>
                                            <Btn onClick={downloadGeoJSON} variant="ghost" style={{ flex: 1, justifyContent: "center", fontSize: 10 }}>
                                                ⬇ GeoJSON
                                            </Btn>
                                            <Btn onClick={downloadPDF} variant="secondary" style={{ flex: 1, justifyContent: "center", fontSize: 10 }}>
                                                ⬇ PDF Report
                                            </Btn>
                                        </div>
                                    </>
                                )}

                                {osmLoading && <Spinner label="Fetching road network..." />}
                            </div>
                        )}

                        {/* ── SATELLITE TAB ── */}
                        {tab === "satellite" && (
                            <div style={{ display: "flex", flexDirection: "column", gap: 0 }}>

                                <Section title="Upload Satellite Image">
                                    <div
                                        onClick={() => dropRef.current?.click()}
                                        onDragOver={e => e.preventDefault()}
                                        onDrop={e => { e.preventDefault(); setSatFile(e.dataTransfer.files[0]) }}
                                        style={{
                                            border: `2px dashed ${satFile ? "var(--accent)" : "var(--border)"}`,
                                            borderRadius: 10, padding: "24px 16px", textAlign: "center",
                                            cursor: "pointer", background: satFile ? "rgba(0,212,170,.04)" : "var(--bg3)",
                                            marginBottom: 10, transition: "all .2s"
                                        }}>
                                        <input ref={dropRef} type="file" accept="image/*" style={{ display: "none" }}
                                            onChange={e => setSatFile(e.target.files[0])} />
                                        <p style={{ color: satFile ? "var(--accent)" : "var(--muted)", fontFamily: "monospace", fontSize: 11 }}>
                                            {satFile ? `✓ ${satFile.name}` : "Drop satellite image or click to browse"}
                                        </p>
                                        <p style={{ color: "#333", fontSize: 10, marginTop: 4, fontFamily: "monospace" }}>JPG / PNG from Google Earth or downloaded tiles</p>
                                    </div>

                                    <p style={{ fontSize: 10, color: "var(--muted)", fontFamily: "monospace", marginBottom: 6 }}>
                                        Optional: add coordinates for OSM comparison
                                    </p>
                                    <div style={{ display: "flex", gap: 6, marginBottom: 10 }}>
                                        {[["Tile Lat", satLat, setSatLat], ["Tile Lon", satLon, setSatLon]].map(([ph, val, set]) => (
                                            <input key={ph} value={val} onChange={e => set(e.target.value)}
                                                placeholder={ph} style={{
                                                    flex: 1, padding: "7px 10px", background: "var(--bg3)",
                                                    border: "1px solid var(--border)", borderRadius: 7, color: "var(--text)",
                                                    fontFamily: "monospace", fontSize: 11
                                                }} />
                                        ))}
                                    </div>
                                    <Btn onClick={analyzeSatellite} disabled={!satFile || cvLoading} loading={cvLoading} style={{ width: "100%", justifyContent: "center" }}>
                                        🛰 Extract Roads
                                    </Btn>
                                    <ErrorBox message={cvError} />
                                </Section>

                                {cvLoading && <Spinner label="Running road extraction..." />}

                                {!cvData && !cvLoading && (
                                    <div style={{ padding: "20px 0", textAlign: "center" }}>
                                        <p style={{ color: "var(--muted)", fontSize: 12, fontFamily: "monospace", lineHeight: 1.8 }}>
                                            Upload a satellite image to detect roads using classical image processing (Canny edges + linear morphology).<br /><br />
                                            Add coordinates to compare with OpenStreetMap ground truth.
                                        </p>
                                    </div>
                                )}
                            </div>
                        )}
                    </div>

                    {/* Footer */}
                    <div style={{ padding: "10px 20px", borderTop: "1px solid var(--border)" }}>
                        <p style={{ fontSize: 9, color: "#2a2a40", fontFamily: "monospace", textAlign: "center" }}>
                            Data © OpenStreetMap Contributors (ODbL) · NIT Surat ITS Project 2026
                        </p>
                    </div>
                </aside>

                {/* ── MAP + FLOATING STATS ─────────────────────────────────────── */}
                <main style={{ flex: 1, position: "relative", background: "var(--bg)" }}>
                    {tab === "city" && (
                        <>
                            {osmLoading && (
                                <div style={{ position: "absolute", inset: 0, background: "rgba(8,8,16,.75)", zIndex: 999, display: "flex", alignItems: "center", justifyContent: "center" }}>
                                    <Spinner label="Fetching road network from OpenStreetMap..." />
                                </div>
                            )}
                            <MapPanel lat={lat} lon={lon} radius={radius} data={osmData} activeLayer={activeLayer} onMoveEnd={(la, lo) => {
                                setLatInput(String(la));
                                setLonInput(String(lo));
                            }} />

                            {/* Floating status bar */}
                            {s && (
                                <div style={{
                                    position: "absolute", bottom: 20, left: "50%", transform: "translateX(-50%)",
                                    background: "rgba(13,13,26,.92)", backdropFilter: "blur(10px)",
                                    border: "1px solid rgba(0,212,170,.2)", borderRadius: 10,
                                    padding: "8px 24px", display: "flex", gap: 28, zIndex: 500,
                                    fontFamily: "monospace",
                                }}>
                                    {[
                                        ["Roads", s.total_roads],
                                        ["Length", `${s.total_length_km}km`],
                                        ["Density", `${s.road_density_km_km2}km/km²`],
                                        ["Connect", `${s.connectivity_index}%`],
                                    ].map(([k, v]) => (
                                        <div key={k} style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 2 }}>
                                            <span style={{ color: "#333", fontSize: 8, letterSpacing: "0.12em", textTransform: "uppercase" }}>{k}</span>
                                            <span style={{ color: "var(--accent)", fontWeight: 700, fontSize: 13 }}>{v}</span>
                                        </div>
                                    ))}
                                </div>
                            )}
                        </>
                    )}

                    {tab === "satellite" && (
                        <div style={{ width: "100%", height: "100%", overflowY: "auto", padding: "40px" }}>
                            {cvData ? (
                                <div style={{ maxWidth: 1000, margin: "0 auto" }}>
                                    <CVResult cv={cvData.cv} osm={cvData.osm} />
                                </div>
                            ) : (
                                <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%", flexDirection: "column", gap: 16 }}>
                                    <div style={{ fontSize: 64, opacity: .15 }}>🛰</div>
                                    <p style={{ color: "var(--muted)", fontFamily: "monospace", fontSize: 13, textAlign: "center", lineHeight: 1.8, maxWidth: 340 }}>
                                        Upload a satellite image from the sidebar to see road extraction results here.
                                        <br /><br />
                                        You can use any of the tiles downloaded with <code style={{ color: "var(--accent)", fontSize: 11 }}>dataset_downloader.py</code>
                                    </p>
                                </div>
                            )}
                        </div>
                    )}
                </main>
            </div>
        </>
    )
}