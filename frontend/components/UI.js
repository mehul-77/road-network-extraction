// ── Shared UI primitives ─────────────────────────────────────────────────────

export function MetricCard({ label, value, sub, accent = "var(--accent)" }) {
    return (
        <div style={{
            background: "rgba(255,255,255,0.025)",
            border: "1px solid var(--border)",
            borderRadius: 10, padding: "14px 16px",
            display: "flex", flexDirection: "column", gap: 3,
        }}>
            <span style={{ fontSize: 9, letterSpacing: "0.16em", color: "var(--muted)", textTransform: "uppercase", fontFamily: "monospace" }}>{label}</span>
            <span style={{ fontSize: 20, fontWeight: 700, color: accent, fontFamily: "'Space Mono',monospace", lineHeight: 1.2 }}>{value}</span>
            {sub && <span style={{ fontSize: 10, color: "#444460", fontFamily: "monospace" }}>{sub}</span>}
        </div>
    )
}

export function Spinner({ size = 36, label = "Loading..." }) {
    return (
        <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 12, padding: 32 }}>
            <div style={{
                width: size, height: size,
                border: `3px solid #1a1a2e`,
                borderTop: `3px solid var(--accent)`,
                borderRadius: "50%",
                animation: "spin .8s linear infinite"
            }} />
            <span style={{ color: "var(--muted)", fontFamily: "monospace", fontSize: 12 }}>{label}</span>
        </div>
    )
}

export function Btn({ children, onClick, disabled, loading, variant = "primary", style: s = {} }) {
    const variants = {
        primary: { background: "var(--accent)", color: "#001a10", border: "none" },
        secondary: { background: "transparent", color: "var(--accent)", border: "1px solid rgba(0,212,170,.25)" },
        blue: { background: "var(--accent2)", color: "#fff", border: "none" },
        ghost: { background: "var(--bg3)", color: "var(--muted)", border: "1px solid var(--border)" },
    }
    return (
        <button onClick={onClick} disabled={disabled || loading} style={{
            padding: "9px 18px", borderRadius: 8,
            fontFamily: "monospace", fontSize: 12, fontWeight: 700,
            letterSpacing: "0.07em",
            display: "inline-flex", alignItems: "center", gap: 7,
            ...variants[variant], ...s
        }}>
            {loading && <span style={{ animation: "spin .8s linear infinite", display: "inline-block", fontSize: 14 }}>◌</span>}
            {children}
        </button>
    )
}

export function SectionLabel({ children }) {
    return (
        <p style={{ fontSize: 9, letterSpacing: "0.18em", color: "var(--muted)", textTransform: "uppercase", fontFamily: "monospace", marginBottom: 8 }}>
            {children}
        </p>
    )
}

export function ErrorBox({ message }) {
    if (!message) return null
    return (
        <div style={{ background: "#1a0808", border: "1px solid rgba(234,67,53,.3)", borderRadius: 8, padding: "10px 14px" }}>
            <p style={{ color: "var(--danger)", fontFamily: "monospace", fontSize: 11 }}>⚠ {message}</p>
        </div>
    )
}

export function ReadinessBadge({ its }) {
    if (!its) return null
    const colors = { High: "var(--accent)", Medium: "var(--warn)", Low: "var(--danger)" }
    const bg = { High: "rgba(0,212,170,.08)", Medium: "rgba(249,171,0,.08)", Low: "rgba(234,67,53,.08)" }
    const emoji = { High: "🟢", Medium: "🟡", Low: "🔴" }
    const color = colors[its.label] || "var(--muted)"
    return (
        <div style={{ background: bg[its.label], border: `1px solid ${color}33`, borderRadius: 8, padding: "10px 14px" }}>
            <div style={{ color, fontWeight: 700, fontFamily: "monospace", fontSize: 12, marginBottom: 3 }}>
                {emoji[its.label]} {its.label} ITS Readiness
            </div>
            <div style={{ color: "var(--muted)", fontSize: 11, lineHeight: 1.5 }}>{its.desc}</div>
        </div>
    )
}

export function TypeBar({ item }) {
    return (
        <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 5 }}>
            <div style={{ width: 8, height: 8, borderRadius: 2, background: item.color, flexShrink: 0 }} />
            <span style={{ color: "#888", fontSize: 11, width: 100, flexShrink: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{item.label}</span>
            <div style={{ flex: 1, background: "#1a1a2e", borderRadius: 4, height: 6, overflow: "hidden" }}>
                <div style={{ width: `${item.percent}%`, height: "100%", background: item.color, borderRadius: 4, transition: "width .5s ease" }} />
            </div>
            <span style={{ color: "#444460", fontSize: 10, fontFamily: "monospace", width: 52, textAlign: "right", flexShrink: 0 }}>{item.length_km}km</span>
        </div>
    )
}