import { useEffect, useRef } from "react"

export default function MapPanel({ lat, lon, radius, data, activeLayer, onMoveEnd }) {
    const containerRef = useRef(null)
    const mapRef = useRef(null)
    const layersRef = useRef([])
    const onMoveEndRef = useRef(onMoveEnd)

    useEffect(() => { onMoveEndRef.current = onMoveEnd }, [onMoveEnd])

    useEffect(() => {
        if (!containerRef.current || mapRef.current) return
        const L = require("leaflet")
        const map = L.map(containerRef.current, { center: [lat, lon], zoom: 15, zoomControl: true })
        L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
            attribution: '© OSM © CARTO', maxZoom: 19
        }).addTo(map)

        map.on('moveend', () => {
            if (onMoveEndRef.current) {
                const center = map.getCenter()
                onMoveEndRef.current(center.lat.toFixed(4), center.lng.toFixed(4))
            }
        })
        
        mapRef.current = map
    }, [])

    useEffect(() => {
        if (mapRef.current) mapRef.current.setView([lat, lon], 15, { animate: true })
    }, [lat, lon])

    useEffect(() => {
        const L = require("leaflet")
        const map = mapRef.current
        if (!map) return
        layersRef.current.forEach(l => map.removeLayer(l))
        layersRef.current = []

        const push = (l) => { l.addTo(map); layersRef.current.push(l) }

        // Analysis circle
        push(L.circle([lat, lon], {
            radius, color: "#00d4aa", weight: 1,
            dashArray: "5 4", fillColor: "#00d4aa", fillOpacity: 0.04
        }))

        if (!data) return
        const geojson = data.metrics?.geojson
        const zoneGrid = data.zone_grid || []
        const inters = data.metrics?.intersections || []
        const deadEnds = data.metrics?.dead_ends || []

        if (activeLayer === "roads" && geojson) {
            push(L.geoJSON(geojson, {
                style: f => ({
                    color: f.properties.color || "#00d4aa",
                    weight: Math.max(1.5, 6 - (f.properties.level || 5)),
                    opacity: 0.9,
                }),
                onEachFeature: (f, layer) => {
                    const p = f.properties
                    layer.bindPopup(`<div style="font-family:monospace;font-size:12px">
            <b style="color:${p.color}">${p.label}</b><br/>
            ${p.name ? `<span style="color:#aaa">${p.name}</span><br/>` : ""}
            <span style="color:#666">Length: </span><b>${p.length_km} km</b>
          </div>`)
                    layer.on("mouseover", () => layer.setStyle({ opacity: 1, weight: Math.max(2, 8 - (p.level || 5)) }))
                    layer.on("mouseout", () => layer.setStyle({ opacity: 0.9, weight: Math.max(1.5, 6 - (p.level || 5)) }))
                }
            }))
        }

        if (activeLayer === "heatmap" && zoneGrid.length) {
            zoneGrid.forEach(cell => {
                const n = cell.normalized
                const r = Math.round(n > .5 ? 255 : n * 2 * 255)
                const g = Math.round(n < .5 ? n * 2 * 180 : (1 - n) * 2 * 180)
                const b = Math.round(n < .5 ? 200 - n * 2 * 200 : 0)
                const rect = L.rectangle(
                    [[cell.bounds.lat_min, cell.bounds.lon_min], [cell.bounds.lat_max, cell.bounds.lon_max]],
                    { color: "transparent", fillColor: `rgb(${r},${g},${b})`, fillOpacity: .1 + n * .65, weight: 0 }
                )
                rect.bindPopup(`<div style="font-family:monospace;font-size:11px">
          Road density: <b>${cell.length_km} km</b><br/>Relative: <b>${(n * 100).toFixed(0)}%</b>
        </div>`)
                push(rect)
            })
        }

        if (activeLayer === "intersections") {
            inters.slice(0, 400).forEach(([ilat, ilon]) =>
                push(L.circleMarker([ilat, ilon], {
                    radius: 4, color: "#00d4aa", fillColor: "#00d4aa", fillOpacity: .8, weight: 1
                }).bindPopup(`<span style="font-family:monospace;font-size:11px;color:#00d4aa">Intersection</span>`))
            )
            deadEnds.slice(0, 200).forEach(([dlat, dlon]) =>
                push(L.circleMarker([dlat, dlon], {
                    radius: 3, color: "#ea4335", fillColor: "#ea4335", fillOpacity: .8, weight: 1
                }).bindPopup(`<span style="font-family:monospace;font-size:11px;color:#ea4335">Dead End</span>`))
            )
        }
    }, [data, activeLayer, lat, lon, radius])

    return <div ref={containerRef} style={{ width: "100%", height: "100%" }} />
}