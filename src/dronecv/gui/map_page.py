"""HTML page hosted inside the desktop GUI's QWebEngineView.

Leaflet (OSM tiles, mandatory attribution) + Leaflet.draw for the AOI mask
polygon. The page talks to Python through QWebChannel:
    JS -> Python: maskDrawn(geojson string), mapClicked(lat, lon)
    Python -> JS: setView(lat, lon, zoom), showBBox(s, w, n, e)

Leaflet is loaded from the unpkg CDN — the GIS workflow requires network
anyway (Overpass/DEM downloads); there is no offline map mode.
"""

MAP_HTML = """<!doctype html>
<html><head>
<meta charset="utf-8"/>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<link rel="stylesheet" href="https://unpkg.com/leaflet-draw@1.0.4/dist/leaflet.draw.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://unpkg.com/leaflet-draw@1.0.4/dist/leaflet.draw.js"></script>
<script src="qrc:///qtwebchannel/qwebchannel.js"></script>
<style>html, body, #map { height: 100%; margin: 0; }</style>
</head><body>
<div id="map"></div>
<script>
const map = L.map('map').setView([43.32, 11.33], 13);
L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 19,
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
}).addTo(map);

const drawn = new L.FeatureGroup().addTo(map);
map.addControl(new L.Control.Draw({
    draw: { polyline: false, circle: false, marker: false, circlemarker: false,
            rectangle: { shapeOptions: { color: '#0b3954' } },
            polygon: { shapeOptions: { color: '#0b3954' } } },
    edit: { featureGroup: drawn }
}));

let bridge = null;
new QWebChannel(qt.webChannelTransport, (channel) => { bridge = channel.objects.bridge; });

map.on(L.Draw.Event.CREATED, (e) => {
    drawn.clearLayers();
    drawn.addLayer(e.layer);
    if (bridge) bridge.maskDrawn(JSON.stringify(e.layer.toGeoJSON()));
});
map.on(L.Draw.Event.EDITED, (e) => {
    e.layers.eachLayer((layer) => {
        if (bridge) bridge.maskDrawn(JSON.stringify(layer.toGeoJSON()));
    });
});
map.on(L.Draw.Event.DELETED, () => {
    if (bridge) bridge.maskDeleted();
});

// Called from Python.
function setView(lat, lon, zoom) { map.setView([lat, lon], zoom); }
function showBBox(s, w, n, e) {
    // PREVIEW ONLY: frame the searched place with a dashed hint rectangle.
    // It does NOT become the selection — the user draws the exact AOI
    // (a whole-city bbox from geocoding must never start a build/coverage).
    drawn.clearLayers();
    drawn.addLayer(L.rectangle([[s, w], [n, e]],
        { color: '#888', dashArray: '6 6', fill: false }));
    map.fitBounds([[s, w], [n, e]]);
}
</script>
</body></html>"""
