/* Native Leaflet route viewer. Each road segment remains independent. */
(() => {
  'use strict';
  const data = JSON.parse(document.getElementById('map-data').textContent);
  const map = L.map('route-map', {preferCanvas: true, zoomControl: true}).setView([0, 0], 2);
  const networkLayer = L.featureGroup().addTo(map);
  (data.network_segments || []).forEach(segment => {
    L.polyline(segment, {color: '#727f89', weight: 5, opacity: .65}).bindTooltip('Supplied road network').addTo(networkLayer);
  });
  const routeLayer = L.featureGroup().addTo(map);
  const gpsLayer = L.featureGroup().addTo(map);
  const endpoints = L.featureGroup().addTo(map);
  const bounds = L.latLngBounds([]);
  const segments = data.predicted_segments || [];
  segments.forEach((segment, index) => {
    if (segment.length < 2) return;
    L.polyline(segment, {color: '#087f5b', weight: 6, opacity: .92, lineCap: 'round'})
      .bindTooltip(`Predicted road segment ${index + 1}`).addTo(routeLayer);
    segment.forEach(point => bounds.extend(point));
  });
  const gps = data.gps_coords || [];
  const step = Math.max(1, Math.ceil(gps.length / 1200));
  gps.forEach((point, index) => {
    bounds.extend(point);
    if (index % step !== 0 && index !== gps.length - 1) return;
    L.circleMarker(point, {radius: 3, color: '#2563eb', weight: 1, fillOpacity: .7})
      .bindTooltip(`GPS observation ${index}`).addTo(gpsLayer);
  });
  if (segments.length) {
    const start = segments[0][0];
    const last = segments[segments.length - 1];
    const end = last[last.length - 1];
    [[start, '#15803d', 'Start'], [end, '#dc2626', 'End']].forEach(([point, color, label]) => {
      L.circleMarker(point, {radius: 8, color: '#fff', weight: 3, fillColor: color, fillOpacity: 1})
        .bindTooltip(label, {permanent: true, direction: 'top'}).addTo(endpoints);
    });
  }
  routeLayer.bringToFront();
  endpoints.bringToFront();
  const plain = L.layerGroup();
  const street = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}', {
    attribution: 'Tiles &copy; Esri — Sources: Esri, HERE, Garmin, USGS, NGA, EPA, NPS',
    maxZoom: 19, opacity: .35
  }).addTo(map);
  const status = document.getElementById('map-status');
  street.on('tileerror', () => {
    map.removeLayer(street);
    plain.addTo(map);
    status.textContent = 'Street tiles unavailable. GPS points and predicted roads remain visible.';
  });
  map.on('baselayerchange', event => {
    if (event.layer === plain) status.textContent = 'Offline background · road geometry and GPS remain interactive';
    else status.textContent = 'Street background requires internet';
  });
  L.control.layers({'Offline background': plain, 'Street map (online)': street}, {
    'Supplied road network': networkLayer, 'Predicted roads': routeLayer, 'Raw GPS observations': gpsLayer, 'Start / end': endpoints
  }, {collapsed: false}).addTo(map);
  L.control.scale({imperial: false}).addTo(map);
  function fit() {
    if (bounds.isValid()) map.fitBounds(bounds, {padding: [35, 35], maxZoom: 17});
  }
  document.getElementById('fit-route').addEventListener('click', fit);
  document.getElementById('full-screen').addEventListener('click', async () => {
    const panel = document.getElementById('map-panel');
    try {
      if (document.fullscreenElement) await document.exitFullscreen();
      else await panel.requestFullscreen();
    } catch (_) {
      status.textContent = 'Fullscreen is unavailable in this browser. Zoom and pan remain available.';
    }
  });
  document.addEventListener('fullscreenchange', () => setTimeout(() => {map.invalidateSize(); fit();}, 100));
  new ResizeObserver(() => map.invalidateSize()).observe(document.getElementById('route-map'));
  document.getElementById('map-title').textContent = `Trip ${data.trip_id || ''} · ${segments.length} predicted road segments`;
  fit();
  document.body.dataset.mapReady = 'true';
  document.body.dataset.routeSegments = String(segments.length);
  document.body.dataset.gpsObservations = String(gps.length);
})();
