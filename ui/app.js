/* Alternate evaluator UI. Uses the same API and native Leaflet map endpoint. */
'use strict';
let API_BASE = window.location.origin;
if (!/^https?:/.test(API_BASE)) API_BASE = 'http://127.0.0.1:8000';
let availableTrips = [];
let generation = 0;
let activePrediction = null;
const byId = id => document.getElementById(id);
const apiUrlInput = byId('api-url');
const tripSelect = byId('trip-select');
const chkViterbi = byId('chk-viterbi');
const btnEvaluate = byId('btn-evaluate');
const mapFrame = byId('map-frame');
const mapPlaceholder = byId('map-placeholder');
apiUrlInput.value = API_BASE;
const percent = value => Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : 'N/A';

function clearResult(message = 'Select a trip and run the prediction pipeline.') {
  generation += 1;
  activePrediction = null;
  byId('eval-spinner').classList.add('hidden');
  ['val-accuracy', 'val-continuity', 'val-missrate', 'val-latency'].forEach(id => { byId(id).textContent = '—'; });
  byId('sub-accuracy').textContent = 'Awaiting evaluation';
  byId('route-tags').replaceChildren();
  mapFrame.classList.add('hidden');
  mapFrame.removeAttribute('src');
  mapPlaceholder.textContent = message;
  mapPlaceholder.classList.remove('hidden');
}

async function jsonRequest(path, options = {}, timeout = 10000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeout);
  try {
    const response = await fetch(API_BASE + path, {...options, signal: controller.signal});
    const data = await response.json();
    if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : `HTTP ${response.status}`);
    return data;
  } finally {
    clearTimeout(timer);
  }
}

async function checkHealthAndLoadTrips() {
  clearResult();
  const requestGeneration = generation;
  byId('conn-text').textContent = 'Connecting…';
  byId('status-dot').className = 'status-dot';
  btnEvaluate.disabled = true;
  try {
    const parsed = new URL(apiUrlInput.value);
    if (!['http:', 'https:'].includes(parsed.protocol)) throw new Error('Use an HTTP or HTTPS API address.');
    API_BASE = parsed.href.replace(/\/+$/, '');
    const health = await jsonRequest('/api/v1/health');
    const catalog = await jsonRequest('/api/v1/trips');
    if (requestGeneration !== generation) return;
    if (!health.model_loaded) throw new Error('The API has not loaded the model.');
    availableTrips = catalog.trips;
    tripSelect.replaceChildren();
    ['evaluator_test', 'student_test', 'train'].forEach(partition => {
      const trips = availableTrips.filter(trip => trip.partition === partition);
      if (!trips.length) return;
      const group = document.createElement('optgroup');
      group.label = `${partition} (${trips.length} trips)`;
      trips.forEach(trip => {
        const option = document.createElement('option');
        option.value = `${trip.partition}/${trip.trip_id}`;
        option.textContent = `Trip ${trip.trip_id} (${trip.points} GPS observations)`;
        group.appendChild(option);
      });
      tripSelect.appendChild(group);
    });
    byId('status-dot').className = 'status-dot connected';
    byId('conn-text').textContent = `Connected · ${health.model_name}`;
    tripSelect.disabled = false;
    btnEvaluate.disabled = availableTrips.length === 0;
    updateMetadata();
  } catch (error) {
    if (requestGeneration !== generation) return;
    byId('status-dot').className = 'status-dot disconnected';
    byId('conn-text').textContent = `API unavailable: ${error.message}`;
    tripSelect.replaceChildren();
    tripSelect.disabled = true;
    btnEvaluate.disabled = true;
  }
}

function selectedTrip() {
  return availableTrips.find(trip => `${trip.partition}/${trip.trip_id}` === tripSelect.value);
}

function updateMetadata() {
  clearResult();
  const trip = selectedTrip();
  if (!trip) return;
  byId('meta-partition').textContent = trip.partition;
  byId('meta-points').textContent = trip.points.toLocaleString();
  byId('meta-nodes').textContent = trip.nodes.toLocaleString();
  byId('meta-arcs').textContent = trip.arcs.toLocaleString();
  byId('meta-route').textContent = trip.reference_available ? (trip.reference_valid ? 'Valid, server-side only' : 'Invalid server-side reference') : 'Unavailable';
  chkViterbi.disabled = trip.partition === 'evaluator_test';
  if (chkViterbi.disabled) chkViterbi.checked = true;
  btnEvaluate.disabled = false;
}

async function runEvaluation() {
  const trip = selectedTrip();
  if (!trip) return;
  clearResult(`Processing trip ${trip.trip_id}…`);
  const requestGeneration = generation;
  activePrediction = requestGeneration;
  btnEvaluate.disabled = true;
  byId('eval-spinner').classList.remove('hidden');
  try {
    const data = await jsonRequest('/api/v1/evaluate', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({trip_id: trip.trip_id, partition: trip.partition, use_viterbi: chkViterbi.checked})
    }, 300000);
    if (requestGeneration !== generation) return;
    const metrics = data.metrics;
    byId('val-accuracy').textContent = percent(metrics.point_accuracy);
    byId('val-continuity').textContent = percent(metrics.route_continuity);
    byId('val-missrate').textContent = percent(metrics.candidate_miss_rate);
    byId('val-latency').textContent = Number.isFinite(metrics.inference_time_ms) ? `${metrics.inference_time_ms.toFixed(0)} ms` : 'N/A';
    byId('sub-accuracy').textContent = `${metrics.evaluation_observations || 0} aligned labels · ${percent(metrics.evaluation_coverage)} coverage · classifier Top-1 ${percent(metrics.top1_accuracy)}`;
    const preview = data.predicted_route_preview || [];
    preview.forEach(arc => {
      const tag = document.createElement('span');
      tag.className = 'arc-tag'; tag.textContent = `Arc ${arc}`;
      byId('route-tags').appendChild(tag);
    });
    const remaining = Math.max(0, metrics.sequence_length - preview.length);
    if (remaining) byId('route-tags').appendChild(document.createTextNode(` +${remaining} more route segments`));
    const mapURL = new URL(data.map_url, API_BASE + '/');
    if (mapURL.origin !== new URL(API_BASE).origin) throw new Error('Map URL does not belong to the configured API.');
    mapFrame.src = mapURL.href;
    mapFrame.title = `Leaflet predicted route for trip ${trip.trip_id}`;
    mapFrame.classList.remove('hidden');
    mapPlaceholder.classList.add('hidden');
    byId('map-heading').textContent = `Leaflet route · ${trip.partition}/${trip.trip_id}`;
  } catch (error) {
    if (requestGeneration !== generation) return;
    clearResult(`Prediction failed: ${error.message}`);
    btnEvaluate.disabled = !selectedTrip();
  } finally {
    if (activePrediction === requestGeneration) {
      activePrediction = null;
      byId('eval-spinner').classList.add('hidden');
      btnEvaluate.disabled = !selectedTrip();
    }
  }
}

byId('btn-reconnect').addEventListener('click', checkHealthAndLoadTrips);
apiUrlInput.addEventListener('input', () => {
  clearResult('API address changed. Click Connect.');
  availableTrips = []; tripSelect.replaceChildren(); btnEvaluate.disabled = true;
});
tripSelect.addEventListener('change', updateMetadata);
chkViterbi.addEventListener('change', () => { clearResult(); btnEvaluate.disabled = !selectedTrip(); });
btnEvaluate.addEventListener('click', runEvaluation);
checkHealthAndLoadTrips();
