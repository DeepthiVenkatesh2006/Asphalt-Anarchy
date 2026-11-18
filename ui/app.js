let API_BASE = window.location.origin;
if (!API_BASE || API_BASE.startsWith("file") || API_BASE.includes("localhost:3000")) {
  API_BASE = "http://127.0.0.1:8000";
}

let availableTrips = [];

// DOM elements
const apiUrlInput = document.getElementById("api-url");
const btnReconnect = document.getElementById("btn-reconnect");
const statusDot = document.getElementById("status-dot");
const connText = document.getElementById("conn-text");
const tripSelect = document.getElementById("trip-select");
const chkViterbi = document.getElementById("chk-viterbi");
const btnEvaluate = document.getElementById("btn-evaluate");
const evalSpinner = document.getElementById("eval-spinner");

// Metadata elements
const metaPartition = document.getElementById("meta-partition");
const metaPoints = document.getElementById("meta-points");
const metaNodes = document.getElementById("meta-nodes");
const metaArcs = document.getElementById("meta-arcs");
const metaRoute = document.getElementById("meta-route");

// Metrics elements
const valAccuracy = document.getElementById("val-accuracy");
const subAccuracy = document.getElementById("sub-accuracy");
const valContinuity = document.getElementById("val-continuity");
const valMissrate = document.getElementById("val-missrate");
const valLatency = document.getElementById("val-latency");
const routeTags = document.getElementById("route-tags");
const mapFrame = document.getElementById("map-frame");
const mapPlaceholder = document.getElementById("map-placeholder");
const mapHeading = document.getElementById("map-heading");

apiUrlInput.value = API_BASE;

async function checkHealthAndLoadTrips() {
  API_BASE = apiUrlInput.value.replace(/\/+$/, "");
  connText.textContent = "Connecting...";
  statusDot.className = "status-dot";

  try {
    const healthRes = await fetch(`${API_BASE}/api/v1/health`);
    if (!healthRes.ok) throw new Error("Health check failed");
    const healthData = await healthRes.json();

    statusDot.className = "status-dot connected";
    connText.textContent = `Connected (${healthData.service} v${healthData.version})`;

    // Load Trips
    const tripsRes = await fetch(`${API_BASE}/api/v1/trips`);
    if (!tripsRes.ok) throw new Error("Could not fetch trips");
    const tripsData = await tripsRes.json();
    availableTrips = tripsData.trips;

    populateTripSelect();
  } catch (err) {
    statusDot.className = "status-dot disconnected";
    connText.textContent = "Disconnected / Cannot Reach Laptop 1";
    tripSelect.innerHTML = `<option disabled selected>Failed to load trips</option>`;
    btnEvaluate.disabled = true;
    console.error("API Connection error:", err);
  }
}

function populateTripSelect() {
  tripSelect.innerHTML = "";
  // Show student_test trips first as they are reserved for evaluation
  const testTrips = availableTrips.filter(t => t.partition === "student_test");
  const trainTrips = availableTrips.filter(t => t.partition === "train");

  const optGroupTest = document.createElement("optgroup");
  optGroupTest.label = "Reserved Evaluation Partitions (Student Test - 10 Trips)";

  testTrips.forEach(t => {
    const opt = document.createElement("option");
    opt.value = t.trip_id;
    opt.textContent = `Trip ${t.trip_id} (${t.points} GPS pts, ${t.arcs} arcs)`;
    opt.dataset.partition = t.partition;
    optGroupTest.appendChild(opt);
  });
  tripSelect.appendChild(optGroupTest);

  if (trainTrips.length > 0) {
    const optGroupTrain = document.createElement("optgroup");
    optGroupTrain.label = "Training Partitions (80 Trips)";
    trainTrips.slice(0, 15).forEach(t => {
      const opt = document.createElement("option");
      opt.value = t.trip_id;
      opt.textContent = `Trip ${t.trip_id} (${t.points} pts)`;
      opt.dataset.partition = t.partition;
      optGroupTrain.appendChild(opt);
    });
    tripSelect.appendChild(optGroupTrain);
  }

  tripSelect.disabled = false;
  btnEvaluate.disabled = false;
  updateMetadata();
}

function updateMetadata() {
  const selectedId = tripSelect.value;
  const trip = availableTrips.find(t => t.trip_id === selectedId);
  if (!trip) return;

  metaPartition.textContent = trip.partition;
  metaPoints.textContent = trip.points.toLocaleString();
  metaNodes.textContent = trip.nodes.toLocaleString();
  metaArcs.textContent = trip.arcs.toLocaleString();
  metaRoute.textContent = trip.route_arcs > 0 ? `${trip.route_arcs} segments (Verified)` : "No ground truth";
}

async function runEvaluation() {
  const tripId = tripSelect.value;
  if (!tripId) return;

  const trip = availableTrips.find(t => t.trip_id === tripId);
  const partition = trip ? trip.partition : "student_test";
  const useViterbi = chkViterbi.checked;

  btnEvaluate.disabled = true;
  evalSpinner.classList.remove("hidden");
  mapPlaceholder.innerHTML = `<p>Processing trip <strong>${tripId}</strong> through candidate generation & Viterbi decoding...</p>`;
  mapPlaceholder.classList.remove("hidden");
  mapFrame.classList.add("hidden");

  try {
    const res = await fetch(`${API_BASE}/api/v1/evaluate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        trip_id: tripId,
        partition: partition,
        use_viterbi: useViterbi
      })
    });

    if (!res.ok) {
      const errData = await res.json();
      throw new Error(errData.detail || "Evaluation failed");
    }

    const data = await res.json();
    displayResults(data);
  } catch (err) {
    alert(`Evaluation Error: ${err.message}`);
    mapPlaceholder.innerHTML = `<p style="color:#e63946;">Failed to evaluate: ${err.message}</p>`;
  } finally {
    btnEvaluate.disabled = false;
    evalSpinner.classList.add("hidden");
  }
}

function displayResults(data) {
  const m = data.metrics;
  
  // Scorecard
  valAccuracy.textContent = `${(m.top1_accuracy * 100).toFixed(1)}%`;
  subAccuracy.textContent = `${m.total_observations} points evaluated`;
  valContinuity.textContent = `${(m.route_continuity * 100).toFixed(1)}%`;
  valMissrate.textContent = `${(m.candidate_miss_rate * 100).toFixed(1)}%`;
  valLatency.textContent = `${m.inference_time_ms.toFixed(0)} ms`;

  // Color code based on accuracy
  if (m.top1_accuracy >= 0.85) {
    valAccuracy.style.color = "#2a9d8f";
  } else if (m.top1_accuracy >= 0.70) {
    valAccuracy.style.color = "#e76f51";
  } else {
    valAccuracy.style.color = "#e63946";
  }

  // Map
  mapHeading.textContent = `Map Matching Result for Trip ${data.trip_id} (${data.partition})`;
  mapPlaceholder.classList.add("hidden");
  mapFrame.classList.remove("hidden");
  mapFrame.src = `${API_BASE}${data.map_url}`;

  // Sequence Tags
  if (data.predicted_route_preview && data.predicted_route_preview.length > 0) {
    routeTags.innerHTML = data.predicted_route_preview
      .map((arc, i) => `<span class="arc-badge">#${i + 1}: Arc ${arc}</span>`)
      .join(" ");
    if (data.total_predicted_arcs > data.predicted_route_preview.length) {
      routeTags.innerHTML += ` <span style="font-size:12px; color:#6c757d;">... (+${data.total_predicted_arcs - data.predicted_route_preview.length} more)</span>`;
    }
  } else {
    routeTags.innerHTML = `<em>No arcs predicted.</em>`;
  }
}

// Event Listeners
btnReconnect.addEventListener("click", checkHealthAndLoadTrips);
tripSelect.addEventListener("change", updateMetadata);
btnEvaluate.addEventListener("click", runEvaluation);

// Auto-connect on load
window.addEventListener("DOMContentLoaded", checkHealthAndLoadTrips);
