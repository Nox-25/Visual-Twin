async function fetchState() {
    try {
        const response = await fetch('/api/state');
        const state = await response.json();
        updateDashboard(state);
    } catch (error) {
        console.error("Error fetching state:", error);
    }
}

function updateDashboard(state) {
    // Update Header Status
    const dt = state.digital_twin;
    const inf = state.inference;

    document.getElementById('state-dot').className = `dot ${dt.current_state}`;
    document.getElementById('state-text').innerText = dt.current_state.replace('_', ' ').toUpperCase();
    document.getElementById('fps-counter').innerText = `FPS: ${inf.fps}`;

    // Update Predictions
    const predList = document.getElementById('prediction-list');
    predList.innerHTML = '';
    inf.predictions.forEach((pred, index) => {
        const li = document.createElement('li');
        const isDefect = index === 0 && pred.prob > 0.5 && !pred.class.toLowerCase().includes('normal');
        li.className = isDefect ? 'prediction-item defect' : 'prediction-item';

        li.innerHTML = `
            <div style="display:flex; justify-content:space-between;">
                <span title="${pred.class}">${pred.class.length > 30 ? pred.class.substring(0,30)+'...' : pred.class}</span>
                <span>${(pred.prob * 100).toFixed(1)}%</span>
            </div>
            <div class="bar-bg">
                <div class="bar-fill" style="width: ${pred.prob * 100}%"></div>
            </div>
        `;
        predList.appendChild(li);
    });

    // Update Ontology
    const ontologyList = document.getElementById('ontology-list');
    ontologyList.innerHTML = '';
    dt.active_ontology.forEach((desc, idx) => {
        const li = document.createElement('li');
        li.className = 'ontology-item';
        li.innerHTML = `
            <span title="${desc}">${desc.length > 30 ? desc.substring(0,30)+'...' : desc}</span>
            <button onclick="removeOntologyClass(${idx})" title="Remove">×</button>
        `;
        ontologyList.appendChild(li);
    });

    // Update Anomaly Log
    const anomalyList = document.getElementById('anomaly-list');
    if (dt.recent_anomalies.length > 0) {
        anomalyList.innerHTML = '';
        // Reverse to show newest at top
        const reversed = [...dt.recent_anomalies].reverse();
        reversed.forEach(anom => {
            const li = document.createElement('li');
            li.className = 'anomaly-item';
            const date = new Date(anom.timestamp);
            li.innerHTML = `
                <span class="anomaly-time">${date.toLocaleTimeString()} | Conf: ${(anom.confidence_score*100).toFixed(0)}%</span>
                Detected: ${anom.detected_class}
            `;
            anomalyList.appendChild(li);
        });
    }
}

async function addOntologyClass() {
    const input = document.getElementById('new-class-input');
    const desc = input.value.trim();
    if (!desc) return;

    try {
        await fetch('/api/ontology', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ action: 'add', description: desc })
        });
        input.value = '';
        fetchState(); // Immediate refresh
    } catch (e) {
        console.error("Error adding class:", e);
    }
}

async function removeOntologyClass(idx) {
    try {
        await fetch('/api/ontology', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ action: 'remove', index: idx })
        });
        fetchState(); // Immediate refresh
    } catch (e) {
        console.error("Error removing class:", e);
    }
}

async function triggerSnapshot() {
    try {
        const response = await fetch('/api/snapshot', { method: 'POST' });
        const result = await response.json();
        alert(`Snapshot saved to: ${result.filepath}`);
    } catch (e) {
        console.error("Error saving snapshot:", e);
        alert("Failed to save snapshot.");
    }
}

// Poll state every 2 seconds as required
setInterval(fetchState, 2000);
fetchState(); // Initial call
