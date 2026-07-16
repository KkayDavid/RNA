// Configuración global de Chart.js
Chart.defaults.color = '#a1a1aa';
Chart.defaults.borderColor = 'rgba(255, 255, 255, 0.05)';
Chart.defaults.font.family = "'Inter', sans-serif";

let currentStationId = null;
let histChartInstance = null;
let forecastChartInstance = null;
let lossChartInstance = null;

let currentHistoryData = null;

// Elementos del DOM
const els = {
    stationList: document.getElementById('station-list'),
    stationName: document.getElementById('station-name'),
    stationMeta: document.getElementById('station-meta'),
    connStatus: document.getElementById('conn-status'),
    pulse: document.querySelector('.pulse'),
    loader: document.getElementById('main-loader'),
    kpis: document.getElementById('kpis-container'),
    histColSelector: document.getElementById('hist-col-selector'),
    targetColSelector: document.getElementById('target-col-selector'),
    trainBtn: document.getElementById('train-btn'),
    aiResults: document.getElementById('ai-results'),
    epochs: document.getElementById('epochs-input'),
    periods: document.getElementById('periods-input')
};

// Inicialización
async function init() {
    await loadStations();
}

// Cargar lista de estaciones
async function loadStations() {
    try {
        const res = await fetch('/api/stations');
        const stations = await res.json();
        
        els.stationList.innerHTML = '';
        stations.forEach(s => {
            const li = document.createElement('li');
            li.className = 'station-item';
            li.innerHTML = `<i class="fa-solid fa-tower-broadcast"></i> ${s.nombre}`;
            li.onclick = () => selectStation(s.id, li);
            els.stationList.appendChild(li);
        });

        // Autoseleccionar la primera
        if (stations.length > 0) {
            selectStation(stations[0].id, els.stationList.firstChild);
        }
    } catch (e) {
        els.stationList.innerHTML = '<li class="loading-text" style="color:#ef4444">Error cargando estaciones</li>';
    }
}

// Seleccionar estación
async function selectStation(id, el) {
    document.querySelectorAll('.station-item').forEach(li => li.classList.remove('active'));
    if (el) el.classList.add('active');
    
    currentStationId = id;
    els.loader.classList.remove('hidden');
    els.aiResults.classList.add('hidden');
    els.pulse.className = 'pulse';
    els.connStatus.textContent = 'Cargando datos...';

    try {
        const res = await fetch(`/api/station/${id}`);
        if (!res.ok) throw new Error('Error al conectar con la estación');
        
        const data = await res.json();
        
        // Actualizar UI
        els.stationName.textContent = data.info.nombre;
        els.stationMeta.innerHTML = `<i class="fa-solid fa-satellite-dish"></i> Canal ${data.info.canal} &bull; <i class="fa-solid fa-database"></i> ${data.registros} registros`;
        els.pulse.className = 'pulse online';
        els.connStatus.textContent = 'Conectado';

        renderKPIs(data.last_value, data.alerts);
        updateSelectors(data.columns);
        
        currentHistoryData = data.history;
        renderHistoricalChart(els.histColSelector.value || data.columns[0]);

    } catch (e) {
        els.pulse.className = 'pulse error';
        els.connStatus.textContent = 'Error de conexión';
        alert(e.message);
    } finally {
        els.loader.classList.add('hidden');
    }
}

function renderKPIs(lastVal, alerts) {
    els.kpis.innerHTML = '';
    const ignoreCols = ['fecha', 'entry_id'];
    
    // Crear un mapa de alertas rápido
    const alertMap = {};
    alerts.forEach(a => alertMap[a.variable] = a);

    Object.entries(lastVal).forEach(([key, val]) => {
        if (ignoreCols.includes(key)) return;
        
        let valStr = typeof val === 'number' ? val.toFixed(2) : val;
        let alertHtml = '';
        let borderStyle = '';
        
        if (alertMap[key]) {
            const al = alertMap[key];
            borderStyle = `border-bottom: 3px solid ${al.nivel === 'Alto' ? '#ef4444' : '#f59e0b'};`;
            alertHtml = `<div style="color: ${al.nivel === 'Alto' ? '#ef4444' : '#f59e0b'}; font-size: 0.7rem; margin-top: 5px;">
                <i class="fa-solid fa-triangle-exclamation"></i> ${al.umbral}
            </div>`;
        }

        els.kpis.innerHTML += `
            <div class="kpi-card" style="${borderStyle}">
                <div class="kpi-title">${key}</div>
                <div class="kpi-value">${valStr}</div>
                ${alertHtml}
            </div>
        `;
    });
}

function updateSelectors(columns) {
    els.histColSelector.innerHTML = '';
    els.targetColSelector.innerHTML = '';
    
    columns.forEach(c => {
        els.histColSelector.innerHTML += `<option value="${c}">${c}</option>`;
        els.targetColSelector.innerHTML += `<option value="${c}">${c}</option>`;
    });

    // Seleccionar temperatura por defecto si existe
    if (columns.includes('temperatura')) {
        els.histColSelector.value = 'temperatura';
        els.targetColSelector.value = 'temperatura';
    }
}

els.histColSelector.addEventListener('change', (e) => {
    if (currentHistoryData) renderHistoricalChart(e.target.value);
});

function renderHistoricalChart(colName) {
    if (!currentHistoryData || !currentHistoryData[colName]) return;
    
    if (histChartInstance) histChartInstance.destroy();
    
    const ctx = document.getElementById('historicalChart').getContext('2d');
    
    // Gradiente
    let gradient = ctx.createLinearGradient(0, 0, 0, 300);
    gradient.addColorStop(0, 'rgba(59, 130, 246, 0.2)');
    gradient.addColorStop(1, 'rgba(59, 130, 246, 0)');

    histChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: currentHistoryData.labels,
            datasets: [{
                label: colName,
                data: currentHistoryData[colName],
                borderColor: '#3b82f6',
                backgroundColor: gradient,
                borderWidth: 2,
                pointRadius: 0,
                pointHitRadius: 10,
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { ticks: { maxTicksLimit: 10, maxRotation: 0 }, grid: { display: false } },
                y: { grid: { color: 'rgba(255,255,255,0.05)' } }
            },
            interaction: { mode: 'index', intersect: false }
        }
    });
}

// Entrenar modelo
els.trainBtn.addEventListener('click', async () => {
    const targetCol = els.targetColSelector.value;
    const epochs = els.epochs.value;
    const periods = els.periods.value;

    els.loader.querySelector('p').textContent = `Entrenando PyTorch (${epochs} épocas)...`;
    els.loader.classList.remove('hidden');
    els.trainBtn.disabled = true;

    try {
        const res = await fetch('/api/train', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ target_col: targetCol, epochs: epochs, periods: periods })
        });
        
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || 'Error entrenando');

        renderAIResults(data, targetCol);
        els.aiResults.classList.remove('hidden');

        // Scroll a los resultados
        setTimeout(() => {
            els.aiResults.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 100);

    } catch (e) {
        alert("Error de Neural AI: " + e.message);
    } finally {
        els.loader.classList.add('hidden');
        els.trainBtn.disabled = false;
    }
});

function renderAIResults(data, targetCol) {
    const m = data.metrics;
    document.getElementById('metrics-container').innerHTML = `
        <div class="metric-box">
            <div class="metric-val">${m['R²']?.toFixed(3)}</div>
            <div class="metric-lbl">Score R²</div>
        </div>
        <div class="metric-box">
            <div class="metric-val">${m['MAPE (%)']?.toFixed(1)}%</div>
            <div class="metric-lbl">Error (MAPE)</div>
        </div>
        <div class="metric-box">
            <div class="metric-val">${m['RMSE']?.toFixed(2)}</div>
            <div class="metric-lbl">RMSE</div>
        </div>
        <div class="metric-box">
            <div class="metric-val">${m['epochs_run']}</div>
            <div class="metric-lbl">Épocas</div>
        </div>
    `;

    renderForecastChart(data.forecast, targetCol);
    renderLossChart(data.loss_history);
    
    // Acciones
    const act = data.analysis.plan_accion || [];
    document.getElementById('actions-container').innerHTML = act.map(a => `
        <div class="action-item">
            <div class="action-title prio-${a.prioridad}"><i class="fa-solid fa-circle-exclamation"></i> ${a.accion}</div>
            <div class="action-desc">${a.detalle}</div>
        </div>
    `).join('');

    // Importancia
    const imp = data.importance || {};
    const maxImp = Math.max(...Object.values(imp));
    const impHtml = Object.entries(imp).slice(0, 6).map(([k, v]) => `
        <div class="imp-row">
            <div class="imp-lbl">${k}</div>
            <div class="imp-track"><div class="imp-fill" style="width: ${(v/maxImp)*100}%"></div></div>
            <div class="imp-val">${((v/maxImp)*100).toFixed(0)}%</div>
        </div>
    `).join('');
    document.getElementById('importance-container').innerHTML = impHtml;
}

function renderForecastChart(fc, colName) {
    if (forecastChartInstance) forecastChartInstance.destroy();
    
    const labels = fc.map(f => f.periodo);
    const preds = fc.map(f => f.prediccion);
    const infs = fc.map(f => f.intervalo_inf);
    const sups = fc.map(f => f.intervalo_sup);

    forecastChartInstance = new Chart(document.getElementById('forecastChart'), {
        type: 'bar',
        data: {
            labels,
            datasets: [
                {
                    label: 'Predicción',
                    data: preds,
                    backgroundColor: '#3b82f6',
                    borderRadius: 4,
                    order: 2
                },
                {
                    label: 'Límite Superior',
                    data: sups,
                    type: 'line',
                    borderColor: 'rgba(255,255,255,0.2)',
                    borderDash: [5, 5],
                    pointRadius: 0,
                    fill: false,
                    order: 1
                },
                {
                    label: 'Límite Inferior',
                    data: infs,
                    type: 'line',
                    borderColor: 'rgba(255,255,255,0.2)',
                    borderDash: [5, 5],
                    pointRadius: 0,
                    fill: '-1',
                    backgroundColor: 'rgba(255,255,255,0.05)',
                    order: 1
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: { x: { grid: { display: false } }, y: { grid: { color: 'rgba(255,255,255,0.05)' } } }
        }
    });
}

function renderLossChart(history) {
    if (lossChartInstance) lossChartInstance.destroy();
    if (!history || !history.train) return;

    const labels = history.train.map((_, i) => i + 1);
    
    lossChartInstance = new Chart(document.getElementById('lossChart'), {
        type: 'line',
        data: {
            labels,
            datasets: [
                {
                    label: 'Train Loss',
                    data: history.train,
                    borderColor: '#3b82f6',
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0.3
                },
                {
                    label: 'Val Loss',
                    data: history.val || [],
                    borderColor: '#10b981',
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0.3
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { position: 'top', labels: { boxWidth: 10, font: { size: 10 } } } },
            scales: { x: { display: false }, y: { grid: { color: 'rgba(255,255,255,0.05)' } } }
        }
    });
}

// Lógica de Pestañas (Tabs)
document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        // Desactivar todos los botones
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        // Activar el clickeado
        btn.classList.add('active');
        
        // Ocultar todos los paneles
        document.querySelectorAll('.tab-pane').forEach(p => p.classList.add('hidden'));
        // Mostrar el panel correspondiente
        const targetId = btn.getAttribute('data-target');
        document.getElementById(targetId).classList.remove('hidden');
    });
});

// Iniciar aplicación
init();
