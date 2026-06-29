const API = 'http://localhost:8000';

function setupAutocomplete(inputId, dropdownId, onSelect) {
  const input = document.getElementById(inputId);
  const dropdown = document.getElementById(dropdownId);
  let timer = null;

  input.addEventListener('input', function() {
    const val = this.value.trim();
    clearTimeout(timer);
    if (val.length < 2) { dropdown.style.display = 'none'; return; }
    timer = setTimeout(async () => {
      try {
        const res = await fetch(`${API}/autocomplete?query=${encodeURIComponent(val)}`);
        const data = await res.json();
        if (!data.results || data.results.length === 0) { dropdown.style.display = 'none'; return; }
        dropdown.innerHTML = '';
        data.results.forEach(r => {
          const item = document.createElement('div');
          item.className = 'dropdown-item';
          const meta = [r.admin1, r.country].filter(Boolean).join(', ');
          item.innerHTML = `<span class="dropdown-city">${r.name}</span><span class="dropdown-meta">${meta}</span>`;
          item.addEventListener('mousedown', function(e) {
            e.preventDefault();
            input.value = r.name;
            dropdown.style.display = 'none';
            if (onSelect) onSelect();
          });
          dropdown.appendChild(item);
        });
        dropdown.style.display = 'block';
      } catch(e) { dropdown.style.display = 'none'; }
    }, 150);
  });

  input.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') dropdown.style.display = 'none';
  });

  document.addEventListener('click', function(e) {
    if (!e.target.closest(`#${inputId}`) && !e.target.closest(`#${dropdownId}`)) {
      dropdown.style.display = 'none';
    }
  });
}

setupAutocomplete('cityInput', 'cityDropdown', () => doSearch());
setupAutocomplete('cityAInput', 'cityADropdown');
setupAutocomplete('cityBInput', 'cityBDropdown');

let trendChartInstance = null;
let forecastChartInstance = null;
let featureChartInstance = null;
let compareChartInstance = null;

Chart.defaults.color = '#666666';
Chart.defaults.borderColor = 'rgba(0,0,0,0.06)';

document.getElementById('searchBtn').addEventListener('click', doSearch);
document.getElementById('cityInput').addEventListener('keydown', e => {
  if (e.key === 'Enter') doSearch();
});

async function doSearch() {
  const city = document.getElementById('cityInput').value.trim();
  if (!city) return;
  const btn = document.getElementById('searchBtn');
  btn.textContent = 'Loading...';
  btn.disabled = true;
  try {
    const res = await fetch(`${API}/weather?city=${encodeURIComponent(city)}`);
    const data = await res.json();
    if (data.error) { alert(data.error); return; }
    renderWeather(data);
  } catch(e) {
    alert('Failed to fetch weather. Is the API running?');
  } finally {
    btn.textContent = 'Search';
    btn.disabled = false;
  }
}

function renderWeather(data) {
  const t = data.today;
  document.getElementById('cityTitle').textContent = data.city;
  document.getElementById('cityCoords').textContent = `${data.lat.toFixed(2)}, ${data.lon.toFixed(2)}`;
  document.getElementById('tempMax').textContent = t.temp_max;
  document.getElementById('tempMin').textContent = t.temp_min;
  document.getElementById('precip').textContent = t.precipitation;
  document.getElementById('wind').textContent = t.windspeed;
  document.getElementById('sunrise').textContent = t.sunrise.slice(11,16);
  document.getElementById('sunset').textContent = t.sunset.slice(11,16);
  const labels = data.trend.map(d => d.date.slice(5));
  const maxTemps = data.trend.map(d => d.temp_max);
  const minTemps = data.trend.map(d => d.temp_min);
  if (trendChartInstance) trendChartInstance.destroy();
  trendChartInstance = new Chart(document.getElementById('trendChart'), {
    type: 'line',
    data: {
      labels,
      datasets: [
        { label: 'Max Temp (°C)', data: maxTemps, borderColor: '#38bdf8', backgroundColor: 'rgba(56,189,248,0.08)', fill: true, tension: 0.4, pointBackgroundColor: '#38bdf8', pointRadius: 4 },
        { label: 'Min Temp (°C)', data: minTemps, borderColor: '#94a3b8', backgroundColor: 'rgba(148,163,184,0.06)', fill: true, tension: 0.4, pointBackgroundColor: '#94a3b8', pointRadius: 4 }
      ]
    },
    options: { responsive: true, plugins: { legend: { position: 'top' } }, scales: { x: { grid: { color: 'rgba(0,0,0,0.06)' } }, y: { grid: { color: 'rgba(0,0,0,0.06)' } } } }
  });
  document.getElementById('weatherSection').style.display = 'block';
  document.getElementById('predictSection').style.display = 'none';
}

document.getElementById('predictBtn').addEventListener('click', doPredict);

async function doPredict() {
  const city = document.getElementById('cityInput').value.trim();
  if (!city) return;
  const btn = document.getElementById('predictBtn');
  btn.textContent = 'Running ML Model...';
  btn.disabled = true;
  try {
    const res = await fetch(`${API}/predict?city=${encodeURIComponent(city)}`);
    const data = await res.json();
    if (data.error) { alert(data.error); return; }
    renderPredictions(data);
  } catch(e) { alert('Prediction failed.'); }
  finally { btn.textContent = 'Run ML Forecast'; btn.disabled = false; }
}

const conditionIcons = { 'Rainy': '🌧️', 'Hot': '🔥', 'Sunny': '☀️', 'Partly Cloudy': '⛅', 'Cloudy': '🌥️' };

function renderPredictions(data) {
  const badge = document.getElementById('rainBadge');
  if (data.rain_tomorrow === 1) {
    badge.textContent = `🌧️ Rain expected tomorrow — ${data.rain_probability}% confidence`;
    badge.style.borderColor = '#38bdf8';
    badge.style.color = '#0369a1';
  } else {
    badge.textContent = `☀️ No rain expected tomorrow — ${data.rain_probability}% rain probability`;
    badge.style.borderColor = 'rgba(0,0,0,0.12)';
    badge.style.color = '#666666';
  }
  const tbody = document.getElementById('forecastTableBody');
  tbody.innerHTML = '';
  data.predictions.forEach(p => {
    const icon = conditionIcons[p.condition] || '';
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${p.date}</td><td style="color:#0a0a0a;font-weight:600">${p.temp_max}°C</td><td>${icon} ${p.condition}</td><td style="color:#888">± ${p.confidence_range}°C</td>`;
    tbody.appendChild(tr);
  });
  const labels = data.predictions.map(p => p.date.slice(5));
  const temps = data.predictions.map(p => p.temp_max);
  if (forecastChartInstance) forecastChartInstance.destroy();
  forecastChartInstance = new Chart(document.getElementById('forecastChart'), {
    type: 'line',
    data: { labels, datasets: [{ label: 'Predicted Max Temp (°C)', data: temps, borderColor: '#38bdf8', backgroundColor: 'rgba(56,189,248,0.08)', fill: true, tension: 0.4, pointBackgroundColor: '#38bdf8', pointRadius: 5 }] },
    options: { responsive: true, plugins: { legend: { position: 'top' } }, scales: { x: { grid: { color: 'rgba(0,0,0,0.06)' } }, y: { grid: { color: 'rgba(0,0,0,0.06)' } } } }
  });
  const fi = data.feature_importance;
  const fiLabels = Object.keys(fi);
  const fiValues = Object.values(fi);
  if (featureChartInstance) featureChartInstance.destroy();
  featureChartInstance = new Chart(document.getElementById('featureChart'), {
    type: 'bar',
    data: { labels: fiLabels, datasets: [{ label: 'Importance', data: fiValues, backgroundColor: fiValues.map((v,i) => i === fiValues.indexOf(Math.max(...fiValues)) ? 'rgba(56,189,248,0.8)' : 'rgba(56,189,248,0.15)'), borderColor: fiValues.map((v,i) => i === fiValues.indexOf(Math.max(...fiValues)) ? '#38bdf8' : 'rgba(56,189,248,0.3)'), borderWidth: 1, borderRadius: 6 }] },
    options: { responsive: true, plugins: { legend: { display: false } }, scales: { x: { grid: { color: 'rgba(0,0,0,0.06)' } }, y: { grid: { color: 'rgba(0,0,0,0.06)' } } } }
  });
  const grid = document.getElementById('modelStatsGrid');
  grid.innerHTML = '';
  Object.entries(data.model_comparison).forEach(([name, scores]) => {
    const div = document.createElement('div');
    div.className = 'stat-card';
    div.innerHTML = `<div class="stat-name">${name}</div><div class="stat-values">MAE <span>${scores.MAE}</span> · RMSE <span>${scores.RMSE}</span></div>`;
    grid.appendChild(div);
  });
  document.getElementById('predictSection').style.display = 'block';
  document.getElementById('predictSection').scrollIntoView({ behavior: 'smooth' });
}

document.getElementById('compareBtn').addEventListener('click', doCompare);

async function doCompare() {
  const a = document.getElementById('cityAInput').value.trim();
  const b = document.getElementById('cityBInput').value.trim();
  if (!a || !b) return;
  const btn = document.getElementById('compareBtn');
  btn.textContent = 'Loading...';
  btn.disabled = true;
  try {
    const res = await fetch(`${API}/compare?city_a=${encodeURIComponent(a)}&city_b=${encodeURIComponent(b)}`);
    const data = await res.json();
    if (!data.city_a || !data.city_b) { alert('Could not resolve one or both cities.'); return; }
    renderCompare(data);
  } catch(e) { alert('Compare failed.'); }
  finally { btn.textContent = 'Compare'; btn.disabled = false; }
}

function renderCompare(data) {
  function buildCard(el, city) {
    el.innerHTML = `<h3>${city.city}</h3><div class="compare-row"><span class="label">Max Temp</span><span class="value">${city.temp_max}°C</span></div><div class="compare-row"><span class="label">Min Temp</span><span class="value">${city.temp_min}°C</span></div><div class="compare-row"><span class="label">Precipitation</span><span class="value">${city.precipitation} mm</span></div><div class="compare-row"><span class="label">Max Wind</span><span class="value">${city.windspeed} km/h</span></div>`;
  }
  buildCard(document.getElementById('cityACard'), data.city_a);
  buildCard(document.getElementById('cityBCard'), data.city_b);
  const labels = data.city_a.trend.map(d => d.date.slice(5));
  if (compareChartInstance) compareChartInstance.destroy();
  compareChartInstance = new Chart(document.getElementById('compareChart'), {
    type: 'line',
    data: { labels, datasets: [
      { label: data.city_a.city, data: data.city_a.trend.map(d => d.temp_max), borderColor: '#38bdf8', backgroundColor: 'rgba(56,189,248,0.08)', fill: true, tension: 0.4, pointRadius: 4 },
      { label: data.city_b.city, data: data.city_b.trend.map(d => d.temp_max), borderColor: '#f97316', backgroundColor: 'rgba(249,115,22,0.06)', fill: true, tension: 0.4, pointRadius: 4 }
    ]},
    options: { responsive: true, plugins: { legend: { position: 'top' } }, scales: { x: { grid: { color: 'rgba(0,0,0,0.06)' } }, y: { grid: { color: 'rgba(0,0,0,0.06)' } } } }
  });
  document.getElementById('compareResults').style.display = 'block';
  document.getElementById('compareResults').scrollIntoView({ behavior: 'smooth' });
}
