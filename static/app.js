let predictionChart = null;
let rsiChart = null;
let backtestChart = null;
let featureChart = null;

// Video Canvas Animation Variables
let isVideoPlaying = true;
let videoAnimationFrame = null;

document.addEventListener('DOMContentLoaded', () => {
    fetchPredictions();
    initTheme();
    initVideoCanvas();
});

/* Explicit Global Window Expose for Button Click Handlers */
window.selectCompany = function(symbol, el) {
    document.querySelectorAll('.stock-chip').forEach(chip => chip.classList.remove('active'));
    if (el) el.classList.add('active');
    
    document.getElementById('ticker').value = symbol;
    fetchPredictions();
};

window.switchTab = function(tabId, el) {
    console.log("Switching tab to:", tabId);
    
    // Deactivate all nav tabs
    document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
    
    // Hide all tab contents
    document.querySelectorAll('.tab-content').forEach(c => {
        c.classList.remove('active');
    });
    
    // Activate clicked tab element
    if (el) {
        el.classList.add('active');
    }
    
    // Activate target tab content container
    const targetContent = document.getElementById(tabId);
    if (targetContent) {
        targetContent.classList.add('active');
    }
    
    // Trigger Chart.js resize/update for hidden charts when tab becomes active
    setTimeout(() => {
        if (tabId === 'tab-backtest' && backtestChart) {
            backtestChart.resize();
            backtestChart.update();
        } else if (tabId === 'tab-forecast') {
            if (predictionChart) { predictionChart.resize(); predictionChart.update(); }
            if (rsiChart) { rsiChart.resize(); rsiChart.update(); }
            if (featureChart) { featureChart.resize(); featureChart.update(); }
        }
    }, 50);
};

window.toggleTheme = function() {
    const currentTheme = document.body.classList.contains('light-theme') ? 'light' : 'dark';
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    setTheme(newTheme);
};

window.exportDataCSV = async function() {
    const rawTicker = document.getElementById('ticker').value.trim() || 'GOOGL';
    const period = document.getElementById('period').value;
    
    const btn = document.querySelector('.btn-secondary');
    if (btn) btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Generating CSV Report...';
    
    try {
        const response = await fetch('/api/export_csv', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ticker: rawTicker, period: period })
        });
        
        if (!response.ok) throw new Error('Failed to generate CSV');
        
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.style.display = 'none';
        a.href = url;
        a.download = `${rawTicker}_Stock_ML_Forecast_Report.csv`;
        document.body.appendChild(a);
        a.click();
        
        setTimeout(() => {
            window.URL.revokeObjectURL(url);
            a.remove();
        }, 1000);
        
    } catch (e) {
        alert('CSV Export failed: ' + e.message);
    } finally {
        if (btn) btn.innerHTML = '<i class="fa-solid fa-file-csv text-accent"></i> Download CSV Report File';
    }
};

window.toggleVideoPlay = function() {
    isVideoPlaying = !isVideoPlaying;
    const icon = document.getElementById('v-play-icon');
    if (icon) icon.className = isVideoPlaying ? 'fa-solid fa-pause' : 'fa-solid fa-play';
};

window.handlePrediction = function(e) {
    if (e) e.preventDefault();
    fetchPredictions();
};

/* Theme Initialization */
function initTheme() {
    const savedTheme = localStorage.getItem('app-theme') || 'dark';
    setTheme(savedTheme);
}

function setTheme(theme) {
    if (theme === 'light') {
        document.body.classList.remove('dark-theme');
        document.body.classList.add('light-theme');
        const icon = document.getElementById('theme-icon');
        if (icon) icon.className = 'fa-solid fa-sun';
        const txt = document.getElementById('theme-text');
        if (txt) txt.innerText = 'Light Theme';
    } else {
        document.body.classList.remove('light-theme');
        document.body.classList.add('dark-theme');
        const icon = document.getElementById('theme-icon');
        if (icon) icon.className = 'fa-solid fa-moon';
        const txt = document.getElementById('theme-text');
        if (txt) txt.innerText = 'Dark Theme';
    }
    localStorage.setItem('app-theme', theme);
    
    if (predictionChart && window.lastChartData) {
        renderPredictionChart(window.lastChartData.charts, window.lastChartData.company.name);
    }
}

async function fetchPredictions() {
    const rawTicker = document.getElementById('ticker').value.trim() || 'GOOGL';
    const period = document.getElementById('period').value;
    const targetType = document.getElementById('target_type').value;

    const btn = document.getElementById('btn-submit');
    if (btn) {
        btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Running ML Pipeline...';
        btn.disabled = true;
    }

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ticker: rawTicker, period: period, target_type: targetType })
        });

        const data = await response.json();

        if (data.status === 'error') {
            alert('Prediction Error: ' + data.message);
            return;
        }

        window.lastChartData = data;

        // Update Top Stock Header Ticker Bar & Real Corporate Logo
        const comp = data.company;
        const logoImg = document.getElementById('comp-logo-img');
        if (logoImg && comp.logo_url) {
            logoImg.src = comp.logo_url;
            logoImg.onerror = function() {
                this.src = `https://ui-avatars.com/api/?name=${comp.ticker}&background=6366f1&color=fff&bold=true`;
            };
        }
        
        document.getElementById('comp-name').innerText = comp.name;
        document.getElementById('comp-ticker').innerText = `${comp.exchange}: ${comp.ticker}`;
        document.getElementById('comp-sector').innerText = comp.sector;

        document.getElementById('quote-price').innerText = `$${comp.latest_price.toFixed(2)}`;
        
        const changeBadge = document.getElementById('quote-change');
        const sign = comp.price_change >= 0 ? '+' : '';
        changeBadge.innerText = `${sign}${comp.price_change_pct.toFixed(2)}% ($${comp.price_change.toFixed(2)})`;
        changeBadge.className = 'change-badge ' + (comp.price_change >= 0 ? 'positive' : 'negative');

        // Update Supervised Forecast Price
        const forecastVal = typeof data.next_day_forecast === 'number' ? `$${data.next_day_forecast.toFixed(2)}` : data.next_day_forecast;
        document.getElementById('quote-forecast').innerText = forecastVal;
        document.getElementById('quote-forecast-sub').innerText = `Model: ${data.best_model}`;
        document.getElementById('active-ticker-pill').innerText = `${comp.exchange}: ${comp.ticker}`;

        // Update Multi-Day Future Targets
        if (data.future_forecasts) {
            const ff = data.future_forecasts;
            const lp = comp.latest_price;
            
            document.getElementById('future-1d').innerText = `$${ff.day_1.toFixed(2)}`;
            const d1_pct = (((ff.day_1 - lp) / lp) * 100).toFixed(2);
            document.getElementById('future-1d-badge').innerText = (d1_pct >= 0 ? '+' : '') + d1_pct + '%';
            
            document.getElementById('future-3d').innerText = `$${ff.day_3.toFixed(2)}`;
            const d3_pct = (((ff.day_3 - lp) / lp) * 100).toFixed(2);
            document.getElementById('future-3d-badge').innerText = (d3_pct >= 0 ? '+' : '') + d3_pct + '%';

            document.getElementById('future-5d').innerText = `$${ff.day_5.toFixed(2)}`;
            const d5_pct = (((ff.day_5 - lp) / lp) * 100).toFixed(2);
            document.getElementById('future-5d-badge').innerText = (d5_pct >= 0 ? '+' : '') + d5_pct + '%';

            document.getElementById('future-7d').innerText = `$${ff.day_7.toFixed(2)}`;
            const d7_pct = (((ff.day_7 - lp) / lp) * 100).toFixed(2);
            document.getElementById('future-7d-badge').innerText = (d7_pct >= 0 ? '+' : '') + d7_pct + '%';
        }

        // Update Day-by-Day Current Month Forecast Table
        if (data.monthly_daily_forecast && data.current_month_summary) {
            renderMonthlyDailyForecastTable(data.monthly_daily_forecast, data.current_month_summary, comp.name);
        }

        // Update Stat Cards
        document.getElementById('stat-best-model').innerText = data.best_model;
        
        const stratReturn = data.backtest_metrics.total_strategy_return;
        const benchReturn = data.backtest_metrics.total_benchmark_return;
        
        document.getElementById('stat-strategy-return').innerText = (stratReturn >= 0 ? '+' : '') + stratReturn + ' %';
        document.getElementById('stat-strategy-return').className = 'stat-value ' + (stratReturn >= 0 ? 'text-success' : 'text-danger');
        document.getElementById('stat-benchmark-return').innerText = 'Buy & Hold: ' + benchReturn + ' %';

        document.getElementById('stat-sharpe').innerText = data.backtest_metrics.sharpe_ratio;
        document.getElementById('stat-drawdown').innerText = data.backtest_metrics.max_drawdown + ' %';
        document.getElementById('stat-win-rate').innerText = 'Win Rate: ' + data.backtest_metrics.win_rate + ' %';

        // Render Charts
        renderPredictionChart(data.charts, comp.name);
        renderRSIChart(data.charts);
        renderBacktestChart(data.charts);
        renderFeatureChart(data.feature_importance);

        // Render Models Table
        renderModelsTable(data.models_summary, data.target_type);

    } catch (err) {
        console.error('Fetch error:', err);
    } finally {
        if (btn) {
            btn.innerHTML = '<i class="fa-solid fa-bolt"></i> Run Supervised ML Pipeline';
            btn.disabled = false;
        }
    }
}

function renderMonthlyDailyForecastTable(dailyList, summary, companyName) {
    const desc = document.getElementById('month-desc');
    if (desc) desc.innerText = `Predicts daily stock price, dollar increase/decrease amount ($), and directional signal for ${companyName} during ${summary.month_name}.`;
    
    const endP = document.getElementById('month-end-price');
    if (endP) endP.innerText = `$${summary.end_price.toFixed(2)}`;
    
    const nameSub = document.getElementById('month-name-sub');
    if (nameSub) nameSub.innerText = summary.month_name;
    
    const sign = summary.total_change >= 0 ? '+' : '';
    const totChange = document.getElementById('month-total-change');
    if (totChange) {
        totChange.innerText = `${sign}$${summary.total_change.toFixed(2)} (${sign}${summary.total_change_pct.toFixed(2)}%)`;
        totChange.className = 'month-stat-val ' + (summary.total_change >= 0 ? 'text-success' : 'text-danger');
    }
    
    const upDown = document.getElementById('month-up-down');
    if (upDown) upDown.innerText = `${summary.up_days} Days UP / ${summary.down_days} Days DOWN`;

    const tbody = document.querySelector('#daily-month-table tbody');
    if (tbody) {
        tbody.innerHTML = dailyList.map(item => {
            const isUp = item.change_amount >= 0;
            const changeSign = isUp ? '+' : '';
            const badgeClass = isUp ? 'dir-badge up' : 'dir-badge down';
            const textClass = isUp ? 'text-success' : 'text-danger';
            
            return `
                <tr ${item.is_past ? 'style="opacity: 0.85;"' : ''}>
                    <td><strong>${item.date}</strong> ${item.is_past ? '<span style="font-size: 0.7rem; color: #94a3b8;">(Past)</span>' : ''}</td>
                    <td>${item.day}</td>
                    <td><strong>$${item.predicted_price.toFixed(2)}</strong></td>
                    <td><span class="${textClass}"><strong>${changeSign}$${item.change_amount.toFixed(2)}</strong></span></td>
                    <td><span class="${textClass}">${changeSign}${item.change_pct.toFixed(2)}%</span></td>
                    <td><span class="${badgeClass}">${isUp ? '▲ BULLISH (UP)' : '▼ BEARISH (DOWN)'}</span></td>
                </tr>
            `;
        }).join('');
    }
}

function renderPredictionChart(charts, companyName) {
    const ctx = document.getElementById('predictionChart');
    if (!ctx) return;
    if (predictionChart) predictionChart.destroy();

    const isDark = !document.body.classList.contains('light-theme');
    const textColor = isDark ? '#94a3b8' : '#475569';
    const gridColor = isDark ? 'rgba(255, 255, 255, 0.05)' : 'rgba(0, 0, 0, 0.05)';

    predictionChart = new Chart(ctx.getContext('2d'), {
        type: 'line',
        data: {
            labels: charts.test_dates,
            datasets: [
                {
                    label: `${companyName} Actual Price ($)`,
                    data: charts.actual_test_prices,
                    borderColor: '#06b6d4',
                    backgroundColor: 'rgba(6, 182, 212, 0.08)',
                    borderWidth: 2.5,
                    fill: true,
                    pointRadius: 0,
                    tension: 0.1
                },
                {
                    label: 'Supervised Model Predicted Price ($)',
                    data: charts.pred_test_prices,
                    borderColor: '#10b981',
                    borderWidth: 2,
                    borderDash: [4, 4],
                    pointRadius: 0,
                    tension: 0.1
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                legend: { labels: { color: textColor, font: { family: 'Outfit', size: 12 } } }
            },
            scales: {
                x: { ticks: { color: textColor, maxTicksLimit: 12 }, grid: { color: gridColor } },
                y: { ticks: { color: textColor }, grid: { color: gridColor } }
            }
        }
    });
}

function renderRSIChart(charts) {
    const ctx = document.getElementById('rsiChart');
    if (!ctx) return;
    if (rsiChart) rsiChart.destroy();

    const isDark = !document.body.classList.contains('light-theme');
    const textColor = isDark ? '#94a3b8' : '#475569';
    const gridColor = isDark ? 'rgba(255, 255, 255, 0.05)' : 'rgba(0, 0, 0, 0.05)';

    const rsiSubDates = charts.hist_dates.slice(-charts.test_dates.length);
    const rsiSubData = charts.rsi14.slice(-charts.test_dates.length);

    rsiChart = new Chart(ctx.getContext('2d'), {
        type: 'line',
        data: {
            labels: rsiSubDates,
            datasets: [{
                label: 'RSI (14)',
                data: rsiSubData,
                borderColor: '#6366f1',
                borderWidth: 1.8,
                pointRadius: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { ticks: { color: textColor, maxTicksLimit: 6 }, grid: { display: false } },
                y: { min: 0, max: 100, ticks: { color: textColor }, grid: { color: gridColor } }
            }
        }
    });
}

function renderBacktestChart(charts) {
    const ctx = document.getElementById('backtestChart');
    if (!ctx) return;
    if (backtestChart) backtestChart.destroy();

    const isDark = !document.body.classList.contains('light-theme');
    const textColor = isDark ? '#94a3b8' : '#475569';
    const gridColor = isDark ? 'rgba(255, 255, 255, 0.05)' : 'rgba(0, 0, 0, 0.05)';

    backtestChart = new Chart(ctx.getContext('2d'), {
        type: 'line',
        data: {
            labels: charts.backtest_dates,
            datasets: [
                {
                    label: 'ML Strategy Portfolio Growth ($1.00 Start)',
                    data: charts.strategy_equity,
                    borderColor: '#10b981',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    borderWidth: 2.5,
                    fill: true,
                    pointRadius: 0
                },
                {
                    label: 'Buy & Hold Benchmark ($1.00 Start)',
                    data: charts.buy_hold_equity,
                    borderColor: '#64748b',
                    borderWidth: 1.5,
                    borderDash: [3, 3],
                    pointRadius: 0
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { labels: { color: textColor, font: { family: 'Outfit', size: 12 } } }
            },
            scales: {
                x: { ticks: { color: textColor, maxTicksLimit: 8 }, grid: { color: gridColor } },
                y: { ticks: { color: textColor }, grid: { color: gridColor } }
            }
        }
    });
}

function renderFeatureChart(features) {
    const ctx = document.getElementById('featureChart');
    if (!ctx) return;
    if (featureChart) featureChart.destroy();

    const isDark = !document.body.classList.contains('light-theme');
    const textColor = isDark ? '#94a3b8' : '#475569';

    const labels = features.map(f => f.Feature);
    const values = features.map(f => f.Importance);

    featureChart = new Chart(ctx.getContext('2d'), {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Importance Score',
                data: values,
                backgroundColor: 'rgba(139, 92, 246, 0.75)',
                borderColor: '#8b5cf6',
                borderWidth: 1,
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            indexAxis: 'y',
            plugins: { legend: { display: false } },
            scales: {
                x: { ticks: { color: textColor }, grid: { color: 'rgba(255, 255, 255, 0.05)' } },
                y: { ticks: { color: textColor, font: { size: 10 } }, grid: { display: false } }
            }
        }
    });
}

function renderModelsTable(summary, targetType) {
    const table = document.getElementById('models-table');
    if (!table) return;
    const thead = table.querySelector('thead');
    const tbody = table.querySelector('tbody');

    if (targetType === 'regression') {
        thead.innerHTML = `
            <tr>
                <th>Supervised Algorithm</th>
                <th>Mean Absolute Error (MAE)</th>
                <th>Root Mean Sq Error (RMSE)</th>
                <th>R² Score</th>
                <th>Validation Status</th>
            </tr>
        `;
        tbody.innerHTML = summary.map(m => `
            <tr>
                <td><strong>${m.Model}</strong></td>
                <td>$${m.MAE}</td>
                <td>$${m.RMSE}</td>
                <td><span class="${m.R2_Score > 0 ? 'text-success' : 'text-danger'}">${m.R2_Score}</span></td>
                <td><span class="pill">${m.R2_Score > 0.5 ? 'Strong Fit' : 'Evaluated'}</span></td>
            </tr>
        `).join('');
    } else {
        thead.innerHTML = `
            <tr>
                <th>Supervised Algorithm</th>
                <th>Accuracy</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1-Score</th>
            </tr>
        `;
        tbody.innerHTML = summary.map(m => `
            <tr>
                <td><strong>${m.Model}</strong></td>
                <td><span class="text-success">${(m.Accuracy * 100).toFixed(1)}%</span></td>
                <td>${(m.Precision * 100).toFixed(1)}%</td>
                <td>${(m.Recall * 100).toFixed(1)}%</td>
                <td><strong>${m.F1_Score}</strong></td>
            </tr>
        `).join('');
    }
}

/* Custom Interactive Canvas Video Explainer Player */
function initVideoCanvas() {
    const canvas = document.getElementById('videoCanvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    
    let frame = 0;
    const steps = [
        "Step 1: Historical Data Ingestion (Open, High, Low, Close, Volume)",
        "Step 2: Technical Feature Engineering (SMA 20/50/200, RSI 14, MACD)",
        "Step 3: Supervised Model Training (Feature Matrix X -> Next Target Y)",
        "Step 4: Out-of-Sample Walk-Forward Validation & Strategy Backtest"
    ];

    function drawVideo() {
        if (isVideoPlaying) {
            frame++;
        }
        
        ctx.fillStyle = '#080b11';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        const currentStep = Math.floor((frame / 120) % 4);
        const statusEl = document.getElementById('v-status-text');
        if (statusEl) statusEl.innerText = steps[currentStep];

        ctx.strokeStyle = 'rgba(99, 102, 241, 0.1)';
        ctx.lineWidth = 1;
        for (let x = 0; x < canvas.width; x += 40) {
            ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, canvas.height); ctx.stroke();
        }
        for (let y = 0; y < canvas.height; y += 40) {
            ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(canvas.width, y); ctx.stroke();
        }

        ctx.beginPath();
        ctx.strokeStyle = '#06b6d4';
        ctx.lineWidth = 3;
        for (let i = 0; i < canvas.width; i += 5) {
            const y = 220 + Math.sin((i + frame * 2) * 0.02) * 60 + Math.cos(i * 0.05) * 20;
            if (i === 0) ctx.moveTo(i, y);
            else ctx.lineTo(i, y);
        }
        ctx.stroke();

        ctx.beginPath();
        ctx.strokeStyle = '#10b981';
        ctx.lineWidth = 2.5;
        ctx.setLineDash([5, 5]);
        for (let i = 0; i < canvas.width; i += 5) {
            const y = 220 + Math.sin((i + frame * 2 + 10) * 0.02) * 58 + Math.cos(i * 0.05) * 18;
            if (i === 0) ctx.moveTo(i, y);
            else ctx.lineTo(i, y);
        }
        ctx.stroke();
        ctx.setLineDash([]);

        ctx.fillStyle = 'rgba(15, 23, 42, 0.85)';
        ctx.fillRect(30, 30, 600, 50);
        ctx.strokeStyle = '#6366f1';
        ctx.strokeRect(30, 30, 600, 50);

        ctx.font = 'bold 18px Outfit';
        ctx.fillStyle = '#6366f1';
        ctx.fillText(steps[currentStep], 45, 62);

        videoAnimationFrame = requestAnimationFrame(drawVideo);
    }
    
    drawVideo();
}
