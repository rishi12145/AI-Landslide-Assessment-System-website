/**
 * dashboard.js
 * Renders the Analysis Dashboard from Analysis Engine JSON.
 * The JSON output is the ONLY source of truth.
 * Never fabricates metrics. Never displays placeholder values.
 *
 * Sections:
 *   A — Metadata
 *   B — Coherence Analysis
 *   C — Phase Analysis
 *   D — Segmentation Analysis
 *   E — Shape Analysis
 *   F — Confidence Analysis
 *   G — Severity Assessment
 */

// ─────────────────────────────────────────────────────────
// Formatting helpers
// ─────────────────────────────────────────────────────────

function fmt(value, decimals = 4) {
    if (value === null || value === undefined) return '<span class="null-val">N/A</span>';
    if (typeof value === "number" && Number.isFinite(value)) {
        // Round large integers without decimals
        if (Number.isInteger(value) && Math.abs(value) >= 1000) return value.toLocaleString();
        // Format floats
        const fixed = value.toFixed(decimals);
        return fixed.replace(/(\.\d*?)0+$/, '$1').replace(/\.$/, '');
    }
    if (Array.isArray(value)) return value.map(v => fmt(v, decimals)).join(', ');
    return String(value);
}

function fmtPct(value, decimals = 2) {
    if (value === null || value === undefined) return '<span class="null-val">N/A</span>';
    return fmt(value, decimals) + '%';
}

function fmtPx(value) {
    if (value === null || value === undefined) return '<span class="null-val">N/A</span>';
    return fmt(value, 0) + ' px';
}

// ─────────────────────────────────────────────────────────
// Component builders
// ─────────────────────────────────────────────────────────

function sectionHeader(letter, title) {
    return `
        <div class="dash-section-header">
            <div class="dash-section-letter">${letter}</div>
            <h3>${title}</h3>
        </div>`;
}

function statCard(label, value, unit = '') {
    const valHtml = (value === null || value === undefined)
        ? '<span class="null-val">N/A</span>'
        : `${fmt(value)}${unit ? `<span class="stat-unit">${unit}</span>` : ''}`;
    return `
        <div class="stat-card">
            <span class="stat-label">${label}</span>
            <span class="stat-value">${valHtml}</span>
        </div>`;
}

function progressBar(label, pct, colorClass = 'bar-blue') {
    const safe = Math.min(100, Math.max(0, parseFloat(pct) || 0));
    return `
        <div class="progress-row">
            <div class="progress-header">
                <span>${label}</span>
                <span>${fmt(safe, 2)}%</span>
            </div>
            <div class="progress-track">
                <div class="progress-fill ${colorClass}" style="width: ${safe}%"></div>
            </div>
        </div>`;
}

function metaRow(label, value, badge = false) {
    const display = (value === null || value === undefined || value === '')
        ? '<span class="null-val">—</span>'
        : (badge ? `<span class="meta-badge" style="background:rgba(59,130,246,0.12);color:#93c5fd;border:1px solid rgba(59,130,246,0.25)">${value}</span>` : value);
    return `
        <tr>
            <th>${label}</th>
            <td>${display}</td>
        </tr>`;
}

// ─────────────────────────────────────────────────────────
// Section A — Metadata
// ─────────────────────────────────────────────────────────
function renderMetadata(m) {
    return `
        <div class="dash-section">
            ${sectionHeader('A', 'Metadata')}
            <div class="dash-section-body">
                <table class="meta-table">
                    ${metaRow('Sample ID', m.sample_id)}
                    ${metaRow('Image ID', m.image_id)}
                    ${metaRow('Image Name', m.image_name)}
                    ${metaRow('Region', m.region)}
                    ${metaRow('Temporal Baseline', m.temporal_baseline)}
                    ${metaRow('Dataset Split', m.dataset_split, true)}
                    ${metaRow('Patch Number', m.patch_number)}
                    ${metaRow('Analysis Date', m.analysis_date)}
                </table>
            </div>
        </div>`;
}

// ─────────────────────────────────────────────────────────
// Section B — Coherence Analysis
// ─────────────────────────────────────────────────────────
function renderCoherence(coh) {
    const low  = parseFloat(coh.low_coherence_percentage)    || 0;
    const med  = parseFloat(coh.medium_coherence_percentage) || 0;
    const high = parseFloat(coh.high_coherence_percentage)   || 0;

    // Normalize distribution chart heights
    const maxVal = Math.max(low, med, high, 1);

    return `
        <div class="dash-section">
            ${sectionHeader('B', 'Coherence Analysis')}
            <div class="dash-section-body">
                <div class="section-split">
                    <div>
                        <div class="stat-grid cols-3">
                            ${statCard('Mean', coh.mean)}
                            ${statCard('Median', coh.median)}
                            ${statCard('Std Dev', coh.std)}
                            ${statCard('Minimum', coh.minimum)}
                            ${statCard('Maximum', coh.maximum)}
                            ${statCard('Range', coh.range)}
                            ${statCard('Q25', coh.q25)}
                            ${statCard('Q50 (Median)', coh.q50)}
                            ${statCard('Q75', coh.q75)}
                            ${statCard('Skewness', coh.skewness)}
                            ${statCard('Kurtosis', coh.kurtosis)}
                            ${statCard('Variance', coh.variance)}
                        </div>
                    </div>
                    <div>
                        <p class="viz-title">Coherence Distribution</p>
                        ${progressBar('Low Coherence (&lt;0.30)', low, 'bar-red')}
                        ${progressBar('Medium Coherence (0.30–0.70)', med, 'bar-amber')}
                        ${progressBar('High Coherence (&gt;0.70)', high, 'bar-green')}
                        <div class="dist-chart" style="margin-top:12px">
                            <div class="dist-bar-group">
                                <div class="dist-bar bar-red" style="height:${(low/maxVal)*100}%"></div>
                                <span class="dist-label">Low</span>
                                <span class="dist-value">${fmt(low, 2)}%</span>
                            </div>
                            <div class="dist-bar-group">
                                <div class="dist-bar bar-amber" style="height:${(med/maxVal)*100}%"></div>
                                <span class="dist-label">Med</span>
                                <span class="dist-value">${fmt(med, 2)}%</span>
                            </div>
                            <div class="dist-bar-group">
                                <div class="dist-bar bar-green" style="height:${(high/maxVal)*100}%"></div>
                                <span class="dist-label">High</span>
                                <span class="dist-value">${fmt(high, 2)}%</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>`;
}

// ─────────────────────────────────────────────────────────
// Section C — Phase Analysis
// ─────────────────────────────────────────────────────────
function renderPhase(phase) {
    return `
        <div class="dash-section">
            ${sectionHeader('C', 'Phase Analysis')}
            <div class="dash-section-body">
                <div class="stat-grid cols-4">
                    ${statCard('Mean', phase.mean)}
                    ${statCard('Median', phase.median)}
                    ${statCard('Minimum', phase.minimum)}
                    ${statCard('Maximum', phase.maximum)}
                    ${statCard('Std Dev', phase.std)}
                    ${statCard('Variance', phase.variance)}
                    ${statCard('Range', phase.range)}
                    ${statCard('Skewness', phase.skewness)}
                    ${statCard('Kurtosis', phase.kurtosis)}
                    ${statCard('Entropy', phase.entropy)}
                    ${statCard('Energy', phase.energy)}
                    ${statCard('Gradient Mean', phase.gradient_mean)}
                    ${statCard('Gradient Std', phase.gradient_std)}
                </div>
            </div>
        </div>`;
}

// ─────────────────────────────────────────────────────────
// Section D — Segmentation Analysis
// ─────────────────────────────────────────────────────────
function renderSegmentation(seg) {
    const hasGT = seg.ground_truth_area !== null && seg.ground_truth_area !== undefined;

    const classMetrics = [
        ['Dice Score',    seg.dice],
        ['IoU (Jaccard)', seg.iou],
        ['Precision',     seg.precision],
        ['Recall',        seg.recall],
        ['Sensitivity',   seg.sensitivity],
        ['Specificity',   seg.specificity],
        ['Accuracy',      seg.accuracy],
        ['F1 Score',      seg.f1_score],
    ];

    const classHtml = `<div class="confusion-grid">${classMetrics.map(([l, v]) => `
            <div class="confusion-cell">
                <span class="confusion-label">${l}</span>
                <span class="confusion-value">${fmt(v)}</span>
            </div>`).join('')}
        </div>`;

    return `
        <div class="dash-section">
            ${sectionHeader('D', 'Segmentation Analysis')}
            <div class="dash-section-body">
                <div class="section-split">
                    <div>
                        <div class="stat-grid cols-2">
                            ${statCard('Ground Truth Area', seg.ground_truth_area, ' px')}
                            ${statCard('Predicted Area', seg.predicted_area, ' px')}
                            ${statCard('Area Difference', seg.difference, ' px')}
                            ${statCard('Area Percentage', null === seg.area_percentage ? null : fmt(seg.area_percentage, 2), '%')}
                            ${statCard('Avg Probability', seg.average_probability)}
                            ${statCard('Max Probability', seg.maximum_probability)}
                            ${statCard('Min Probability', seg.minimum_probability)}
                        </div>
                    </div>
                    <div>
                        <p class="viz-title">Classification Metrics</p>
                        ${classHtml}
                    </div>
                </div>
            </div>
        </div>`;
}

// ─────────────────────────────────────────────────────────
// Section E — Shape Analysis
// ─────────────────────────────────────────────────────────
function renderShape(shape) {
    const bb = Array.isArray(shape.bounding_box)
        ? `[${shape.bounding_box.map(v => Math.round(v)).join(', ')}]`
        : null;
    const centroid = Array.isArray(shape.centroid)
        ? `(${shape.centroid.map(v => parseFloat(v).toFixed(1)).join(', ')})`
        : null;

    return `
        <div class="dash-section">
            ${sectionHeader('E', 'Shape Analysis')}
            <div class="dash-section-body">
                <div class="stat-grid cols-4">
                    ${statCard('Connected Components', shape.connected_components)}
                    ${statCard('Largest Component', shape.largest_component, ' px')}
                    ${statCard('Smallest Component', shape.smallest_component, ' px')}
                    ${statCard('Avg Component Area', shape.average_component_area, ' px')}
                    ${statCard('Perimeter', shape.perimeter)}
                    ${statCard('Convex Area', shape.convex_area)}
                    ${statCard('Solidity', shape.solidity)}
                    ${statCard('Aspect Ratio', shape.aspect_ratio)}
                    ${statCard('Circularity', shape.circularity)}
                    ${statCard('Shape Complexity', shape.shape_complexity)}
                </div>
                <div class="stat-grid cols-2" style="margin-top:8px">
                    <div class="stat-card">
                        <span class="stat-label">Bounding Box [x, y, w, h]</span>
                        <span class="stat-value" style="font-size:12px">${bb !== null ? bb : '<span class="null-val">N/A</span>'}</span>
                    </div>
                    <div class="stat-card">
                        <span class="stat-label">Centroid [x, y]</span>
                        <span class="stat-value" style="font-size:12px">${centroid !== null ? centroid : '<span class="null-val">N/A</span>'}</span>
                    </div>
                </div>
            </div>
        </div>`;
}

// ─────────────────────────────────────────────────────────
// Section F — Confidence Analysis
// ─────────────────────────────────────────────────────────
function renderConfidence(conf) {
    const avg = parseFloat(conf.average_probability) || 0;
    const avgPct = Math.min(100, avg * 100);

    return `
        <div class="dash-section">
            ${sectionHeader('F', 'Confidence Analysis')}
            <div class="dash-section-body">
                <div class="stat-grid cols-3">
                    ${statCard('Average Probability', conf.average_probability)}
                    ${statCard('Maximum Probability', conf.maximum_probability)}
                    ${statCard('Minimum Probability', conf.minimum_probability)}
                    ${statCard('Confidence Variance', conf.confidence_variance)}
                    ${statCard('Confidence Entropy', conf.confidence_entropy)}
                </div>
                <div style="margin-top:12px">
                    ${progressBar('Average Prediction Confidence', avgPct, 'bar-cyan')}
                </div>
            </div>
        </div>`;
}

// ─────────────────────────────────────────────────────────
// Section G — Severity Assessment
// ─────────────────────────────────────────────────────────
function renderSeverity(severity) {
    const index = parseFloat(severity.severity_index) || 0;
    const pct   = Math.min(100, Math.max(0, index * 100));
    const risk  = (severity.risk_level || 'Unknown').toLowerCase().replace(/\s+/g, '-');
    const conf  = severity.confidence_level || 'N/A';

    // SVG gauge circumference = 2 * pi * 40 = ~251.2
    const circumference = 251.2;
    const offset = circumference - (pct / 100) * circumference;

    // Gauge color based on risk
    const gaugeColors = {
        'very-low':  '#10b981',
        'low':       '#22d3ee',
        'moderate':  '#f59e0b',
        'high':      '#f97316',
        'very-high': '#ef4444'
    };
    const gaugeColor = gaugeColors[risk] || '#3b82f6';

    return `
        <div class="dash-section">
            ${sectionHeader('G', 'Severity Assessment')}
            <div class="dash-section-body">
                <div class="severity-panel">
                    <div class="severity-gauge">
                        <div class="gauge-ring-wrap">
                            <svg class="gauge-ring-svg" viewBox="0 0 100 100">
                                <defs>
                                    <linearGradient id="gaugeGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                                        <stop offset="0%" stop-color="${gaugeColor}" stop-opacity="0.6"/>
                                        <stop offset="100%" stop-color="${gaugeColor}"/>
                                    </linearGradient>
                                </defs>
                                <circle class="gauge-track" cx="50" cy="50" r="40"/>
                                <circle class="gauge-fill" cx="50" cy="50" r="40"
                                    stroke="${gaugeColor}"
                                    stroke-dasharray="${circumference}"
                                    stroke-dashoffset="${offset}"
                                    style="filter:drop-shadow(0 0 6px ${gaugeColor}66)"/>
                            </svg>
                            <div class="gauge-inner">
                                <span class="gauge-value-text">${fmt(index, 3)}</span>
                                <span class="gauge-label-text">Severity</span>
                            </div>
                        </div>
                        <div class="risk-pill risk-${risk}">${severity.risk_level || 'N/A'}</div>
                        <span class="confidence-pill">Confidence: ${conf}</span>
                    </div>
                    <div class="severity-details">
                        <div class="severity-detail-row">
                            <span class="severity-detail-label">Severity Index</span>
                            <span class="severity-detail-value">${fmt(index, 4)}</span>
                        </div>
                        <div class="severity-detail-row">
                            <span class="severity-detail-label">Risk Level</span>
                            <span class="severity-detail-value">${severity.risk_level || 'N/A'}</span>
                        </div>
                        <div class="severity-detail-row">
                            <span class="severity-detail-label">Confidence Level</span>
                            <span class="severity-detail-value">${conf}</span>
                        </div>
                        <div style="margin-top:4px">
                            ${progressBar('Severity Index', pct, pct > 60 ? 'bar-red' : pct > 40 ? 'bar-amber' : 'bar-green')}
                        </div>
                    </div>
                </div>
            </div>
        </div>`;
}

// ─────────────────────────────────────────────────────────
// Main render function
// ─────────────────────────────────────────────────────────

function renderDashboard(metrics) {
    if (!metrics) return '';

    const coh      = metrics.coherence_analysis    || {};
    const phase    = metrics.phase_analysis         || {};
    const seg      = metrics.segmentation_analysis  || {};
    const shape    = metrics.shape_analysis         || {};
    const conf     = metrics.confidence_analysis    || {};
    const severity = metrics.severity_assessment    || {};

    return `
        <div class="analysis-dashboard">
            ${renderMetadata(metrics)}
            ${renderCoherence(coh)}
            ${renderPhase(phase)}
            ${renderSegmentation(seg)}
            ${renderShape(shape)}
            ${renderConfidence(conf)}
            ${renderSeverity(severity)}
        </div>`;
}
