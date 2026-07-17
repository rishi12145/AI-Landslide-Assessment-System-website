/**
 * app.js
 * Main application controller for the Landslide Assessment Platform.
 *
 * Responsible for:
 *  - File upload handling (Coherence + Phase TIFF pair)
 *  - Development sample loading
 *  - Spatial viewport 4-panel image population
 *  - Header status bar updates
 *  - VLM report generation
 *  - PDF export
 *  - Chat assistant
 *
 * All metric values are sourced exclusively from the Analysis Engine JSON.
 * No fake or fabricated values are ever displayed.
 */

// ─────────────────────────────────────────────────────────
// Global Application State
// ─────────────────────────────────────────────────────────
const state = {
    coherenceFile: null,
    phaseFile: null,
    metrics: null,

    // Image paths
    predictionPath:    null,
    heatmapPath:       null,
    overlayPath:       null,
    originalRgbPath:   null,

    // Filesystem paths (for PDF/VLM)
    predictionFsPath:  null,
    heatmapFsPath:     null,
    overlayFsPath:     null,
    originalRgbFsPath: null,

    // Report content
    reportText: '',

    // Chat
    chatHistory: [],

    // Samples
    availableSamples: [],
    activeSampleId: null
};

// ─────────────────────────────────────────────────────────
// DOM References
// ─────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);

const coherenceFileInput    = $('coherence-file-input');
const phaseFileInput        = $('phase-file-input');
const selectCoherenceBtn    = $('select-coherence-btn');
const selectPhaseBtn        = $('select-phase-btn');
const coherenceFilename     = $('coherence-filename');
const phaseFilename         = $('phase-filename');
const uploadValidationErrors = $('upload-validation-errors');
const dropZone              = $('drop-zone');
const analyzeBtn            = $('analyze-btn');
const regionSelect          = $('region-select');
const temporalSelect        = $('temporal-baseline-select');

const loadingOverlay        = $('loading-overlay');
const loadingText           = $('loading-text');

const sampleSelect          = $('sample-select');
const refreshSamplesBtn     = $('refresh-samples-btn');
const loadSampleBtn         = $('load-sample-btn');

// 4-panel viewport images
const vpImgOriginal    = $('vp-img-original');
const vpImgPrediction  = $('vp-img-prediction');
const vpImgHeatmap     = $('vp-img-heatmap');
const vpImgOverlay     = $('vp-img-overlay');

// Placeholder divs
const vpPhOriginal    = $('vp-ph-original');
const vpPhPrediction  = $('vp-ph-prediction');
const vpPhHeatmap     = $('vp-ph-heatmap');
const vpPhOverlay     = $('vp-ph-overlay');

// Header
const headerSampleId    = $('header-sample-id');
const headerRegion      = $('header-region');
const headerRiskLevel   = $('header-risk-level');

// Dashboard
const dashboardContainer = $('dashboard-container');

// Report
const reportBox         = $('report-box');
const pdfDownloadBtn    = $('pdf-download-btn');

// Chat
const chatInput         = $('chat-input');
const chatSendBtn       = $('chat-send-btn');
const chatMessages      = $('chat-messages');

// ─────────────────────────────────────────────────────────
// Loading Spinner
// ─────────────────────────────────────────────────────────
function setLoading(show, text = 'Processing...') {
    loadingText.textContent = text;
    loadingOverlay.classList.toggle('hidden', !show);
}

// ─────────────────────────────────────────────────────────
// Path helpers
// ─────────────────────────────────────────────────────────
function getAssetUrl(fullPath) {
    if (!fullPath) return '';
    const base = fullPath.replace(/\\/g, '/');
    const idx = base.indexOf('data/');
    return idx !== -1 ? '/' + base.substring(idx) : '/data';
}

// ─────────────────────────────────────────────────────────
// Viewport — Show all 4 panels
// ─────────────────────────────────────────────────────────
function setViewportImage(imgEl, phEl, url) {
    if (!url) return;
    phEl.style.display = 'none';
    imgEl.classList.remove('hidden');
    imgEl.src = url;
    imgEl.onload = () => imgEl.classList.add('loaded');
}

function populateViewport() {
    if (state.originalRgbPath)  setViewportImage(vpImgOriginal,   vpPhOriginal,   state.originalRgbPath);
    if (state.predictionPath)   setViewportImage(vpImgPrediction,  vpPhPrediction, state.predictionPath);
    if (state.heatmapPath)      setViewportImage(vpImgHeatmap,     vpPhHeatmap,    state.heatmapPath);
    if (state.overlayPath)      setViewportImage(vpImgOverlay,     vpPhOverlay,    state.overlayPath);
}

// ─────────────────────────────────────────────────────────
// Header status bar
// ─────────────────────────────────────────────────────────
function updateHeader(metrics) {
    if (!metrics) return;

    headerSampleId.textContent  = metrics.sample_id  || '—';
    headerRegion.textContent    = metrics.region      || '—';

    const severity = metrics.severity_assessment || {};
    const risk     = severity.risk_level         || '—';
    headerRiskLevel.textContent  = risk;

    // Color the risk badge
    headerRiskLevel.className = 'status-value';
    const riskClass = risk.toLowerCase().replace(/\s+/g, '-');
    const riskColors = {
        'very-low':  { color: '#6ee7b7' },
        'low':       { color: '#a5f3fc' },
        'moderate':  { color: '#fde68a' },
        'high':      { color: '#fed7aa' },
        'very-high': { color: '#fca5a5' }
    };
    const style = riskColors[riskClass];
    if (style) headerRiskLevel.style.color = style.color;
    else headerRiskLevel.style.color = '';
}

// ─────────────────────────────────────────────────────────
// Dashboard Rendering
// ─────────────────────────────────────────────────────────
function populateDashboard(metrics) {
    if (!metrics) return;
    dashboardContainer.innerHTML = renderDashboard(metrics);
}

// ─────────────────────────────────────────────────────────
// File Upload UI
// ─────────────────────────────────────────────────────────
selectCoherenceBtn.addEventListener('click', e => {
    e.stopPropagation();
    coherenceFileInput.click();
});

selectPhaseBtn.addEventListener('click', e => {
    e.stopPropagation();
    phaseFileInput.click();
});

coherenceFileInput.addEventListener('change', e => {
    if (e.target.files.length > 0) {
        state.coherenceFile = e.target.files[0];
        clearErrors();
        updateFileUI();
    }
});

phaseFileInput.addEventListener('change', e => {
    if (e.target.files.length > 0) {
        state.phaseFile = e.target.files[0];
        clearErrors();
        updateFileUI();
    }
});

dropZone.addEventListener('dragover', e => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', e => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    const files = Array.from(e.dataTransfer.files);
    if (files.length >= 2) {
        state.coherenceFile = files[0];
        state.phaseFile     = files[1];
    } else if (files.length === 1) {
        const name = files[0].name.toLowerCase();
        if (name.includes('coh'))        state.coherenceFile = files[0];
        else if (name.includes('phase')) state.phaseFile     = files[0];
        else if (!state.coherenceFile)   state.coherenceFile = files[0];
        else                             state.phaseFile     = files[0];
    }
    clearErrors();
    updateFileUI();
});

function clearErrors() {
    uploadValidationErrors.innerHTML = '';
    uploadValidationErrors.classList.add('hidden');
}

function showErrors(messages) {
    uploadValidationErrors.innerHTML = messages.map(m => `<p>${m}</p>`).join('');
    uploadValidationErrors.classList.remove('hidden');
}

function updateFileUI() {
    if (state.coherenceFile) {
        coherenceFilename.textContent = state.coherenceFile.name;
        coherenceFilename.classList.add('selected');
    } else {
        coherenceFilename.textContent = 'No file selected';
        coherenceFilename.classList.remove('selected');
    }

    if (state.phaseFile) {
        phaseFilename.textContent = state.phaseFile.name;
        phaseFilename.classList.add('selected');
    } else {
        phaseFilename.textContent = 'No file selected';
        phaseFilename.classList.remove('selected');
    }

    analyzeBtn.disabled = !(state.coherenceFile && state.phaseFile && regionSelect.value && temporalSelect.value);
}

regionSelect.addEventListener('change', updateFileUI);
temporalSelect.addEventListener('change', updateFileUI);

// ─────────────────────────────────────────────────────────
// Development Sample Loading
// ─────────────────────────────────────────────────────────
async function loadDevelopmentSamples() {
    try {
        const res = await fetch('/api/dataset');
        if (!res.ok) throw new Error('Failed to load samples.');

        const samples = await res.json();
        state.availableSamples = Array.isArray(samples) ? samples : [];

        sampleSelect.innerHTML = '';

        if (state.availableSamples.length === 0) {
            const opt = document.createElement('option');
            opt.value = '';
            opt.textContent = 'No development samples available';
            sampleSelect.appendChild(opt);
            sampleSelect.disabled = true;
            loadSampleBtn.disabled = true;
            return;
        }

        sampleSelect.disabled  = false;
        loadSampleBtn.disabled = false;

        state.availableSamples.forEach(sample => {
            const opt = document.createElement('option');
            opt.value = sample.sample_id;
            opt.textContent = `${sample.sample_id} | ${sample.region || 'N/A'} | ${sample.risk_level || 'N/A'}`;
            sampleSelect.appendChild(opt);
        });

        state.activeSampleId = state.availableSamples[0].sample_id;
        sampleSelect.value   = state.activeSampleId;

    } catch (err) {
        console.error('Sample load error:', err);
        sampleSelect.innerHTML = '<option value="">Development samples unavailable</option>';
        sampleSelect.disabled  = true;
        loadSampleBtn.disabled = true;
    }
}

refreshSamplesBtn.addEventListener('click', loadDevelopmentSamples);

sampleSelect.addEventListener('change', () => {
    state.activeSampleId = sampleSelect.value;
});

loadSampleBtn.addEventListener('click', async () => {
    const sampleId = sampleSelect.value || state.activeSampleId;
    if (!sampleId) return;

    setLoading(true, `Loading ${sampleId} and generating VLM report...`);

    try {
        const res = await fetch('/api/analyze-dev', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ sample_id: sampleId })
        });

        if (!res.ok) {
            const err = await res.text();
            throw new Error(err || 'Development analysis failed.');
        }

        const data = await res.json();
        applyAnalysisResult(data, sampleId);
        await fetchVlmReport(sampleId);

    } catch (err) {
        alert(`Failed to load development sample: ${err.message}`);
        console.error(err);
    } finally {
        setLoading(false);
    }
});

// ─────────────────────────────────────────────────────────
// Analyze Pipeline (Production — file upload)
// ─────────────────────────────────────────────────────────
analyzeBtn.addEventListener('click', async () => {
    clearErrors();

    const validationMsgs = [];
    if (!state.coherenceFile) validationMsgs.push('Please select a Coherence TIFF.');
    if (!state.phaseFile)     validationMsgs.push('Please select a Phase TIFF.');
    if (!regionSelect.value)  validationMsgs.push('Please select a Region.');
    if (!temporalSelect.value) validationMsgs.push('Please select a Temporal Baseline.');
    if (validationMsgs.length > 0) {
        showErrors(validationMsgs);
        return;
    }

    setLoading(true, 'Uploading InSAR files and running Analysis Engine...');

    try {
        const formData = new FormData();
        formData.append('coherence_file', state.coherenceFile);
        formData.append('phase_file',     state.phaseFile);
        formData.append('region',         regionSelect.value);
        formData.append('temporal',       temporalSelect.value);

        const res = await fetch('/api/analyze', { method: 'POST', body: formData });

        if (!res.ok) {
            let detail = 'Analysis engine execution failed.';
            const errText = await res.text();
            try {
                const errData = JSON.parse(errText);
                if (typeof errData.detail === 'string') detail = errData.detail;
                else if (Array.isArray(errData.detail)) detail = errData.detail.map(i => i.msg || JSON.stringify(i)).join('\n');
            } catch (_) { if (errText) detail = errText; }
            throw new Error(detail.split('\n')[0]);
        }

        const data = await res.json();
        applyAnalysisResult(data, null);
        await fetchVlmReport(null);

    } catch (err) {
        alert(`Analysis error: ${err.message}`);
        console.error(err);
    } finally {
        setLoading(false);
    }
});

// ─────────────────────────────────────────────────────────
// Apply result to state and UI
// ─────────────────────────────────────────────────────────
function applyAnalysisResult(data, sampleId) {
    state.metrics        = data.metrics;
    state.activeSampleId = sampleId;

    state.predictionPath    = getAssetUrl(data.prediction_path);
    state.heatmapPath       = getAssetUrl(data.heatmap_path);
    state.overlayPath       = getAssetUrl(data.overlay_path);
    state.originalRgbPath   = data.original_rgb_url || getAssetUrl(data.original_rgb_path || '');

    state.predictionFsPath  = data.prediction_path    || null;
    state.heatmapFsPath     = data.heatmap_path       || null;
    state.overlayFsPath     = data.overlay_path        || null;
    state.originalRgbFsPath = data.original_rgb_path  || null;

    // Populate all four viewport panels
    populateViewport();

    // Update header status bar with real JSON data
    updateHeader(state.metrics);

    // Render analysis dashboard sections A–G
    populateDashboard(state.metrics);
}

// ─────────────────────────────────────────────────────────
// VLM Report Generation
// ─────────────────────────────────────────────────────────
async function fetchVlmReport(sampleId) {
    if (!state.metrics) return;

    reportBox.innerHTML = `
        <div class="dashboard-placeholder">
            <div class="placeholder-icon" style="animation:pulse-anim 2s infinite">
                <svg width="40" height="40" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/>
                </svg>
            </div>
            <p>Synthesizing multi-modal assessment report...</p>
        </div>`;

    try {
        const res = await fetch('/api/vlm-report', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                json_data:            state.metrics,
                sample_id:            sampleId,
                prediction_path:      state.predictionFsPath  || state.predictionPath,
                heatmap_path:         state.heatmapFsPath     || state.heatmapPath,
                overlay_path:         state.overlayFsPath     || state.overlayPath,
                original_image_path:  state.originalRgbFsPath || state.originalRgbPath
            })
        });

        if (!res.ok) throw new Error('Report fetch failed.');

        const reportData = await res.json();
        state.reportText = reportData.report || '';
        reportBox.innerHTML = renderMarkdown(state.reportText);

        pdfDownloadBtn.classList.remove('hidden');
        chatInput.removeAttribute('disabled');
        chatSendBtn.removeAttribute('disabled');

    } catch (err) {
        reportBox.innerHTML = `<p class="text-red" style="padding:16px">Failed to generate report: ${err.message}</p>`;
        console.error('VLM report error:', err);
    }
}

// ─────────────────────────────────────────────────────────
// Markdown Renderer
// Handles the unified 8-section report format produced by VLM.
// ─────────────────────────────────────────────────────────
function renderMarkdown(md) {
    if (!md) return '';

    // Escape raw HTML so backend content can't inject scripts
    let html = md
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');

    // Headers (must come before bold/italic so ** in headings is preserved)
    html = html.replace(/^### (.*$)/gim, '<h3>$1</h3>');
    html = html.replace(/^## (.*$)/gim,  '<h2>$1</h2>');
    html = html.replace(/^# (.*$)/gim,   '<h1>$1</h1>');

    // Horizontal rule
    html = html.replace(/^---+$/gim, '<hr>');

    // Bold and italic
    html = html.replace(/\*\*\*(.+?)\*\*\*/gim, '<strong><em>$1</em></strong>');
    html = html.replace(/\*\*(.+?)\*\*/gim,      '<strong>$1</strong>');
    html = html.replace(/\*(.+?)\*/gim,           '<em>$1</em>');

    // Convert numbered list lines into <li> inside <ol>
    // Group consecutive numbered items
    html = html.replace(/((?:^[ \t]*\d+\. .+\n?)+)/gim, match => {
        const items = match.trim().split('\n').map(line => {
            const text = line.replace(/^[ \t]*\d+\. /, '').trim();
            return `<li>${text}</li>`;
        }).join('');
        return `<ol>${items}</ol>\n`;
    });

    // Convert bullet list lines into <li> inside <ul>
    html = html.replace(/((?:^[ \t]*[-*] .+\n?)+)/gim, match => {
        const items = match.trim().split('\n').map(line => {
            const text = line.replace(/^[ \t]*[-*] /, '').trim();
            return `<li>${text}</li>`;
        }).join('');
        return `<ul>${items}</ul>\n`;
    });

    // Wrap double-newline separated blocks in paragraphs
    // (skip lines already wrapped in block tags)
    const blockTags = /^<(h[1-6]|ul|ol|li|hr|blockquote)/i;
    const lines = html.split('\n');
    const output = [];
    let paragraph = [];

    for (const line of lines) {
        if (blockTags.test(line.trim()) || line.trim() === '') {
            if (paragraph.length) {
                output.push(`<p>${paragraph.join('<br>')}</p>`);
                paragraph = [];
            }
            if (line.trim()) output.push(line);
        } else {
            paragraph.push(line);
        }
    }
    if (paragraph.length) output.push(`<p>${paragraph.join('<br>')}</p>`);

    return output.join('\n');
}


// ─────────────────────────────────────────────────────────
// PDF Export
// ─────────────────────────────────────────────────────────
pdfDownloadBtn.addEventListener('click', async () => {
    if (!state.metrics) return;

    setLoading(true, 'Compiling PDF report...');

    try {
        const res = await fetch('/api/pdf', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                json_data:           state.metrics,
                report_content:      state.reportText,
                original_image_path: state.originalRgbFsPath || state.originalRgbPath  || '',
                prediction_path:     state.predictionFsPath  || state.predictionPath   || '',
                heatmap_path:        state.heatmapFsPath     || state.heatmapPath      || '',
                overlay_path:        state.overlayFsPath     || state.overlayPath      || ''
            })
        });

        if (!res.ok) throw new Error('PDF generation failed.');

        const blob = await res.blob();
        const url  = window.URL.createObjectURL(blob);
        const a    = document.createElement('a');
        a.href     = url;
        a.download = `Landslide_Report_${(state.metrics.sample_id || 'report')}.pdf`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        window.URL.revokeObjectURL(url);

    } catch (err) {
        alert(`PDF export failed: ${err.message}`);
        console.error(err);
    } finally {
        setLoading(false);
    }
});

// ─────────────────────────────────────────────────────────
// Chat Assistant
// ─────────────────────────────────────────────────────────
chatSendBtn.addEventListener('click', sendChat);
chatInput.addEventListener('keydown', e => { if (e.key === 'Enter') sendChat(); });

async function sendChat() {
    const query = chatInput.value.trim();
    if (!query || !state.metrics) return;

    appendChatMsg('chat-user', query);
    chatInput.value = '';

    const bubble = appendChatMsg('chat-assistant', 'Thinking...');

    try {
        const res = await fetch('/api/chat?stream=true', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                report_content:      state.reportText,
                user_query:          query,
                chat_history:        state.chatHistory,
                // Enrich with full analysis context so VLM gives grounded answers
                json_data:           state.metrics             || null,
                original_image_path: state.originalRgbFsPath  || state.originalRgbPath  || null,
                prediction_path:     state.predictionFsPath   || state.predictionPath   || null,
                heatmap_path:        state.heatmapFsPath      || state.heatmapPath      || null,
                overlay_path:        state.overlayFsPath      || state.overlayPath      || null,
            })
        });

        if (!res.ok) throw new Error('Chat request failed.');

        bubble.textContent = '';
        const reader  = res.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let   answer  = '';

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            answer += decoder.decode(value, { stream: true });
            bubble.textContent = answer;
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }

        state.chatHistory.push({ role: 'user',      content: query  });
        state.chatHistory.push({ role: 'assistant', content: answer });

    } catch (err) {
        bubble.textContent = `Error: ${err.message}`;
        console.error('Chat error:', err);
    }
}

function appendChatMsg(cssClass, text) {
    const div = document.createElement('div');
    div.className = `chat-msg ${cssClass}`;
    div.textContent = text;
    chatMessages.appendChild(div);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return div;
}

// ─────────────────────────────────────────────────────────
// Initialization
// ─────────────────────────────────────────────────────────
loadDevelopmentSamples();
updateFileUI();
