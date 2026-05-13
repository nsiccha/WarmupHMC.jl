// =============================================================================
// mcmc-viz.js
// =============================================================================
//
// Public API:
//
//   renderMCMC(containers, data, step)
//     Draw the current animation frame across all panels (single dataset).
//
//   initMCMCPlayer(panels, controls)
//     Wire up play/pause/reset and the steps-per-sec input.
//     panels is an array of { containers, data } — all panels share one step
//     counter and advance together.
//     Returns { play, pause, reset, load }.
//     load(newPanels) accepts a matching array of { containers, data }.
//
// Data shape:
// {
//   xGrid:        number[]            // W grid x-positions (ascending)
//   yGrid:        number[]            // H grid y-positions (ascending)
//   logpdfs:      number[][]          // [H][W]  logpdfs[row][col] ~ yGrid[row], xGrid[col]
//   trajectories: [number,number][][] // One array per HMC/NUTS proposal.
//                                     // trajectories[i] is the leapfrog path of proposal i.
//                                     // trajectories[i][0]  = chain position entering step i.
//                                     // trajectories[i][-1] = proposed point (accepted or not).
// }
//
// The "chain" is the sequence of starting points:
//   chainSamples[i] = trajectories[i][0]
// plus the final accepted/rejected endpoint after the last trajectory.
//
// `step` is a float. Math.floor(step) = number of fully drawn trajectories.
// The fractional part controls how far along the current trajectory is drawn.
//
// containers shape:
// {
//   viz:    HTMLElement   // main heatmap panel
//   histX:  HTMLElement   // x-marginal histogram panel
//   histY:  HTMLElement   // y-marginal histogram panel
//   traceX: HTMLElement   // x traceplot panel
//   traceY: HTMLElement   // y traceplot panel
// }
// =============================================================================


// ─── Colour helpers ────────────────────────────────────────────────────────────
function hexToRGB(hex) {
  return [
    parseInt(hex.slice(1,3), 16),
    parseInt(hex.slice(3,5), 16),
    parseInt(hex.slice(5,7), 16),
  ];
}
function rgbaStr(hex, a) {
  const [r,g,b] = hexToRGB(hex);
  return `rgba(${r},${g},${b},${a})`;
}

// chi-feng palette
const BLUE   = '#6699bb';   // density heatmap, histograms, contours  rgb(102,153,187)
const TRAJ   = '#333333';   // leapfrog trajectory (dark gray)
const GREEN  = '#44cc44';   // accepted chain samples
const YELLOW = '#e8a020';   // running mean


// ─── Canvas utility ────────────────────────────────────────────────────────────
function getOrCreateCanvas(container, className) {
  let c = container.querySelector(`canvas.${className}`);
  if (!c) {
    c = document.createElement('canvas');
    c.className = className;
    c.style.cssText = 'display:block;width:100%;height:100%;';
    container.appendChild(c);
  }
  const w = container.clientWidth  || 300;
  const h = container.clientHeight || 300;
  if (c.width !== w || c.height !== h) {
    c.width  = w;
    c.height = h;
    c._heatmapData = null; // invalidate cache when canvas is resized
  }
  return c;
}


// =============================================================================
// renderMCMC — master render function
// =============================================================================
function renderMCMC(containers, data, step = Infinity, options = {}) {
  const { xGrid, yGrid, trajectories } = data;

  // order[animStep] gives the trajectory index for that animation step.
  // Defaults to sequential [0, 1, 2, ...].
  const order = data.order || trajectories.map((_, i) => i);

  const totalTraj   = order.length;
  const clampedStep = Math.min(step, totalTraj - 1 + 0.9999);
  const doneTraj    = Math.min(Math.floor(clampedStep), totalTraj - 1);
  const fraction    = clampedStep - doneTraj; // 0..1, progress into current traj

  // Transition support: a lightweight snapshot that overrides the heatmap/KDE and
  // supplies pre-computed chain samples as the history before the transition point.
  // options.transition = { data: { pdfs, xGrid?, yGrid?, xPdf?, yPdf?, initialSamples }, startStep: N }
  const tr       = options.transition ?? null;
  const startIdx = tr ? Math.floor(tr.startStep) : 0;

  // Base chain samples: only trajectories from startIdx onward.
  const baseChainSamples = order.slice(startIdx, doneTraj + 1).map(k => {
    const ro = data.trajectories_order?.[k];
    return trajectories[k][ro ? ro[0] : 0];
  });

  // Full display samples: pre-computed history from transition + new base samples.
  const chainSamples = tr
    ? [...tr.data.initialSamples, ...baseChainSamples]
    : baseChainSamples;

  // Effective heatmap data: transition's pdfs/grid take priority over base data.
  const effHeatData = tr ? {
    ...data,
    pdfs:  tr.data.pdfs  ?? data.pdfs,
    xGrid: tr.data.xGrid ?? xGrid,
    yGrid: tr.data.yGrid ?? yGrid,
  } : data;
  const effXGrid = effHeatData.xGrid;
  const effYGrid = effHeatData.yGrid;
  const effXPdf  = tr ? (tr.data.xPdf ?? data.xPdf) : data.xPdf;
  const effYPdf  = tr ? (tr.data.yPdf ?? data.yPdf) : data.yPdf;

  // Zoom-to-fit: compute a viewport that tightly frames all visible chain samples.
  let viewport = null;
  if (options.zoomToFit && chainSamples.length > 0) {
    const xs = chainSamples.map(([x]) => x);
    const ys = chainSamples.map(([, y]) => y);
    let vxMin = Math.min(...xs), vxMax = Math.max(...xs);
    let vyMin = Math.min(...ys), vyMax = Math.max(...ys);
    const pw = Math.max(vxMax - vxMin, 0.1) * 0.2;
    const ph = Math.max(vyMax - vyMin, 0.1) * 0.2;
    viewport = { xMin: vxMin - pw, xMax: vxMax + pw, yMin: vyMin - ph, yMax: vyMax + ph };
  }

  _renderHeatmap(containers.viz, effHeatData, order, doneTraj, fraction, chainSamples, viewport);
  _renderHistogram(containers.histX, chainSamples, 'x', effXGrid[0], effXGrid[effXGrid.length-1], effXGrid, effXPdf);
  _renderHistogram(containers.histY, chainSamples, 'y', effYGrid[0], effYGrid[effYGrid.length-1], effYGrid, effYPdf);
  _renderTrace(containers.traceX, chainSamples, 'x', effXGrid[0], effXGrid[effXGrid.length-1], 'X');
  _renderTrace(containers.traceY, chainSamples, 'y', effYGrid[0], effYGrid[effYGrid.length-1], 'Y');
}


// =============================================================================
// Heatmap panel
// =============================================================================
function _renderHeatmap(container, data, order, doneTraj, fraction, chainSamples, viewport) {
  if (!container) return;
  const { xGrid, yGrid, pdfs, trajectories } = data;
  const canvas = getOrCreateCanvas(container, 'mcmc-main');
  const W = canvas.width, H = canvas.height;
  const ctx = canvas.getContext('2d');

  const xMin = xGrid[0], xMax = xGrid[xGrid.length-1];
  const yMin = yGrid[0], yMax = yGrid[yGrid.length-1];

  // When a viewport is active (zoom-to-fit), use it for all coordinate mapping.
  const vxMin = viewport ? viewport.xMin : xMin;
  const vxMax = viewport ? viewport.xMax : xMax;
  const vyMin = viewport ? viewport.yMin : yMin;
  const vyMax = viewport ? viewport.yMax : yMax;

  function toCanvas(wx, wy) {
    return [
      ((wx - vxMin) / (vxMax - vxMin)) * W,
      H - ((wy - vyMin) / (vyMax - vyMin)) * H,
    ];
  }

  // ── Heatmap (cached per data identity as an offscreen canvas) ───────────────
  if (canvas._heatmapData !== data) {
    const rows = yGrid.length, cols = xGrid.length;
    let maxPdf = 0;
    for (let r = 0; r < rows; r++)
      for (let c = 0; c < cols; c++) {
        const v = pdfs[r][c];
        if (v > maxPdf) maxPdf = v;
      }
    if (maxPdf === 0) maxPdf = 1;

    // chi-feng style: white background, muted blue (102,153,187) with alpha = sqrt(t)
    // Render at full data-range into an offscreen canvas; zoomed views crop-draw from it.
    const oc = document.createElement('canvas');
    oc.width = W; oc.height = H;
    const ocCtx = oc.getContext('2d');
    const img = ocCtx.createImageData(W, H);
    for (let py = 0; py < H; py++) {
      const wy = yMin + (1 - py/H) * (yMax - yMin);
      let r1 = 0;
      while (r1 < rows-2 && yGrid[r1+1] < wy) r1++;
      const r2 = Math.min(r1+1, rows-1);
      const ty = yGrid[r2] !== yGrid[r1] ? (wy - yGrid[r1]) / (yGrid[r2] - yGrid[r1]) : 0;
      for (let px = 0; px < W; px++) {
        const wx = xMin + (px/W) * (xMax - xMin);
        let c1 = 0;
        while (c1 < cols-2 && xGrid[c1+1] < wx) c1++;
        const c2 = Math.min(c1+1, cols-1);
        const tx = xGrid[c2] !== xGrid[c1] ? (wx - xGrid[c1]) / (xGrid[c2] - xGrid[c1]) : 0;
        const v  = (1-ty)*(1-tx)*pdfs[r1][c1] + (1-ty)*tx*pdfs[r1][c2]
                 +    ty *(1-tx)*pdfs[r2][c1] +    ty *tx*pdfs[r2][c2];
        const t  = Math.max(0, Math.min(1, v / maxPdf));
        const a  = Math.sqrt(t); // sqrt compression, chi-feng style
        const idx = (py*W + px)*4;
        // blend blue (102,153,187) over white (255,255,255)
        img.data[idx  ] = Math.round(255 + (102 - 255) * a);
        img.data[idx+1] = Math.round(255 + (153 - 255) * a);
        img.data[idx+2] = Math.round(255 + (187 - 255) * a);
        img.data[idx+3] = 255;
      }
    }
    ocCtx.putImageData(img, 0, 0);
    canvas._heatmapCanvas = oc;
    canvas._heatmapData   = data;
  }

  // Draw heatmap — full view or cropped/zoomed
  ctx.fillStyle = '#fff';
  ctx.fillRect(0, 0, W, H);
  const oc = canvas._heatmapCanvas;
  if (viewport) {
    // Map viewport world coords → source pixel rect in the offscreen canvas
    const sx = (vxMin - xMin) / (xMax - xMin) * W;
    const sw = (vxMax - vxMin) / (xMax - xMin) * W;
    const sy = (1 - (vyMax - yMin) / (yMax - yMin)) * H;
    const sh = (vyMax - vyMin) / (yMax - yMin) * H;
    // Clamp source rect to offscreen bounds, adjusting destination accordingly
    const sxC = Math.max(0, sx), syC = Math.max(0, sy);
    const exC = Math.min(W, sx + sw), eyC = Math.min(H, sy + sh);
    if (exC > sxC && eyC > syC) {
      const dx = (sxC - sx) / sw * W, dy = (syC - sy) / sh * H;
      const dw = (exC - sxC) / sw * W, dh = (eyC - syC) / sh * H;
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = 'high';
      ctx.drawImage(oc, sxC, syC, exC - sxC, eyC - syC, dx, dy, dw, dh);
    }
  } else {
    ctx.drawImage(oc, 0, 0);
  }


  // ── Current (active) trajectory, drawn up to `fraction` ───────────────────
  const activeTrajIdx = order[doneTraj];
  const activeTraj    = trajectories[activeTrajIdx];
  if (activeTraj && activeTraj.length > 0) {
    // Per-trajectory reveal order (defaults to linear 0..N-1)
    const ro       = data.trajectories_order?.[activeTrajIdx]
                     ?? activeTraj.map((_, i) => i);
    const N        = ro.length;
    const visLen   = fraction * (N - 1);
    const fullPts  = Math.min(Math.floor(visLen), N - 1);
    const partFrac = visLen - fullPts;

    // Set of revealed point indices
    const revealed = new Set(ro.slice(0, fullPts + 1));

    ctx.save();
    ctx.strokeStyle = TRAJ;
    ctx.lineWidth   = 1.8;

    // Draw each segment between adjacent original indices that are both revealed
    for (let j = 0; j < activeTraj.length - 1; j++) {
      if (revealed.has(j) && revealed.has(j + 1)) {
        ctx.beginPath();
        ctx.moveTo(...toCanvas(...activeTraj[j]));
        ctx.lineTo(...toCanvas(...activeTraj[j + 1]));
        ctx.stroke();
      }
    }

    // Partial tip segment: from last fully-revealed point toward next in reveal order
    if (fullPts + 1 < N && partFrac > 0) {
      const p0 = ro[fullPts], p1 = ro[fullPts + 1];
      if (Math.abs(p1 - p0) === 1) {  // adjacent in original order
        const [x0, y0] = activeTraj[p0], [x1, y1] = activeTraj[p1];
        const [cx, cy] = toCanvas(x0 + partFrac*(x1-x0), y0 + partFrac*(y1-y0));
        ctx.beginPath();
        ctx.moveTo(...toCanvas(x0, y0));
        ctx.lineTo(cx, cy);
        ctx.stroke();
      }
    }

    // Dots at each revealed point
    ctx.globalAlpha = 0.5;
    for (const idx of revealed) {
      ctx.beginPath();
      ctx.arc(...toCanvas(...activeTraj[idx]), 2, 0, Math.PI*2);
      ctx.fillStyle = TRAJ;
      ctx.fill();
    }

    // Two tip dots: one at each extreme (left/right) of the revealed segment.
    // The tip that is currently being extended animates toward the next point.
    ctx.globalAlpha = 1.0;
    const minIdx   = Math.min(...revealed);
    const maxIdx   = Math.max(...revealed);
    const nextOrig = (fullPts + 1 < N) ? ro[fullPts + 1] : null;

    function tipPos(extremeIdx) {
      if (nextOrig !== null && partFrac > 0 && Math.abs(nextOrig - extremeIdx) === 1
          && nextOrig === (extremeIdx < nextOrig ? extremeIdx + 1 : extremeIdx - 1)) {
        const [x0,y0] = activeTraj[extremeIdx], [x1,y1] = activeTraj[nextOrig];
        return toCanvas(x0 + partFrac*(x1-x0), y0 + partFrac*(y1-y0));
      }
      return toCanvas(...activeTraj[extremeIdx]);
    }

    for (const [tx, ty] of [tipPos(minIdx), tipPos(maxIdx)]) {
      ctx.beginPath();
      ctx.arc(tx, ty, 3.5, 0, Math.PI*2);
      ctx.fillStyle = TRAJ;
      ctx.fill();
    }

    ctx.restore();
  }

  // ── Chain samples (accepted positions) ────────────────────────────────────
  if (chainSamples.length > 0) {
    // Draw small dots at each accepted sample
    chainSamples.forEach(([wx,wy], j) => {
      const [cx,cy] = toCanvas(wx,wy);
      const isLast  = j === chainSamples.length - 1;
      ctx.beginPath();
      ctx.arc(cx, cy, isLast ? 3 : 2, 0, Math.PI*2);
      ctx.fillStyle = isLast ? GREEN : 'rgba(100,100,100,0.6)';
      ctx.fill();
    });
  }

}


// =============================================================================
// Marginal histogram panel
// axis: 'x' or 'y'
// =============================================================================
function _renderHistogram(container, chainSamples, axis, worldMin, worldMax, grid, pdf) {
  if (!container) return;
  const canvas = getOrCreateCanvas(container, 'mcmc-hist');
  const W = canvas.width, H = canvas.height;
  const ctx = canvas.getContext('2d');

  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#fff';
  ctx.fillRect(0, 0, W, H);

  // Label
  ctx.save();
  ctx.font      = '11px "Source Code Pro", monospace';
  ctx.fillStyle = 'rgba(80,80,80,0.75)';
  ctx.textAlign = 'left';
  ctx.fillText(axis.toUpperCase(), 6, 14);
  ctx.restore();

  const PAD = { top: 18, bottom: 4, left: 8, right: 8 };
  const plotW = W - PAD.left - PAD.right;
  const plotH = H - PAD.top  - PAD.bottom;

  // Histogram bars (only when enough samples)
  if (chainSamples.length >= 2) {
    const vals   = chainSamples.map(p => axis === 'x' ? p[0] : p[1]);
    const NBINS  = 30;
    const counts = new Array(NBINS).fill(0);
    vals.forEach(v => {
      const bin = Math.floor((v - worldMin) / (worldMax - worldMin) * NBINS);
      counts[Math.max(0, Math.min(NBINS-1, bin))]++;
    });
    const maxCount = Math.max(...counts, 1);
    counts.forEach((count, i) => {
      const bx = PAD.left + (i / NBINS) * plotW;
      const bw = plotW / NBINS - 1;
      const bh = (count / maxCount) * plotH;
      ctx.fillStyle = rgbaStr(BLUE, 0.65);
      ctx.fillRect(bx, PAD.top + plotH - bh, bw, bh);
    });
  }

  // KDE overlay — always drawn when PDF data is available
  if (grid && pdf && pdf.length > 0) {
    const maxPdf = Math.max(...pdf);
    if (maxPdf > 0) {
      ctx.save();
      ctx.strokeStyle = rgbaStr(BLUE, 0.9);
      ctx.lineWidth   = 1.5;
      ctx.beginPath();
      grid.forEach((v, i) => {
        const cx = PAD.left + (v - worldMin) / (worldMax - worldMin) * plotW;
        const cy = PAD.top  + plotH - (pdf[i] / maxPdf) * plotH;
        i === 0 ? ctx.moveTo(cx, cy) : ctx.lineTo(cx, cy);
      });
      ctx.stroke();
      ctx.restore();
    }
  }
}


// =============================================================================
// Traceplot panel
// =============================================================================
// Traceplot: parameter values on x-axis (matching histogram above),
// time/step index on y-axis increasing downward (step 0 at top, latest at bottom).
function _renderTrace(container, chainSamples, axis, worldMin, worldMax, label) {
  if (!container) return;
  const canvas = getOrCreateCanvas(container, 'mcmc-trace');
  const W = canvas.width, H = canvas.height;
  const ctx = canvas.getContext('2d');

  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#fff';
  ctx.fillRect(0, 0, W, H);

  const PAD = { top: 6, bottom: 4, left: 8, right: 8 };
  const plotW = W - PAD.left - PAD.right;
  const plotH = H - PAD.top  - PAD.bottom;

  // Axis label (top-left corner)
  ctx.save();
  ctx.font      = '11px "Source Code Pro", monospace';
  ctx.fillStyle = 'rgba(80,80,80,0.75)';
  ctx.textAlign = 'left';
  ctx.fillText(label, PAD.left, PAD.top + 11);
  ctx.restore();

  if (chainSamples.length < 2) return;

  const vals = chainSamples.map(p => axis === 'x' ? p[0] : p[1]);
  const N    = vals.length;

  // x maps parameter value, y maps step index (N-1=top nearest histogram, 0=bottom)
  function px(v) { return PAD.left + (v - worldMin) / (worldMax - worldMin) * plotW; }
  function py(i) { return PAD.top  + (1 - i / Math.max(N - 1, 1)) * plotH; }

  // Trace line
  ctx.save();
  ctx.strokeStyle = rgbaStr(BLUE, 0.75);
  ctx.lineWidth   = 1;
  ctx.beginPath();
  vals.forEach((v, i) => i === 0 ? ctx.moveTo(px(v), py(i)) : ctx.lineTo(px(v), py(i)));
  ctx.stroke();

  // Current position dot (bottom of trace) — green to match heatmap current sample
  const last = vals[N-1];
  ctx.beginPath();
  ctx.arc(px(last), py(N-1), 3, 0, Math.PI*2);
  ctx.fillStyle = GREEN;
  ctx.fill();
  ctx.restore();
}


// =============================================================================
// Shared traceplot panel
//
// spec: { label, logScale, traces: [{ values, label }] }
// nSteps: how many values to show (= Math.floor(currentStep) + 1)
// =============================================================================
const SHARED_COLORS = ['#6699bb', '#cc6644', '#44aa88', '#9966cc', '#ccaa22'];

function _renderSharedTrace(container, spec, nSteps) {
  if (!container) return;
  const canvas = getOrCreateCanvas(container, 'mcmc-shared-trace');
  const W = canvas.width, H = canvas.height;
  const ctx = canvas.getContext('2d');

  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#fff';
  ctx.fillRect(0, 0, W, H);

  const PAD   = { top: 18, bottom: 6, left: 52, right: 10 };
  const plotW = W - PAD.left - PAD.right;
  const plotH = H - PAD.top  - PAD.bottom;

  // Panel label (top-left)
  ctx.save();
  ctx.font      = '11px "Source Code Pro", monospace';
  ctx.fillStyle = 'rgba(80,80,80,0.75)';
  ctx.textAlign = 'left';
  ctx.fillText(spec.label || '', 4, 13);
  ctx.restore();

  if (!spec.traces || spec.traces.length === 0 || nSteps < 1) return;

  const logScale = !!spec.logScale;
  const n        = Math.min(nSteps, Math.max(...spec.traces.map(t => t.values.length)));

  // Compute y-range across all visible trace values
  let yMin = Infinity, yMax = -Infinity;
  spec.traces.forEach(({ values }) => {
    for (let i = 0; i < Math.min(n, values.length); i++) {
      const v = logScale ? Math.log(values[i]) : values[i];
      if (isFinite(v)) { if (v < yMin) yMin = v; if (v > yMax) yMax = v; }
    }
  });
  if (!isFinite(yMin)) return;
  if (yMin === yMax) { yMin -= 0.5; yMax += 0.5; }

  function px(i) { return PAD.left + (i / Math.max(n - 1, 1)) * plotW; }
  function py(v) {
    const val = logScale ? Math.log(v) : v;
    return PAD.top + plotH - (val - yMin) / (yMax - yMin) * plotH;
  }

  // y-axis labels (left side): min at bottom, max at top
  ctx.save();
  ctx.font      = '10px "Source Code Pro", monospace';
  ctx.fillStyle = 'rgba(80,80,80,0.75)';
  ctx.textAlign = 'right';
  const fmtY = v => logScale ? Math.exp(v).toPrecision(2) : v.toPrecision(3);
  ctx.fillText(fmtY(yMax), PAD.left - 4, PAD.top + 8);
  ctx.fillText(fmtY(yMin), PAD.left - 4, PAD.top + plotH);
  ctx.restore();

  // Traces
  spec.traces.forEach(({ values, label }, ti) => {
    const color = SHARED_COLORS[ti % SHARED_COLORS.length];
    const N = Math.min(n, values.length);
    if (N < 1) return;

    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth   = 1.5;
    ctx.beginPath();
    let started = false;
    for (let i = 0; i < N; i++) {
      const v = values[i];
      if (logScale && v <= 0) { started = false; continue; }
      const x = px(i), y = py(v);
      if (!started) { ctx.moveTo(x, y); started = true; }
      else            ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Trace label: anchored inside the right margin so it never goes off-screen
    if (label) {
      ctx.font      = '10px "Source Code Pro", monospace';
      ctx.fillStyle = color;
      ctx.textAlign = 'right';
      ctx.fillText(label, W - PAD.right - 2, PAD.top + 10 + ti * 12);
    }
    ctx.restore();
  });
}


// =============================================================================
// initMCMCPlayer
//
// panels:       Array of { containers, data }
// controls:     { playBtn, stepsInput, stepDisplay }
// sharedTraces: Array of { container, spec } (optional)
//               All panels share a single step counter and render together.
// =============================================================================
function initMCMCPlayer(panels, controls, sharedTraces) {
  const { playBtn, stepDisplay, progressCanvas, speedDisplay } = controls;

  let currentPanels = panels;
  let currentStep   = 0;
  let playing       = false;
  let lastTime      = null;
  let rafId         = null;

  // Per-panel zoom-to-fit toggle
  let zoomStates = panels.map(() => false);

  // Transitions: [{ step: N, panels: [{containers, data}, ...] }], sorted by step.
  // When currentStep >= step, that transition's datasets replace the base panels for rendering.
  let transitions = [];

  // Maximized subplot overlay: when set, draw() also renders that subplot live into overlayEl.
  let maximizedInfo = null;

  function setMaximized(panelIdx, containerKey, overlayEl) {
    if (maximizedInfo) maximizedInfo.overlayEl.remove();
    maximizedInfo = { type: 'panel', panelIdx, containerKey, overlayEl };
    if (!playing) draw();
  }

  function setMaximizedShared(traceIdx, overlayEl) {
    if (maximizedInfo) maximizedInfo.overlayEl.remove();
    maximizedInfo = { type: 'shared', traceIdx, overlayEl };
    if (!playing) draw();
  }

  function clearMaximized() {
    if (!maximizedInfo) return;
    maximizedInfo.overlayEl.remove();
    maximizedInfo = null;
    if (!playing) draw();
  }

  // ── Focus tracking ────────────────────────────────────────────────────────────
  // Keyboard events are only handled when the viz is "focused" — i.e. the user
  // has clicked on one of its elements more recently than clicking elsewhere.
  // External code (e.g. reveal.js) can call focus()/blur() directly.
  let vizFocused = false;
  function focus() { vizFocused = true;  }
  function blur()  { vizFocused = false; }

  // All DOM elements that belong to this player (panels + controls + shared traces).
  const vizRootEls = [
    playBtn, progressCanvas,
    ...panels.flatMap(({ containers }) => Object.values(containers)),
    ...(sharedTraces || []).map(({ container }) => container),
  ].filter(Boolean);

  const docMousedownFocusHandler = e => {
    vizFocused = vizRootEls.some(el => el === e.target || el.contains(e.target))
      || !!(maximizedInfo?.overlayEl && (maximizedInfo.overlayEl === e.target || maximizedInfo.overlayEl.contains(e.target)));
  };
  document.addEventListener('mousedown', docMousedownFocusHandler, { capture: true });

  // Compute total leapfrog steps across the first panel's full order
  function totalLeapfrogSteps() {
    const p = currentPanels[0];
    if (!p) return 60;
    const ord = p.data.order || p.data.trajectories.map((_, i) => i);
    return ord.reduce((sum, k) => {
      const traj = p.data.trajectories[k];
      if (!traj) return sum;
      const ro = p.data.trajectories_order?.[k];
      return sum + Math.max(1, (ro || traj).length - 1);
    }, 0);
  }

  // Default: finish in ~60 seconds
  let speed = Math.max(0.5, totalLeapfrogSteps() / 60);

  function stepsPerSec() { return speed; }

  function updateSpeedDisplay() {
    if (!speedDisplay) return;
    const s = speed;
    speedDisplay.textContent = s < 1   ? `${s.toFixed(2)}/s`
                             : s < 10  ? `${s.toFixed(1)}/s`
                             :           `${Math.round(s)}/s`;
  }

  function speedUp()   { speed = speed * 2; updateSpeedDisplay(); }
  function speedDown() { speed = speed / 2; updateSpeedDisplay(); }

  updateSpeedDisplay();

  function orderLen(data) {
    return (data.order || data.trajectories).length;
  }

  // Cap at the shortest order so no panel runs past its data
  function maxStep() {
    return Math.min(...currentPanels.map(p => orderLen(p.data))) - 0.0001;
  }

  function updateDisplay() {
    if (stepDisplay) {
      const traj  = Math.floor(currentStep);
      const total = Math.min(...currentPanels.map(p => orderLen(p.data)));
      stepDisplay.textContent = `traj ${traj} / ${total}`;
    }
  }

  // Cumulative leapfrog-step fractions: cum[i] = fraction of total steps at trajectory i's start.
  // Length = nTrajs + 1, with cum[0]=0 and cum[nTrajs]=1.
  function leapfrogCumFractions() {
    const p = currentPanels[0];
    if (!p) return null;
    const ord = p.data.order || p.data.trajectories.map((_, i) => i);
    const lens = ord.map(k => {
      const traj = p.data.trajectories[k];
      const ro   = p.data.trajectories_order?.[k];
      return traj ? Math.max(1, (ro || traj).length - 1) : 1;
    });
    const total = lens.reduce((a, b) => a + b, 0);
    if (total === 0) return null;
    const cum = [0];
    let s = 0;
    for (const l of lens) { s += l; cum.push(s / total); }
    return cum;
  }

  // Map a [0,1] leapfrog-fraction back to currentStep
  function fracToStep(frac, cum) {
    let i = 0;
    while (i < cum.length - 2 && cum[i + 1] <= frac) i++;
    const segLen = cum[i + 1] - cum[i];
    const segFrac = segLen > 0 ? (frac - cum[i]) / segLen : 0;
    return i + Math.max(0, Math.min(1, segFrac));
  }

  function renderProgressBar() {
    if (!progressCanvas) return;
    const W = progressCanvas.clientWidth || 300;
    const H = progressCanvas.clientHeight || 20;
    if (progressCanvas.width !== W) progressCanvas.width = W;
    if (progressCanvas.height !== H) progressCanvas.height = H;
    const ctx = progressCanvas.getContext('2d');
    ctx.clearRect(0, 0, W, H);

    const cum = leapfrogCumFractions();
    if (!cum) return;

    const doneTraj  = Math.min(Math.floor(currentStep), cum.length - 2);
    const fraction  = currentStep - doneTraj;
    const fillFrac  = cum[doneTraj] + fraction * (cum[doneTraj + 1] - cum[doneTraj]);
    const cy        = H / 2;

    // Track — white background
    ctx.fillStyle = '#e8e8e8';
    ctx.fillRect(0, cy - 2, W, 4);

    // Fill — dark for completed portion
    if (fillFrac > 0) {
      ctx.fillStyle = '#555';
      ctx.fillRect(0, cy - 2, fillFrac * W, 4);
    }

    // Trajectory ticks — white (visible against dark fill)
    ctx.fillStyle = 'rgba(255,255,255,0.7)';
    for (let i = 1; i < cum.length - 1; i++) {
      const x = cum[i] * W;
      if (x > 1 && x < W - 1) ctx.fillRect(x - 0.5, cy - 4, 1, 8);
    }

    // Transition ticks — amber, full bar height, clearly distinct from trajectory ticks
    if (transitions.length > 0) {
      ctx.fillStyle = YELLOW;
      for (const t of transitions) {
        const ti = Math.min(Math.floor(t.step), cum.length - 2);
        const x  = cum[ti] * W;
        if (x > 1 && x < W - 1) ctx.fillRect(x - 1, 0, 2, H);
      }
    }

    // Knob
    const kx = fillFrac * W;
    ctx.beginPath();
    ctx.arc(kx, cy, 6, 0, Math.PI * 2);
    ctx.fillStyle = '#444';
    ctx.fill();
    ctx.beginPath();
    ctx.arc(kx, cy, 3, 0, Math.PI * 2);
    ctx.fillStyle = '#fff';
    ctx.fill();
  }

  function draw() {
    // Find the highest-threshold transition crossed so far (supports backward scrubbing).
    let activeTr = null;
    for (const t of transitions) {
      if (currentStep >= t.step) activeTr = t;
    }
    currentPanels.forEach(({ containers, data }, i) => {
      const panelTd = activeTr ? activeTr.panelTransitions[i] : null;
      renderMCMC(containers, data, currentStep, {
        zoomToFit:  zoomStates[i],
        transition: panelTd ? { data: panelTd, startStep: activeTr.step } : null,
      });
    });

    // Maximized subplot overlay — live re-render of the clicked subplot.
    if (maximizedInfo) {
      const { type, overlayEl } = maximizedInfo;
      if (type === 'shared') {
        const { traceIdx } = maximizedInfo;
        if (sharedTraces?.[traceIdx]) {
          _renderSharedTrace(overlayEl, sharedTraces[traceIdx].spec, Math.floor(currentStep) + 1);
        }
      } else {
        const { panelIdx, containerKey } = maximizedInfo;
        const p = currentPanels[panelIdx];
        if (p) {
          const panelTd = activeTr ? activeTr.panelTransitions[panelIdx] : null;
          const fakeContainers = {
            viz:    containerKey === 'viz'    ? overlayEl : null,
            histX:  containerKey === 'histX'  ? overlayEl : null,
            histY:  containerKey === 'histY'  ? overlayEl : null,
            traceX: containerKey === 'traceX' ? overlayEl : null,
            traceY: containerKey === 'traceY' ? overlayEl : null,
          };
          renderMCMC(fakeContainers, p.data, currentStep, {
            zoomToFit: zoomStates[panelIdx],
            transition: panelTd ? { data: panelTd, startStep: activeTr.step } : null,
          });
        }
      }
    }

    if (sharedTraces) {
      const nSteps = Math.floor(currentStep) + 1;
      sharedTraces.forEach(({ container, spec }) =>
        _renderSharedTrace(container, spec, nSteps)
      );
    }
    renderProgressBar();
    updateDisplay();
  }

  function tick(timestamp) {
    if (!playing) return;
    if (lastTime === null) lastTime = timestamp;
    const dt = Math.min((timestamp - lastTime) / 1000, 0.1);  // cap at 100ms to avoid jumps
    lastTime = timestamp;

    // Advance in leapfrog-fraction space so the bar progresses at constant speed
    // regardless of per-trajectory leapfrog counts.
    const cum  = leapfrogCumFractions();
    const total = totalLeapfrogSteps();
    if (cum && total > 0) {
      const doneTraj = Math.min(Math.floor(currentStep), cum.length - 2);
      const fraction = currentStep - doneTraj;
      const curFrac  = cum[doneTraj] + fraction * (cum[doneTraj + 1] - cum[doneTraj]);
      const newFrac  = Math.min(curFrac + dt * stepsPerSec() / total, 1);
      currentStep    = newFrac >= 1 ? maxStep() : fracToStep(newFrac, cum);
    } else {
      currentStep = Math.min(currentStep + dt * stepsPerSec(), maxStep());
    }
    draw();

    if (currentStep >= maxStep()) { pause(); return; }
    rafId = requestAnimationFrame(tick);
  }

  function play() {
    if (playing) return;
    if (currentStep >= maxStep()) currentStep = 0;
    playing  = true;
    lastTime = null;
    playBtn.textContent = '⏸';
    rafId = requestAnimationFrame(tick);
  }

  function pause() {
    playing = false;
    if (rafId) cancelAnimationFrame(rafId);
    rafId    = null;
    lastTime = null;
    playBtn.textContent = '▶';
  }

  function reset()    { pause(); currentStep = 0; draw(); }

  function stepFwd()  {
    pause();
    currentStep = Math.min(Math.floor(currentStep) + 1, Math.floor(maxStep()));
    draw();
  }

  function stepBack() {
    pause();
    currentStep = Math.max(Math.ceil(currentStep - 1), 0);
    draw();
  }

  // Named handlers so they can be removed by destroy()
  let mousemoveHandler = null, mouseupHandler = null, mousedownHandler = null;

  const keydownHandler = e => {
    if (!vizFocused) return;
    if (e.key === 'Escape') { e.preventDefault(); clearMaximized(); return; }
    if (e.key === 'ArrowRight') {
      e.preventDefault();
      currentStep = Math.min(Math.floor(currentStep) + 1, Math.floor(maxStep()));
      if (!playing) draw();
    }
    if (e.key === 'ArrowLeft') {
      e.preventDefault();
      currentStep = Math.max(Math.ceil(currentStep - 1), 0);
      if (!playing) draw();
    }
    if (e.key === 'ArrowUp')   { e.preventDefault(); speedUp(); }
    if (e.key === 'ArrowDown') { e.preventDefault(); speedDown(); }
    if (e.key === ' ')         { e.preventDefault(); playing ? pause() : play(); }
  };
  document.addEventListener('keydown', keydownHandler);

  // Draggable / clickable progress bar
  if (progressCanvas) {
    let dragging = false;
    let wasPlaying = false;

    function scrubToX(clientX) {
      const rect = progressCanvas.getBoundingClientRect();
      const frac = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
      const cum = leapfrogCumFractions();
      currentStep = cum ? fracToStep(frac, cum) : frac * (maxStep() + 0.0001);
      draw();
    }

    mousedownHandler = e => { dragging = true; wasPlaying = playing; pause(); scrubToX(e.clientX); };
    mousemoveHandler = e => { if (dragging) scrubToX(e.clientX); };
    mouseupHandler   = () => { if (dragging) { dragging = false; if (wasPlaying) play(); } };

    progressCanvas.addEventListener('mousedown', mousedownHandler);
    document.addEventListener('mousemove', mousemoveHandler);
    document.addEventListener('mouseup',   mouseupHandler);
  }

  // load accepts an array of { containers, data } matching the original panels,
  // or an array of just data objects (containers reused from original panels).
  function load(newPanels) {
    pause();
    if (newPanels[0] && newPanels[0].containers) {
      currentPanels = newPanels;
    } else {
      // Convenience: just new data objects, reuse existing containers
      currentPanels = currentPanels.map((p, i) => ({
        containers: p.containers,
        data: newPanels[i] ?? p.data,
      }));
    }
    currentStep = 0;
    draw();
  }

  // ── Zoom-to-fit API ──────────────────────────────────────────────────────────
  function getZoom(i)    { return zoomStates[i] ?? false; }
  function setZoom(i, v) { zoomStates[i] = !!v; if (!playing) draw(); }
  function toggleZoom(i) { setZoom(i, !getZoom(i)); }

  // ── Transitions API ──────────────────────────────────────────────────────────
  // addTransition(step, transitionDatasets)
  //   step:               trajectory index at which the transition fires
  //   transitionDatasets: array (one per panel) of lightweight snapshot objects:
  //     { initialSamples: [[x,y],...],  // pre-computed chain history before this step
  //       pdfs?,                         // replacement heatmap (uses base data if omitted)
  //       xGrid?, yGrid?,               // replacement grid  (uses base data if omitted)
  //       xPdf?,  yPdf? }               // replacement marginal KDEs (uses base if omitted)
  function addTransition(step, transitionDatasets) {
    transitions.push({ step, panelTransitions: transitionDatasets });
    transitions.sort((a, b) => a.step - b.step);
  }

  function clearTransitions() {
    transitions = [];
  }

  const playClickHandler = () => playing ? pause() : play();
  playBtn.addEventListener('click', playClickHandler);

  function destroy() {
    clearMaximized();
    pause();
    document.removeEventListener('keydown',    keydownHandler);
    document.removeEventListener('mousemove',  mousemoveHandler);
    document.removeEventListener('mouseup',    mouseupHandler);
    document.removeEventListener('mousedown',  docMousedownFocusHandler, { capture: true });
    if (progressCanvas && mousedownHandler)
      progressCanvas.removeEventListener('mousedown', mousedownHandler);
    playBtn.removeEventListener('click', playClickHandler);
  }

  draw();
  return {
    play, pause, reset, load, stepFwd, stepBack, speedUp, speedDown,
    getZoom, setZoom, toggleZoom,
    addTransition, clearTransitions,
    setMaximized, setMaximizedShared, clearMaximized,
    focus, blur,
    destroy,
  };
}


// =============================================================================
// setup_viz — public entry point
//
// Call this once after mcmc-viz.js is loaded (the DOM must already exist):
//
//   const player = setup_viz(datasets)
//
// datasets: Array of data objects, one per panel (-1, -2, …).
//           Each object has the shape described at the top of this file.
//
// Returns the player { play, pause, reset, load }.
// To swap in new data later call player.load(newDatasets).
// =============================================================================
function setup_viz(datasets, sharedTraceSpecs, options = {}) {
  // Tear down any previously running player (removes all event listeners).
  if (window.mcmcPlayer) { window.mcmcPlayer.destroy(); window.mcmcPlayer = null; }

  // Clear shared-trace panels so they don't accumulate across reloads.
  const sharedHost = document.getElementById('shared-traces');
  if (sharedHost) sharedHost.innerHTML = '';

  // Remove any maximize overlays left by the previous player.
  document.querySelectorAll('.subplot-overlay').forEach(o => o.remove());

  function getContainers(suffix) {
    return {
      viz:    document.getElementById(`viz${suffix}`),
      histX:  document.getElementById(`hist-x${suffix}`),
      histY:  document.getElementById(`hist-y${suffix}`),
      traceX: document.getElementById(`trace-x${suffix}`),
      traceY: document.getElementById(`trace-y${suffix}`),
    };
  }

  // Allow trajectories_order to be specified once here and shared across all panels
  const sharedOrder = options.trajectories_order ?? null;

  const panels = datasets.map((data, i) => ({
    containers: getContainers(`-${i + 1}`),
    data: sharedOrder && !data.trajectories_order ? { ...data, trajectories_order: sharedOrder } : data,
  }));

  // Build shared trace containers if specs provided
  let sharedTraces = null;
  if (sharedTraceSpecs && sharedTraceSpecs.length > 0) {
    const host = document.getElementById('shared-traces');
    if (host) {
      sharedTraces = sharedTraceSpecs.map(spec => {
        const div = document.createElement('div');
        div.className = 'panel shared-trace';
        host.appendChild(div);
        return { container: div, spec };
      });
    }
  }

  const player = initMCMCPlayer(panels, {
    playBtn:        document.getElementById('btn-play'),
    stepDisplay:    document.getElementById('step-display'),
    progressCanvas: document.getElementById('progress-bar'),
    speedDisplay:   document.getElementById('speed-display'),
  }, sharedTraces);

  const slowBtn = document.getElementById('btn-slow');
  if (slowBtn) slowBtn.addEventListener('click', () => player.speedDown());

  const fastBtn = document.getElementById('btn-fast');
  if (fastBtn) fastBtn.addEventListener('click', () => player.speedUp());

  // Per-panel zoom toggle button (floating overlay inside viz container)
  panels.forEach(({ containers }, i) => {
    const vizEl = containers.viz;
    if (!vizEl) return;
    // Remove any button left by a previous setup_viz call
    vizEl.querySelectorAll('.viz-zoom-btn').forEach(b => b.remove());
    vizEl.style.position = 'relative';
    const btn = document.createElement('button');
    btn.className   = 'viz-zoom-btn';
    btn.textContent = 'zoom';
    btn.title       = 'Toggle zoom to fit samples';
    btn.style.cssText = [
      'position:absolute', 'top:4px', 'right:4px',
      'padding:1px 5px', 'font-size:0.5rem', 'line-height:1.6',
      'min-width:auto', 'z-index:10', 'opacity:0.4',
    ].join(';');
    btn.addEventListener('click', () => {
      player.toggleZoom(i);
      btn.style.opacity = player.getZoom(i) ? '1' : '0.4';
    });
    vizEl.appendChild(btn);
    // Apply initial zoom if requested
    const initZoom = Array.isArray(options.zoomToFit) ? options.zoomToFit[i] : !!options.zoomToFit;
    if (initZoom) { player.setZoom(i, true); btn.style.opacity = '1'; }
  });

  // Subplot maximize: clicking any panel container opens a live overlay covering .panels.
  const ALL_CONTAINER_KEYS = ['viz', 'histX', 'histY', 'traceX', 'traceY'];
  panels.forEach(({ containers }, panelIdx) => {
    ALL_CONTAINER_KEYS.forEach(key => {
      const el = containers[key];
      if (!el) return;
      el.style.cursor = 'zoom-in';
      el.addEventListener('click', e => {
        if (e.target.closest && e.target.closest('.viz-zoom-btn')) return;
        const panelsEl = el.closest('.panels') || el.parentElement?.parentElement;
        if (!panelsEl) return;
        panelsEl.style.position = 'relative';
        // Remove any lingering overlay from a previous click
        panelsEl.querySelectorAll('.subplot-overlay').forEach(o => o.remove());
        const overlay = document.createElement('div');
        overlay.className = 'subplot-overlay';
        overlay.style.cssText = 'position:absolute;inset:0;z-index:10;background:#fff;cursor:zoom-out;';
        // Small close button
        const closeBtn = document.createElement('button');
        closeBtn.textContent = '✕';
        closeBtn.title = 'Close (Esc)';
        closeBtn.style.cssText = [
          'position:absolute', 'top:6px', 'right:6px', 'z-index:11',
          'padding:2px 7px', 'font-size:0.6rem', 'opacity:0.5', 'min-width:auto',
        ].join(';');
        closeBtn.addEventListener('click', ev => { ev.stopPropagation(); player.clearMaximized(); });
        overlay.appendChild(closeBtn);
        panelsEl.appendChild(overlay);
        player.setMaximized(panelIdx, key, overlay);
      });
    });
  });

  // Shared trace maximize: clicking a shared trace opens a live overlay covering #shared-traces.
  if (sharedTraces) {
    sharedTraces.forEach(({ container }, traceIdx) => {
      container.style.cursor = 'zoom-in';
      container.addEventListener('click', () => {
        const hostEl = document.getElementById('shared-traces') || container.parentElement;
        if (!hostEl) return;
        hostEl.style.position = 'relative';
        hostEl.querySelectorAll('.subplot-overlay').forEach(o => o.remove());
        const overlay = document.createElement('div');
        overlay.className = 'subplot-overlay';
        overlay.style.cssText = 'position:absolute;inset:0;z-index:10;background:#fff;cursor:zoom-out;';
        const closeBtn = document.createElement('button');
        closeBtn.textContent = '✕';
        closeBtn.title = 'Close (Esc)';
        closeBtn.style.cssText = [
          'position:absolute', 'top:4px', 'right:6px', 'z-index:11',
          'padding:2px 7px', 'font-size:0.6rem', 'opacity:0.5', 'min-width:auto',
        ].join(';');
        closeBtn.addEventListener('click', ev => { ev.stopPropagation(); player.clearMaximized(); });
        overlay.appendChild(closeBtn);
        hostEl.appendChild(overlay);
        player.setMaximizedShared(traceIdx, overlay);
      });
    });
  }

  // Register any transitions supplied at construction time.
  // options.transitions: Array of { step, datasets } — same shape as addTransition().
  if (options.transitions) {
    for (const { step, datasets } of options.transitions) {
      player.addTransition(step, datasets);
    }
  }

  // Store globally so re-calling setup_viz can destroy the old player cleanly,
  // and so external code (reveal.js hooks, htmx handlers, etc.) can reach it.
  window.mcmcPlayer = player;
  return player;
}