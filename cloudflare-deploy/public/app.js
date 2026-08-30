// SRT Studio — single-file frontend
// All processing happens in the browser. No backend, no storage.

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

// ============================================================================
// State
// ============================================================================
const state = {
  video: null,        // { file: File, url: string, name: string, duration: number }
  srt: null,          // { cues: [{ id, start, end, text }], fileName: string }
  currentTime: 0,
  activeCueId: null,
  selectedCueId: null,
  style: {
    fontFamily: 'Arial',
    fontSize: 20,
    fontWeight: 'bold',
    fontStyle: 'normal',
    primaryColor: '#ffffff',
    outlineColor: '#000000',
    outlineWidth: 2,
    backColor: '#000000',
    backOpacity: 50,
    shadow: false,
    alignment: 2, // ASS alignment (1-9). 2 = bottom-center
    marginV: 40,
    marginH: 40,
  },
  // Per-cue position override (for drag-to-reposition). null = use style alignment.
  cueOverrides: {}, // { cueId: { x: '50%', y: '50%' } }
  history: [],
  historyIndex: -1,
  rendering: false,
  translating: false,
};

const HISTORY_LIMIT = 50;

// ============================================================================
// Utils
// ============================================================================
function uid() { return Math.random().toString(36).slice(2, 9); }
function clamp(n, lo, hi) { return Math.max(lo, Math.min(hi, n)); }
function fmtTime(sec, sep = ',') {
  if (!isFinite(sec)) return '00:00:00' + sep + '000';
  const ms = Math.floor((sec % 1) * 1000);
  const s = Math.floor(sec) % 60;
  const m = Math.floor(sec / 60) % 60;
  const h = Math.floor(sec / 3600);
  return (
    String(h).padStart(2, '0') + ':' +
    String(m).padStart(2, '0') + ':' +
    String(s).padStart(2, '0') + sep +
    String(ms).padStart(3, '0')
  );
}
function parseTime(str) {
  // accepts 00:00:00,000 or 00:00:00.000
  const m = String(str).trim().match(/^(\d+):(\d{1,2}):(\d{1,2})[,.](\d{1,3})$/);
  if (!m) return null;
  return (+m[1]) * 3600 + (+m[2]) * 60 + (+m[3]) + (+m[4]) / 1000;
}
function debounce(fn, ms = 200) {
  let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
}
function toast(msg, kind = '') {
  const el = $('#toast');
  el.textContent = msg;
  el.className = 'toast ' + kind;
  el.hidden = false;
  clearTimeout(el._t);
  el._t = setTimeout(() => { el.hidden = true; }, 3000);
}

// ============================================================================
// SRT parsing / generation
// ============================================================================
function parseSRT(text) {
  const out = [];
  const blocks = text.replace(/\r\n/g, '\n').replace(/\r/g, '\n').trim().split(/\n\s*\n/);
  for (const block of blocks) {
    const lines = block.split('\n').map(l => l.trim()).filter(Boolean);
    if (lines.length < 2) continue;
    let i = 0;
    let seq = i + 1;
    // optional sequence number on first line
    if (/^\d+$/.test(lines[0])) { seq = +lines[0]; i = 1; }
    const tm = lines[i] && lines[i].match(/(\S+)\s*-->\s*(\S+)/);
    if (!tm) continue;
    const start = parseTime(tm[1]);
    const end = parseTime(tm[2]);
    if (start == null || end == null) continue;
    const textLines = lines.slice(i + 1);
    out.push({ id: uid(), seq, start, end, text: textLines.join('\n') });
  }
  out.sort((a, b) => a.start - b.start);
  out.forEach((c, i) => c.seq = i + 1);
  return out;
}

function generateSRT(cues) {
  return cues.map((c, i) =>
    `${i + 1}\n${fmtTime(c.start)} --> ${fmtTime(c.end)}\n${c.text}`
  ).join('\n\n') + '\n';
}

// ============================================================================
// History (undo/redo)
// ============================================================================
function pushHistory() {
  const snap = JSON.parse(JSON.stringify({
    cues: state.srt ? state.srt.cues : [],
    overrides: state.cueOverrides,
  }));
  // truncate forward history
  state.history = state.history.slice(0, state.historyIndex + 1);
  state.history.push(snap);
  if (state.history.length > HISTORY_LIMIT) state.history.shift();
  state.historyIndex = state.history.length - 1;
  updateSaveStatus();
}
function undo() {
  if (state.historyIndex <= 0) return;
  state.historyIndex--;
  restoreHistory();
}
function redo() {
  if (state.historyIndex >= state.history.length - 1) return;
  state.historyIndex++;
  restoreHistory();
}
function restoreHistory() {
  if (!state.srt) return;
  const snap = state.history[state.historyIndex];
  if (!snap) return;
  state.srt.cues = JSON.parse(JSON.stringify(snap.cues || []));
  state.cueOverrides = JSON.parse(JSON.stringify(snap.overrides || {}));
  renderCueList();
  renderOverlay();
  updateSaveStatus();
}
function updateSaveStatus() {
  const el = $('#saveStatus');
  if (!el) return;
  const dirty = state.historyIndex < state.history.length - 1 || state.historyIndex < 0;
  el.textContent = (state.historyIndex >= 0 && state.historyIndex === state.history.length - 1) ? 'saved' : 'unsaved';
}

// ============================================================================
// Video loading
// ============================================================================
function loadVideoFile(file) {
  if (!file) return;
  if (!file.type.startsWith('video/')) {
    toast('Please drop a video file', 'error');
    return;
  }
  if (state.video && state.video.url) URL.revokeObjectURL(state.video.url);
  const url = URL.createObjectURL(file);
  const v = $('#video');
  v.src = url;
  v.load();
  state.video = { file, url, name: file.name, duration: 0 };
  $('#dropzone').classList.add('hidden');
  v.onloadedmetadata = () => {
    state.video.duration = v.duration;
    const aspect = v.videoWidth / v.videoHeight;
    state.video.aspect = aspect;
    $('#totalTime').textContent = fmtTime(v.duration).replace(',', '.');
    // Set the stage's aspect-ratio to match the video. The stage
    // will then size itself to the largest box with that ratio that
    // fits in the wrap. The video element fills the stage (100% ×
    // 100%) and the overlay (inset:0 of stage) covers exactly the
    // video frame — no letterbox mismatch.
    const stage = $('#videoStage');
    stage.style.aspectRatio = aspect.toString();
    // Reset any prior pixel sizing
    stage.style.width = '';
    stage.style.height = '';
    stage.style.left = '';
    stage.style.top = '';
  };
  toast('Video loaded: ' + file.name, 'success');
}

// Kept as a no-op for backward compatibility with renderOverlay's call
function alignOverlayToVideo() { /* no-op: stage is sized via aspect-ratio CSS */ }

// ============================================================================
// SRT loading
// ============================================================================
function loadSRTFromText(text, fileName = 'pasted.srt') {
  const cues = parseSRT(text);
  if (cues.length === 0) {
    toast('No valid SRT cues found', 'error');
    return;
  }
  state.srt = { cues, fileName };
  state.cueOverrides = {}; // reset positions for a new SRT
  state.history = [];
  state.historyIndex = -1;
  pushHistory();
  renderCueList();
  renderOverlay();
  toast(`Loaded ${cues.length} cues`, 'success');
}

function loadSRTFile(file) {
  if (!file) return;
  const reader = new FileReader();
  reader.onload = e => loadSRTFromText(e.target.result, file.name);
  reader.readAsText(file);
}

// ============================================================================
// Cue list rendering
// ============================================================================
function renderCueList() {
  const list = $('#cueList');
  list.innerHTML = '';
  if (!state.srt) {
    list.innerHTML = '<div class="muted" style="padding: 20px; text-align: center;">No subtitles loaded yet.<br>Open an .srt file, paste, or add a cue.</div>';
    $('#cueCount').textContent = '0 cues';
    return;
  }
  const search = ($('#searchBox').value || '').toLowerCase();
  state.srt.cues.forEach((c, i) => {
    if (search && !c.text.toLowerCase().includes(search)) return;
    const card = document.createElement('div');
    card.className = 'cue-card' + (c.id === state.selectedCueId ? ' active' : '');
    card.dataset.id = c.id;
    card.innerHTML = `
      <div class="row">
        <span class="seq">${i + 1}</span>
        <input type="text" class="time-input start" value="${fmtTime(c.start)}" />
        <span class="arrow">→</span>
        <input type="text" class="time-input end" value="${fmtTime(c.end)}" />
        <span style="flex:1"></span>
        <button class="del" title="Delete cue">×</button>
      </div>
      <textarea class="text-input" rows="2">${escapeHtml(c.text)}</textarea>
    `;
    // time edits
    card.querySelector('.start').addEventListener('change', e => {
      const v = parseTime(e.target.value);
      if (v == null) { e.target.value = fmtTime(c.start); return; }
      c.start = v;
      if (c.end < v) c.end = v + 1;
      pushHistory();
      renderOverlay();
      $('#nowEditing').hidden = false;
      $('#nowEditingText').textContent = `cue ${i + 1} start`;
    });
    card.querySelector('.end').addEventListener('change', e => {
      const v = parseTime(e.target.value);
      if (v == null) { e.target.value = fmtTime(c.end); return; }
      c.end = Math.max(v, c.start + 0.1);
      pushHistory();
      renderOverlay();
      $('#nowEditing').hidden = false;
      $('#nowEditingText').textContent = `cue ${i + 1} end`;
    });
    // text edits
    const ta = card.querySelector('.text-input');
    ta.addEventListener('focus', () => { state.selectedCueId = c.id; $('#nowEditing').hidden = false; $('#nowEditingText').textContent = `cue ${i + 1}`; });
    ta.addEventListener('input', e => {
      c.text = e.target.value;
      pushHistory();
      renderOverlay();
    });
    // delete
    card.querySelector('.del').addEventListener('click', e => {
      e.stopPropagation();
      state.srt.cues = state.srt.cues.filter(x => x.id !== c.id);
      pushHistory();
      renderCueList();
      renderOverlay();
    });
    // click → seek to cue
    card.addEventListener('click', e => {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'BUTTON') return;
      state.selectedCueId = c.id;
      $('#video').currentTime = c.start;
      renderCueList();
    });
    list.appendChild(card);
  });
  $('#cueCount').textContent = `${state.srt.cues.length} cue${state.srt.cues.length === 1 ? '' : 's'}`;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
}

// ============================================================================
// Live overlay rendering
// ============================================================================
function renderOverlay() {
  const overlay = $('#overlay');
  overlay.innerHTML = '';
  if (!state.srt) return;
  // Read currentTime from the <video> element, NOT from state.video
  // (state.video doesn't store currentTime — it's a property of the
  // HTMLMediaElement, not our state object). state.currentTime is kept
  // up to date by the video event handlers + interval loop.
  const t = (state.currentTime != null) ? state.currentTime : (video.currentTime || 0);
  const s = state.style;
  const activeCue = state.srt.cues.find(c => t >= c.start && t <= c.end + 0.05);
  state.activeCueId = activeCue ? activeCue.id : null;
  if (!activeCue) return;
  const cue = document.createElement('div');
  cue.className = 'cue';
  cue.dataset.cueId = activeCue.id;
  cue.textContent = activeCue.text;

  // Position: per-cue override takes priority, else style alignment
  const override = state.cueOverrides[activeCue.id];
  if (override) {
    // Free-form position from drag (in % of stage)
    cue.style.left = override.x;
    cue.style.top = override.y;
    cue.style.right = 'auto';
    cue.style.bottom = 'auto';
    cue.style.transform = 'translate(-50%, -50%)';
  } else {
    // ASS alignment numeric:
    //   1=bottom-left  2=bottom-center  3=bottom-right
    //   4=middle-left  5=middle-center  6=middle-right
    //   7=top-left     8=top-center     9=top-right
    const a = s.alignment;
    const vert = a >= 7 ? 'top' : a >= 4 ? 'middle' : 'bottom';
    const horiz = (a === 1 || a === 4 || a === 7) ? 'left' :
                  (a === 3 || a === 6 || a === 9) ? 'right' : 'center';
    cue.style.top = vert === 'top' ? s.marginV + 'px' :
                    vert === 'bottom' ? `calc(100% - ${s.marginV}px)` : '50%';
    cue.style.bottom = '';
    cue.style.left = horiz === 'left' ? s.marginH + 'px' :
                     horiz === 'right' ? `calc(100% - ${s.marginH}px)` : '50%';
    cue.style.right = '';
    cue.style.transform = horiz === 'center' && vert === 'middle' ? 'translate(-50%, -50%)' :
                         horiz === 'center' ? 'translateX(-50%)' :
                         vert === 'middle' ? 'translateY(-50%)' : 'none';
  }

  // Style
  cue.style.fontFamily = s.fontFamily;
  cue.style.fontSize = s.fontSize + 'px';
  cue.style.fontWeight = s.fontWeight;
  cue.style.fontStyle = s.fontStyle;
  cue.style.color = s.primaryColor;
  cue.style.lineHeight = '1.3';
  const bg = hexToRgb(s.backColor);
  cue.style.backgroundColor = `rgba(${bg.r},${bg.g},${bg.b},${s.backOpacity / 100})`;
  cue.style.padding = '4px 10px';
  cue.style.borderRadius = '3px';
  const oc = hexToRgb(s.outlineColor);
  if (s.shadow) {
    cue.style.textShadow = `${s.outlineWidth}px ${s.outlineWidth}px 0 rgba(${oc.r},${oc.g},${oc.b},0.9),
                            -${s.outlineWidth}px ${s.outlineWidth}px 0 rgba(${oc.r},${oc.g},${oc.b},0.9),
                            ${s.outlineWidth}px -${s.outlineWidth}px 0 rgba(${oc.r},${oc.g},${oc.b},0.9),
                            -${s.outlineWidth}px -${s.outlineWidth}px 0 rgba(${oc.r},${oc.g},${oc.b},0.9),
                            2px 2px 4px rgba(0,0,0,0.5)`;
  } else if (s.outlineWidth > 0) {
    const ow = s.outlineWidth;
    cue.style.textShadow = `${ow}px ${ow}px 0 rgba(${oc.r},${oc.g},${oc.b},0.9),
                            -${ow}px ${ow}px 0 rgba(${oc.r},${oc.g},${oc.b},0.9),
                            ${ow}px -${ow}px 0 rgba(${oc.r},${oc.g},${oc.b},0.9),
                            -${ow}px -${ow}px 0 rgba(${oc.r},${oc.g},${oc.b},0.9)`;
  }

  // ===== Drag-to-reposition with Shift-snap (Photoshop-style) =====
  // The overlay is sized to match the video's actual rendered rect
  // (via alignOverlayToVideo), so drag math uses the overlay rect
  // (which is already captured in `overlay` at the top of this function).
  const snapH = $('#snapH');
  const snapV = $('#snapV');
  let drag = null;

  cue.addEventListener('pointerdown', e => {
    e.preventDefault();
    e.stopPropagation();
    try { cue.setPointerCapture(e.pointerId); } catch (_) {}
    const overlayRect = overlay.getBoundingClientRect();
    const cueRect = cue.getBoundingClientRect();
    // Current cue center in % of overlay (= video frame)
    const centerX = ((cueRect.left + cueRect.width / 2) - overlayRect.left) / overlayRect.width * 100;
    const centerY = ((cueRect.top + cueRect.height / 2) - overlayRect.top) / overlayRect.height * 100;
    drag = {
      startX: e.clientX, startY: e.clientY,
      centerX, centerY,
      overlayRect,
    };
    cue.classList.add('dragging');
  });

  cue.addEventListener('pointermove', e => {
    if (!drag) return;
    const rect = drag.overlayRect;
    const dx = (e.clientX - drag.startX) / rect.width * 100;
    const dy = (e.clientY - drag.startY) / rect.height * 100;
    let newX = drag.centerX + dx;
    let newY = drag.centerY + dy;

    // Shift = snap to center / thirds / edges (Photoshop-style)
    if (e.shiftKey) {
      const snap = 6; // 6% snap radius
      const targetsX = [0, 25, 33.33, 50, 66.66, 75, 100];
      const targetsY = [0, 25, 33.33, 50, 66.66, 75, 100];
      let bestX = 50, bestDx = Infinity;
      let bestY = 50, bestDy = Infinity;
      for (const t of targetsX) { const d = Math.abs(newX - t); if (d < bestDx) { bestDx = d; bestX = t; } }
      for (const t of targetsY) { const d = Math.abs(newY - t); if (d < bestDy) { bestDy = d; bestY = t; } }
      let snapped = false;
      if (bestDx < snap) { newX = bestX; snapped = true; }
      if (bestDy < snap) { newY = bestY; snapped = true; }
      if (snapped) {
        snapH.style.top = bestY + '%';
        snapH.hidden = false;
        snapV.style.left = bestX + '%';
        snapV.hidden = false;
      } else {
        snapH.hidden = true;
        snapV.hidden = true;
      }
    } else {
      snapH.hidden = true;
      snapV.hidden = true;
    }

    // Clamp so the cue stays inside the stage
    newX = clamp(newX, 5, 95);
    newY = clamp(newY, 5, 95);

    cue.style.left = newX + '%';
    cue.style.top = newY + '%';
    cue.style.right = 'auto';
    cue.style.bottom = 'auto';
    cue.style.transform = 'translate(-50%, -50%)';
  });

  cue.addEventListener('pointerup', e => {
    if (!drag) return;
    try { cue.releasePointerCapture(e.pointerId); } catch (_) {}
    snapH.hidden = true;
    snapV.hidden = true;
    cue.classList.remove('dragging');
    state.cueOverrides[activeCue.id] = {
      x: cue.style.left,
      y: cue.style.top,
    };
    pushHistory();
    drag = null;
  });

  cue.addEventListener('pointercancel', () => {
    if (!drag) return;
    snapH.hidden = true;
    snapV.hidden = true;
    cue.classList.remove('dragging');
    drag = null;
  });

  // Double-click resets to the style alignment
  cue.addEventListener('dblclick', e => {
    e.preventDefault();
    delete state.cueOverrides[activeCue.id];
    pushHistory();
    renderOverlay();
    toast('Position reset to style alignment');
  });

  overlay.appendChild(cue);
  // also highlight active in list
  $$('.cue-card').forEach(el => {
    if (el.dataset.id === activeCue.id) el.classList.add('active');
    else el.classList.remove('active');
  });
}

function hexToRgb(hex) {
  const m = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  if (!m) return { r: 0, g: 0, b: 0 };
  return { r: parseInt(m[1], 16), g: parseInt(m[2], 16), b: parseInt(m[3], 16) };
}

// ============================================================================
// Visual timeline (cue positions + playhead)
// ============================================================================
function renderTimeline() {
  const track = $('#timelineTrack');
  if (!track) return;
  const playhead = $('#timelinePlayhead');
  // Re-render cues only when the cue list changes (cheap optimization):
  // we re-render if the rendered count differs or the active state changed.
  const wantCues = state.srt ? state.srt.cues : [];
  const haveNodes = track.querySelectorAll('.timeline-cue').length;
  if (haveNodes !== wantCues.length) {
    // Re-build cue nodes
    track.querySelectorAll('.timeline-cue').forEach(n => n.remove());
    const dur = state.video ? state.video.duration : 0;
    if (dur > 0) {
      for (const c of wantCues) {
        const node = document.createElement('div');
        node.className = 'timeline-cue';
        node.dataset.cueId = c.id;
        const left = (c.start / dur) * 100;
        const width = Math.max(0.3, ((c.end - c.start) / dur) * 100);
        node.style.left = left + '%';
        node.style.width = width + '%';
        node.title = (c.text || '').replace(/\n/g, ' / ');
        node.addEventListener('click', e => {
          e.stopPropagation();
          video.currentTime = c.start;
          state.selectedCueId = c.id;
        });
        track.appendChild(node);
      }
    }
  }
  // Update playhead position (cheap)
  if (state.video && state.video.duration > 0) {
    const pct = (video.currentTime / state.video.duration) * 100;
    playhead.style.left = pct + '%';
  }
  // Highlight active cue
  const activeId = state.activeCueId;
  track.querySelectorAll('.timeline-cue').forEach(n => {
    if (n.dataset.cueId === activeId) n.classList.add('active');
    else n.classList.remove('active');
  });
}

// Click on the timeline track background = seek to that time
function bindTimeline() {
  const tl = $('#timeline');
  const track = $('#timelineTrack');
  if (!tl || !track) return;
  tl.addEventListener('click', e => {
    if (e.target.classList.contains('timeline-cue')) return;
    if (!state.video || !state.video.duration) return;
    const rect = track.getBoundingClientRect();
    const pct = (e.clientX - rect.left) / rect.width;
    video.currentTime = clamp(pct * state.video.duration, 0, state.video.duration);
  });
}

// ============================================================================
// Video event handlers
// ============================================================================
const video = $('#video');
let lastRenderedTime = -1;
function maybeRender() {
  if (state.video && Math.abs(video.currentTime - lastRenderedTime) > 0.01) {
    lastRenderedTime = video.currentTime;
    state.currentTime = video.currentTime;
    $('#curTime').textContent = fmtTime(video.currentTime).replace(',', '.');
    renderOverlay();
    renderTimeline();
  }
}
video.addEventListener('timeupdate', maybeRender);
video.addEventListener('seeked', maybeRender);
video.addEventListener('loadeddata', maybeRender);
video.addEventListener('play', () => { $('#playPause').textContent = '⏸'; maybeRender(); });
video.addEventListener('pause', () => { $('#playPause').textContent = '▶'; maybeRender(); });
// Fallback render loop using requestAnimationFrame for smooth updates
// synced with the display refresh. Stops when the page is hidden.
let rafRunning = true;
function rafLoop() {
  if (!rafRunning) return;
  maybeRender();
  requestAnimationFrame(rafLoop);
}
requestAnimationFrame(rafLoop);
document.addEventListener('visibilitychange', () => {
  rafRunning = !document.hidden;
  if (rafRunning) requestAnimationFrame(rafLoop);
});
$('#playPause').addEventListener('click', () => {
  if (video.paused) video.play(); else video.pause();
});
$('#prevCue').addEventListener('click', () => {
  if (!state.srt) return;
  const t = video.currentTime;
  const prev = [...state.srt.cues].reverse().find(c => c.start < t - 0.1);
  if (prev) { video.currentTime = prev.start; state.selectedCueId = prev.id; renderCueList(); }
});
$('#nextCue').addEventListener('click', () => {
  if (!state.srt) return;
  const t = video.currentTime;
  const next = state.srt.cues.find(c => c.start > t + 0.1);
  if (next) { video.currentTime = next.start; state.selectedCueId = next.id; renderCueList(); }
});
$('#shiftLeft').addEventListener('click', () => {
  if (!state.srt) return;
  state.srt.cues.forEach(c => { c.start = Math.max(0, c.start - 0.5); c.end = Math.max(c.start + 0.1, c.end - 0.5); });
  pushHistory(); renderCueList(); renderOverlay();
  toast('Shifted all cues -0.5s');
});
$('#shiftRight').addEventListener('click', () => {
  if (!state.srt) return;
  state.srt.cues.forEach(c => { c.start += 0.5; c.end += 0.5; });
  pushHistory(); renderCueList(); renderOverlay();
  toast('Shifted all cues +0.5s');
});
$('#setStart').addEventListener('click', () => {
  if (!state.srt || !state.selectedCueId) { toast('Select a cue first', 'error'); return; }
  const c = state.srt.cues.find(x => x.id === state.selectedCueId);
  c.start = video.currentTime;
  if (c.end < c.start + 0.1) c.end = c.start + 1;
  pushHistory(); renderCueList(); renderOverlay();
  toast('Start set to current time');
});
$('#setEnd').addEventListener('click', () => {
  if (!state.srt || !state.selectedCueId) { toast('Select a cue first', 'error'); return; }
  const c = state.srt.cues.find(x => x.id === state.selectedCueId);
  c.end = Math.max(c.start + 0.1, video.currentTime);
  pushHistory(); renderCueList(); renderOverlay();
  toast('End set to current time');
});

// ============================================================================
// File inputs + dropzone
// ============================================================================
$('#videoInput').addEventListener('change', e => loadVideoFile(e.target.files[0]));
$('#srtInput').addEventListener('change', e => loadSRTFile(e.target.files[0]));
const dz = $('#dropzone');
['dragenter', 'dragover'].forEach(ev => {
  document.body.addEventListener(ev, e => {
    e.preventDefault();
    if (e.dataTransfer && Array.from(e.dataTransfer.types).includes('Files')) {
      dz.classList.remove('hidden');
    }
  });
});
['dragleave', 'drop'].forEach(ev => {
  document.body.addEventListener(ev, e => { e.preventDefault(); });
});
document.body.addEventListener('drop', e => {
  e.preventDefault();
  if (!e.dataTransfer.files.length) return;
  const f = e.dataTransfer.files[0];
  if (f.name.toLowerCase().endsWith('.srt') || f.type === 'text/plain' || f.type === 'text/vtt') {
    loadSRTFile(f);
  } else {
    loadVideoFile(f);
  }
});

// ============================================================================
// SRT actions
// ============================================================================
$('#newCue').addEventListener('click', () => {
  if (!state.srt) {
    state.srt = { cues: [], fileName: 'new.srt' };
  }
  const t = video.currentTime || 0;
  const c = { id: uid(), seq: state.srt.cues.length + 1, start: t, end: t + 2, text: 'New cue' };
  state.srt.cues.push(c);
  state.srt.cues.sort((a, b) => a.start - b.start);
  state.srt.cues.forEach((x, i) => x.seq = i + 1);
  state.selectedCueId = c.id;
  pushHistory();
  renderCueList();
  renderOverlay();
});
$('#pasteSrt').addEventListener('click', () => { $('#pasteModal').hidden = false; $('#pasteArea').focus(); });
$('#pasteCancel').addEventListener('click', () => { $('#pasteModal').hidden = true; });
$('#pasteApply').addEventListener('click', () => {
  const text = $('#pasteArea').value;
  if (text.trim()) {
    loadSRTFromText(text, 'pasted.srt');
    $('#pasteModal').hidden = true;
    $('#pasteArea').value = '';
  }
});
$('#exportSrt').addEventListener('click', () => {
  if (!state.srt || !state.srt.cues.length) { toast('No SRT to export', 'error'); return; }
  const blob = new Blob([generateSRT(state.srt.cues)], { type: 'text/plain;charset=utf-8' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = (state.srt.fileName || 'subtitles.srt').replace(/\.srt$/, '') + '.srt';
  a.click();
  URL.revokeObjectURL(a.href);
  toast('SRT downloaded', 'success');
});
$('#searchBox').addEventListener('input', () => renderCueList());

$('#findReplace').addEventListener('click', () => {
  if (!state.srt) { toast('No SRT loaded', 'error'); return; }
  $('#frModal').hidden = false;
  $('#frFind').focus();
});
$('#frCancel').addEventListener('click', () => { $('#frModal').hidden = true; });
$('#frApply').addEventListener('click', () => {
  const f = $('#frFind').value;
  const r = $('#frReplace').value;
  if (!f) { toast('Enter text to find', 'error'); return; }
  let n = 0;
  state.srt.cues.forEach(c => {
    const before = c.text;
    c.text = c.text.split(f).join(r);
    if (c.text !== before) n += (before.split(f).length - 1);
  });
  pushHistory(); renderCueList(); renderOverlay();
  $('#frModal').hidden = true;
  toast(`Replaced ${n} occurrence${n === 1 ? '' : 's'}`, 'success');
});

// ============================================================================
// Style controls
// ============================================================================
const styleMap = {
  fontFamily: ['fontFamily', 'change'],
  fontSize: ['fontSize', 'input'],
  fontWeight: ['fontWeight', 'change'],
  fontStyle: ['fontStyle', 'change'],
  primaryColor: ['primaryColor', 'input'],
  outlineColor: ['outlineColor', 'input'],
  outlineWidth: ['outlineWidth', 'input'],
  backColor: ['backColor', 'input'],
  backOpacity: ['backOpacity', 'input'],
  shadow: ['shadow', 'change'],
  marginV: ['marginV', 'input'],
  marginH: ['marginH', 'input'],
};
for (const [id, [key, ev]] of Object.entries(styleMap)) {
  const el = $('#' + id);
  if (!el) continue;
  el.addEventListener(ev, e => {
    let v = e.target.value;
    if (el.type === 'checkbox') v = e.target.checked;
    else if (el.type === 'range' || el.type === 'number') v = +v;
    state.style[key] = v;
    const valEl = $('#' + id + 'Val');
    if (valEl) valEl.textContent = v;
    renderOverlay();
  });
}
$$('input[name="alignment"]').forEach(r => {
  r.addEventListener('change', e => {
    if (e.target.checked) {
      state.style.alignment = +e.target.value;
      // Changing alignment clears per-cue overrides so the new alignment applies
      // (user can re-drag if they want)
      // (We keep the overrides so the user can still see their custom positions if they want;
      //  uncomment below to reset on alignment change: state.cueOverrides = {};)
      renderOverlay();
    }
  });
});

// ============================================================================
// Tab switching
// ============================================================================
$$('.tab').forEach(btn => {
  btn.addEventListener('click', () => {
    $$('.tab').forEach(t => t.classList.toggle('active', t === btn));
    const target = btn.dataset.tab;
    $$('.panel').forEach(p => p.hidden = p.dataset.panel !== target);
  });
});

// ============================================================================
// Translation (MyMemory)
// ============================================================================
$('#runTranslate').addEventListener('click', async () => {
  if (!state.srt || !state.srt.cues.length) { toast('No SRT loaded', 'error'); return; }
  if (state.translating) return;
  state.translating = true;
  const cancel = $('#cancelTranslate');
  cancel.hidden = false;
  const status = $('#translateStatus');
  const src = $('#srcLang').value;
  const tgt = $('#tgtLang').value;

  let done = 0;
  for (const c of state.srt.cues) {
    if (!state.translating) break;
    status.textContent = `Translating ${done + 1} / ${state.srt.cues.length}…`;
    try {
      const lines = c.text.split('\n');
      const out = [];
      for (const line of lines) {
        if (!line.trim()) { out.push(line); continue; }
        const url = `https://api.mymemory.translated.net/get?q=${encodeURIComponent(line)}&langpair=${src === 'auto' ? 'autodetect' : src}|${tgt}&de=team@srt.studio`;
        const r = await fetch(url);
        if (!r.ok) throw new Error('HTTP ' + r.status);
        const j = await r.json();
        if (j.responseStatus !== 200) throw new Error(j.responseDetails || 'API error');
        out.push(j.responseData.translatedText);
      }
      c.text = out.join('\n');
      done++;
      renderOverlay();
    } catch (e) {
      console.error(e);
      status.textContent = `Error: ${e.message}. Continuing…`;
    }
  }
  state.translating = false;
  cancel.hidden = true;
  status.textContent = `Done. ${done} cues translated.`;
  pushHistory();
  renderCueList();
  toast(`Translated ${done} cues`, 'success');
});
$('#cancelTranslate').addEventListener('click', () => { state.translating = false; });

// ============================================================================
// Render (FFmpeg.wasm)
// ============================================================================
let ffmpegInstance = null;
async function ensureFFmpeg() {
  if (ffmpegInstance) return ffmpegInstance;
  if (typeof FFmpeg === 'undefined') throw new Error('FFmpeg library not loaded yet — wait a sec and try again');
  const { createFFmpeg } = FFmpeg;
  ffmpegInstance = createFFmpeg({
    log: false,
    corePath: 'https://unpkg.com/@ffmpeg/core@0.12.6/dist/umd/ffmpeg-core.js',
  });
  await ffmpegInstance.load();
  return ffmpegInstance;
}

function buildASS(cues, style) {
  // Build an ASS subtitle file from cues + style state
  const hex = c => '&H' + c.replace('#', '').toUpperCase().split('').reverse().join('') + '&';
  const alignMap = { 1:7, 2:8, 3:9, 4:4, 5:5, 6:6, 7:1, 8:2, 9:3 }; // ASS numeric alignment (different from numpad)
  // use a standard layout
  const header = `[Script Info]
Title: SRT Studio
ScriptType: v4.00+
WrapStyle: 0
PlayResX: 1280
PlayResY: 720
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,${style.fontFamily},${style.fontSize * 2},${hex(style.primaryColor)},&H000000FF,${hex(style.outlineColor)},${hex(style.backColor)},${style.fontWeight === 'bold' ? -1 : 0},${style.fontStyle === 'italic' ? -1 : 0},0,0,100,100,0,0,1,${style.outlineWidth * 2},${style.shadow ? 2 : 0},${alignMap[style.alignment] || 2},${style.marginH * 4},${style.marginH * 4},${style.marginV * 2},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text`;

  const lines = cues.map(c => {
    const s = fmtASS(c.start);
    const e = fmtASS(c.end);
    const txt = c.text.replace(/\n/g, '\\N');
    return `Dialogue: 0,${s},${e},Default,,0,0,0,,${txt}`;
  });
  return header + '\n' + lines.join('\n') + '\n';
}

function fmtASS(sec) {
  const h = Math.floor(sec / 3600);
  const m = Math.floor(sec / 60) % 60;
  const s = Math.floor(sec) % 60;
  const cs = Math.floor((sec % 1) * 100);
  return `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}.${String(cs).padStart(2, '0')}`;
}

$('#runRender').addEventListener('click', async () => {
  if (state.rendering) return;
  if (!state.video) { toast('Load a video first', 'error'); return; }
  if (!state.srt || !state.srt.cues.length) { toast('Load subtitles first', 'error'); return; }
  state.rendering = true;
  $('#cancelRender').hidden = false;
  $('#downloadRendered').hidden = true;
  const wrap = $('#renderProgress');
  const fill = $('#renderFill');
  const text = $('#renderText');
  wrap.hidden = false;
  fill.style.width = '0%';
  text.textContent = 'Loading FFmpeg…';
  try {
    const ff = await ensureFFmpeg();
    text.textContent = 'Building subtitle file…';
    fill.style.width = '5%';
    const ass = buildASS(state.srt.cues, state.style);
    const quality = $('#renderQuality').value;
    const crf = quality === 'high' ? '20' : quality === 'fast' ? '28' : '23';
    const preset = quality === 'high' ? 'slow' : quality === 'fast' ? 'ultrafast' : 'medium';
    const format = $('#renderFormat').value;
    const inName = 'input.' + (state.video.file.name.split('.').pop() || 'mp4');
    const outName = 'output.' + format;
    const subName = 'subs.ass';
    ff.FS('writeFile', subName, new TextEncoder().encode(ass));
    text.textContent = 'Loading video into FFmpeg…';
    const buf = new Uint8Array(await state.video.file.arrayBuffer());
    ff.FS('writeFile', inName, buf);
    fill.style.width = '15%';
    text.textContent = 'Encoding (this can take a few minutes)…';

    const args = ['-i', inName, '-vf', `ass=${subName}`,
      '-c:v', 'libx264', '-preset', preset, '-crf', crf, '-pix_fmt', 'yuv420p',
      '-c:a', 'aac', '-b:a', '128k'];
    if (format === 'mp4') {
      args.push('-movflags', '+faststart');
    } else {
      args.push('-c:v', 'libvpx-vp9', '-b:v', '0', '-crf', crf);
    }
    args.push(outName);

    const prog = setInterval(() => {
      const lines = ff && ff.logger && ff.logger.buffer ? ff.logger.buffer : [];
      // ffmpeg.wasm v0.12 doesn't expose progress reliably; just show indeterminate activity
    }, 1000);
    await ff.run(...args);
    clearInterval(prog);

    fill.style.width = '95%';
    text.textContent = 'Reading output…';
    const data = ff.FS('readFile', outName);
    const blob = new Blob([data.buffer], { type: format === 'mp4' ? 'video/mp4' : 'video/webm' });
    const url = URL.createObjectURL(blob);
    const a = $('#downloadRendered');
    a.href = url;
    a.download = 'rendered.' + format;
    a.hidden = false;
    fill.style.width = '100%';
    text.textContent = `Done! ${(blob.size / 1024 / 1024).toFixed(1)} MB`;
    toast('Render complete', 'success');
  } catch (e) {
    console.error(e);
    text.textContent = 'Error: ' + e.message;
    toast('Render failed: ' + e.message, 'error');
  } finally {
    state.rendering = false;
    $('#cancelRender').hidden = true;
  }
});
$('#cancelRender').addEventListener('click', () => {
  state.rendering = false;
  // FFmpeg.wasm v0.12 doesn't support true cancellation mid-run; just stop polling
});

// ============================================================================
// Keyboard shortcuts
// ============================================================================
document.addEventListener('keydown', e => {
  const inField = ['INPUT', 'TEXTAREA', 'SELECT'].includes(e.target.tagName);
  if (e.ctrlKey || e.metaKey) {
    if (e.key === 'z' && !e.shiftKey) { e.preventDefault(); undo(); return; }
    if ((e.key === 'y') || (e.key === 'z' && e.shiftKey)) { e.preventDefault(); redo(); return; }
    if (e.key === 'Enter') { e.preventDefault(); $('#newCue').click(); return; }
    if (e.key === 's') { e.preventDefault(); $('#exportSrt').click(); return; }
  }
  // Esc closes any open modal
  if (e.key === 'Escape') {
    ['pasteModal', 'frModal', 'shortcutsModal'].forEach(id => {
      const m = document.getElementById(id);
      if (m && !m.hidden) m.hidden = true;
    });
    return;
  }
  // ? opens shortcuts (only when not in a text field)
  if (!inField && e.key === '?') {
    e.preventDefault();
    $('#shortcutsModal').hidden = false;
    return;
  }
  if (inField) return;
  if (e.key === ' ') { e.preventDefault(); $('#playPause').click(); return; }
  if (e.key === 'Tab' && !e.shiftKey) { e.preventDefault(); $('#nextCue').click(); return; }
  if (e.key === 'Tab' && e.shiftKey) { e.preventDefault(); $('#prevCue').click(); return; }
});

// ============================================================================
// Init
// ============================================================================
renderCueList();
updateSaveStatus();
bindTimeline();

// auto-load from URL hash if present
if (location.hash) {
  const params = new URLSearchParams(location.hash.slice(1));
  if (params.get('cue')) state.selectedCueId = params.get('cue');
}

// ============================================================================
// Custom font upload
// ============================================================================
let customFontFamily = null;
$('#customFont').addEventListener('change', e => {
  const file = e.target.files[0];
  if (!file) return;
  if (!/\.(ttf|otf|woff2?|TTF|OTF|WOFF2?)$/.test(file.name)) {
    toast('Please drop a .ttf, .otf, or .woff/.woff2 file', 'error');
    return;
  }
  if (customFontFamily) URL.revokeObjectURL(customFontFamily);
  const url = URL.createObjectURL(file);
  // We can't use the @font-face @font-family trick with an arbitrary
  // name cleanly, so we inject a stylesheet that defines a unique-named
  // font for this file. We pick a stable name and store it.
  const fontName = 'SRTStudioCustom';
  customFontFamily = fontName;
  const style = document.createElement('style');
  style.textContent = `
    @font-face {
      font-family: '${fontName}';
      src: url('${url}') format('${file.name.toLowerCase().endsWith('woff2') ? 'woff2' :
        file.name.toLowerCase().endsWith('woff') ? 'woff' :
        file.name.toLowerCase().endsWith('otf') ? 'opentype' : 'truetype'}');
      font-display: block;
    }
  `;
  // Remove previous injection for this font
  document.querySelectorAll('style[data-customfont]').forEach(n => n.remove());
  style.dataset.customfont = '1';
  document.head.appendChild(style);
  // Switch the dropdown to "Custom" and apply
  const sel = $('#fontFamily');
  if (sel) {
    sel.value = 'custom';
    state.style.fontFamily = `'${fontName}', sans-serif`;
    renderOverlay();
    toast('Custom font loaded: ' + file.name, 'success');
  }
});
$('#fontFamily').addEventListener('change', e => {
  const v = e.target.value;
  const label = $('#customFontLabel');
  if (v === 'custom') {
    label.hidden = false;
    if (customFontFamily) {
      state.style.fontFamily = `'${customFontFamily}', sans-serif`;
    } else {
      toast('Upload a font file below', '');
      state.style.fontFamily = 'Inter, sans-serif';
    }
  } else {
    label.hidden = true;
    state.style.fontFamily = v;
  }
  renderOverlay();
});

// ============================================================================
// Keyboard shortcuts modal
// ============================================================================
$('#showShortcuts').addEventListener('click', () => { $('#shortcutsModal').hidden = false; });
$('#closeShortcuts').addEventListener('click', () => { $('#shortcutsModal').hidden = true; });
$('#shortcutsModal').addEventListener('click', e => {
  if (e.target === e.currentTarget) e.currentTarget.hidden = true;
});

// ============================================================================
// Download from URL (cobalt.tools API)
// ============================================================================
// Cobalt is a free, open-source download service. We call its public API
// (no key required) which returns either a stream URL or a tunnel we can
// pull the bytes from. The video bytes are downloaded directly to your
// browser — they are NOT stored on cobalt's servers.
//
// If cobalt is down, unreachable, or has changed their API, the user gets
// a clear error and is told to use yt-dlp on their own machine.
const COBALT_INSTANCES = [
  'https://api.cobalt.tools/',  // official
];

async function cobaltResolveStream(url, type, quality, container) {
  // Map our UI choices to cobalt's API params
  const body = { url };
  if (type === 'audio') {
    body.downloadMode = 'audio';
    body.audioFormat = container === 'webm' ? 'webm' : 'mp3';
    body.audioBitrate = '128';
  } else {
    body.downloadMode = 'auto';
    if (quality && quality !== 'max') body.videoQuality = quality;
    if (container && container !== 'auto') body.container = container;
  }
  body.filenameStyle = 'classic';
  body.tiktokFullAudio = false;
  body.youtubeDubLang = null;

  const r = await fetch('https://api.cobalt.tools/api/json', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
    },
    body: JSON.stringify(body),
  });
  if (!r.ok) {
    const text = await r.text().catch(() => '');
    throw new Error(`Cobalt API error (HTTP ${r.status}): ${text.slice(0, 200)}`);
  }
  const j = await r.json();
  if (j.status === 'error') throw new Error(j.error?.code || 'Cobalt error');
  if (j.status === 'redirect' || j.status === 'tunnel') {
    return { streamUrl: j.url, filename: j.filename || 'download' };
  }
  if (j.status === 'picker') {
    // Multiple items (e.g. carousel) — pick first
    if (j.picker && j.picker.length) {
      return { streamUrl: j.picker[0].url, filename: j.picker[0].filename || 'download' };
    }
    throw new Error('Cobalt returned multiple items; please refine the URL');
  }
  throw new Error('Unexpected cobalt response: ' + JSON.stringify(j).slice(0, 200));
}

async function downloadWithProgress(streamUrl, onProgress) {
  const r = await fetch(streamUrl);
  if (!r.ok) throw new Error(`Stream fetch failed: HTTP ${r.status}`);
  const total = +r.headers.get('content-length') || 0;
  const reader = r.body.getReader();
  const chunks = [];
  let received = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    if (total) onProgress(received / total);
  }
  const blob = new Blob(chunks);
  return blob;
}

$('#runDownload').addEventListener('click', async () => {
  const url = $('#dlUrl').value.trim();
  if (!url) { toast('Paste a URL first', 'error'); return; }
  if (!/^https?:\/\//.test(url)) { toast('Must start with http:// or https://', 'error'); return; }
  const type = $('#dlType').value;
  const quality = $('#dlQuality').value;
  const container = $('#dlContainer').value;
  const loadToEditor = $('#dlSaveToBrowser').checked;

  $('#cancelDownload').hidden = false;
  $('#runDownload').disabled = true;
  const wrap = $('#dlProgress');
  const fill = $('#dlFill');
  const text = $('#dlText');
  wrap.hidden = false;
  fill.style.width = '0%';

  let cancelled = false;
  $('#cancelDownload').onclick = () => { cancelled = true; };

  try {
    text.textContent = 'Asking cobalt to resolve the URL…';
    fill.style.width = '5%';
    const { streamUrl, filename } = await cobaltResolveStream(url, type, quality, container);
    if (cancelled) return;
    text.textContent = 'Downloading…';
    const blob = await downloadWithProgress(streamUrl, p => {
      fill.style.width = (5 + p * 90) + '%';
      text.textContent = `Downloading… ${(p * 100).toFixed(0)}%`;
    });
    if (cancelled) return;
    fill.style.width = '100%';
    text.textContent = `Done! ${(blob.size / 1024 / 1024).toFixed(1)} MB`;

    if (loadToEditor) {
      // Build a File and load into the video player
      const ext = (filename.split('.').pop() || 'mp4').toLowerCase();
      const mime = ext === 'webm' ? 'video/webm'
                : ext === 'mkv' ? 'video/x-matroska'
                : ext === 'mp3' ? 'audio/mpeg'
                : ext === 'm4a' ? 'audio/mp4'
                : 'video/mp4';
      const f = new File([blob], filename, { type: mime });
      loadVideoFile(f);
      // Auto-switch to Editor tab so the user sees the loaded video
      const ed = document.querySelector('.tab[data-tab="editor"]');
      if (ed) ed.click();
      toast('Loaded into editor', 'success');
    } else {
      // Save to disk
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = filename;
      a.click();
      URL.revokeObjectURL(a.href);
      toast('Downloaded to your device', 'success');
    }
  } catch (e) {
    console.error(e);
    fill.style.width = '0%';
    text.textContent = '';
    toast('Download failed: ' + e.message, 'error');
    if (/Cobalt|fetch|API/i.test(e.message)) {
      // Most likely cause: cobalt down or unreachable
      toast('Cobalt might be down — try yt-dlp on your own machine and drag the result here', 'error');
    }
  } finally {
    $('#runDownload').disabled = false;
    $('#cancelDownload').hidden = true;
    setTimeout(() => { wrap.hidden = true; }, 3000);
  }
});
