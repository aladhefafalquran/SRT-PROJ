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
  const snap = JSON.parse(JSON.stringify({ cues: state.srt ? state.srt.cues : [] }));
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
  state.srt.cues = JSON.parse(JSON.stringify(snap.cues));
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
    $('#totalTime').textContent = fmtTime(v.duration).replace(',', '.');
  };
  toast('Video loaded: ' + file.name, 'success');
}

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
  if (!state.srt || !state.video) return;
  const t = state.video.currentTime || 0;
  const s = state.style;
  const activeCue = state.srt.cues.find(c => t >= c.start && t <= c.end + 0.05);
  state.activeCueId = activeCue ? activeCue.id : null;
  if (!activeCue) return;
  const cue = document.createElement('div');
  cue.className = 'cue';
  cue.textContent = activeCue.text;
  // position via ASS alignment
  // alignment: 1..3 top, 4..6 mid, 7..9 bottom
  // 1,4,7 = left   2,5,8 = center   3,6,9 = right
  // vertical thirds:
  //   1,2,3   = top
  //   4,5,6   = middle
  //   7,8,9   = bottom
  const a = s.alignment;
  const vert = a <= 3 ? 'top' : a <= 6 ? 'center' : 'bottom';
  const horiz = (a === 1 || a === 4 || a === 7) ? 'left' :
                (a === 3 || a === 6 || a === 9) ? 'right' : 'center';
  cue.style.top = vert === 'top' ? s.marginV + 'px' :
                  vert === 'bottom' ? `calc(100% - ${s.marginV}px)` : '50%';
  cue.style.bottom = '';
  cue.style.left = horiz === 'left' ? s.marginH + 'px' :
                   horiz === 'right' ? `calc(100% - ${s.marginH}px)` : '50%';
  cue.style.right = '';
  cue.style.transform = horiz === 'center' && vert === 'center' ? 'translate(-50%, -50%)' :
                       horiz === 'center' ? 'translateX(-50%)' :
                       vert === 'center' ? 'translateY(-50%)' : 'none';
  cue.style.fontFamily = s.fontFamily;
  cue.style.fontSize = s.fontSize + 'px';
  cue.style.fontWeight = s.fontWeight;
  cue.style.fontStyle = s.fontStyle;
  cue.style.color = s.primaryColor;
  cue.style.lineHeight = '1.3';
  // background with opacity
  const bg = hexToRgb(s.backColor);
  cue.style.backgroundColor = `rgba(${bg.r},${bg.g},${bg.b},${s.backOpacity / 100})`;
  cue.style.padding = '4px 8px';
  cue.style.borderRadius = '3px';
  // text shadow as outline
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
  // drag-to-reposition
  let drag = null;
  cue.addEventListener('pointerdown', e => {
    e.preventDefault();
    cue.setPointerCapture(e.pointerId);
    const rect = overlay.getBoundingClientRect();
    drag = { x: e.clientX, y: e.clientY, left: cue.offsetLeft, top: cue.offsetTop, rect };
    cue.classList.add('dragging');
  });
  cue.addEventListener('pointermove', e => {
    if (!drag) return;
    const dx = (e.clientX - drag.x) / drag.rect.width * 100;
    const dy = (e.clientY - drag.y) / drag.rect.height * 100;
    cue.style.left = `calc(${drag.left / drag.rect.width * 100}% + ${dx}%)`;
    cue.style.top = `calc(${drag.top / drag.rect.height * 100}% + ${dy}%)`;
    cue.style.transform = 'none';
    cue.style.right = 'auto';
    cue.style.bottom = 'auto';
  });
  cue.addEventListener('pointerup', e => {
    if (!drag) return;
    cue.releasePointerCapture(e.pointerId);
    drag = null;
    cue.classList.remove('dragging');
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
// Video event handlers
// ============================================================================
const video = $('#video');
video.addEventListener('timeupdate', () => {
  state.currentTime = video.currentTime;
  $('#curTime').textContent = fmtTime(video.currentTime).replace(',', '.');
  renderOverlay();
});
video.addEventListener('play', () => { $('#playPause').textContent = '⏸'; });
video.addEventListener('pause', () => { $('#playPause').textContent = '▶'; });
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
    if (e.target.checked) { state.style.alignment = +e.target.value; renderOverlay(); }
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

// auto-load from URL hash if present
if (location.hash) {
  const params = new URLSearchParams(location.hash.slice(1));
  if (params.get('cue')) state.selectedCueId = params.get('cue');
}
