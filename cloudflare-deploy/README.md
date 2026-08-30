# SRT Studio

A 100% browser-based subtitle editor. Upload a video, edit the SRT, style the
subtitles in real time, render the result as MP4/WebM. No server, no storage,
no signup.

**Live demo:** https://srt-studio.pages.dev

## Features

- 🎬 Drag-and-drop video (any format your browser plays)
- 📂 Load `.srt` file, paste raw SRT, or build cues from scratch
- ✏️ Edit timing + text in the sidebar, with live preview
- 🎨 Style controls: font, size, color, outline, background, shadow, position
- 🖱️ **Drag subtitles** on the video to reposition them
- 🔍 Find & replace across all cues
- ⏪⏩ Shift all cues by 0.5s for easy syncing
- 🌐 Translate to Arabic / 12+ languages via MyMemory (free, no key)
- 💾 Export `.srt` or render final video with subs burned in (FFmpeg.wasm)
- ⌨️ Keyboard shortcuts: `Space` play/pause, `Tab` next cue, `Ctrl+Z` undo, `Ctrl+Enter` new cue
- 💽 Auto-saves your work to your browser (no data ever leaves your machine)
- 📱 Mobile-friendly

## Tech

- Pure HTML/CSS/JS — no build step, no framework
- FFmpeg.wasm for video rendering (runs in browser, ~30MB cached)
- MyMemory API for translation
- Cloudflare Pages for hosting (free)

## Local development

```bash
cd cloudflare-deploy
npm install
npm run dev    # http://localhost:8788
```

## Deploy

```bash
npm install
npm run deploy
```

First-time deploy creates the Cloudflare Pages project `srt-studio`. URL will be
`https://srt-studio.pages.dev`.

## Privacy

- Videos and SRTs are processed in your browser only
- Nothing is uploaded to any server (no backend exists)
- The only network call is to MyMemory for translation, and only when you click "Translate"
- Closing the tab erases everything (auto-save is in browser localStorage)
