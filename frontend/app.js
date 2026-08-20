/* THE EXCHANGE — voice RAG frontend wiring.
   Same API contract as before; new visuals.
   Stages: guard → embed → retrieve → gate → generate → verify
*/
(function () {
  "use strict";

  const $ = (s) => document.querySelector(s);
  const $$ = (s) => Array.from(document.querySelectorAll(s));

  /* ================= loader ================= */
  const loader = $("#loader");
  function hideLoader() {
    if (!loader || loader.classList.contains("is-gone")) return;
    loader.classList.add("is-gone");
  }
  document.addEventListener("DOMContentLoaded", hideLoader);
  window.addEventListener("load", hideLoader);
  setTimeout(hideLoader, 50);

  /* ================= hero oscilloscope ================= */
  (function wave() {
    const cv = $("#waveCanvas");
    if (!cv) return;
    const ctx = cv.getContext("2d");
    let W = 0, H = 0;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    function size() {
      W = cv.clientWidth; H = cv.clientHeight;
      cv.width = W * dpr; cv.height = H * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }
    size();
    window.addEventListener("resize", size);

    const gold = "rgba(240,193,99,";
    const dim = "rgba(139,126,96,";

    function trace(offset) {
      const n = Math.max(60, Math.floor(W / 3));
      const pts = [];
      let t = performance.now() / 1000;
      for (let i = 0; i <= n; i++) {
        const x = (i / n) * W;
        const f = 2.1 + 1.7 * Math.sin(t * 0.9 + i * 0.03);
        const y =
          H / 2 +
          Math.sin(x * 0.02 + t * 2.2 + offset) * H * 0.16 +
          Math.sin(x * 0.05 - t * 1.6) * H * 0.07 +
          Math.sin(x * 0.008 + t * 0.8) * H * 0.12;
        pts.push([x, y]);
      }
      return pts;
    }

    function draw() {
      ctx.clearRect(0, 0, W, H);

      // grid
      ctx.strokeStyle = dim + ".14)";
      ctx.lineWidth = 1;
      for (let g = 1; g < 8; g++) {
        const gy = (g / 8) * H;
        ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(W, gy); ctx.stroke();
      }
      ctx.strokeStyle = dim + ".06)";
      for (let g = 1; g < 20; g++) {
        const gx = (g / 20) * W;
        ctx.beginPath(); ctx.moveTo(gx, 0); ctx.lineTo(gx, H); ctx.stroke();
      }

      // baseline
      ctx.strokeStyle = dim + ".25)";
      ctx.beginPath(); ctx.moveTo(0, H / 2); ctx.lineTo(W, H / 2); ctx.stroke();

      const t = performance.now() / 1000;
      // ghost trace
      paint(trace(-0.9), gold + ".12)", 1.4);
      // main trace
      paint(trace(0), gold + ".92)", 1.9);

      // sweep dot
      const n = Math.max(60, Math.floor(W / 3));
      const i = Math.floor(((t * 3) % 1.2) * n);
      if (i <= n) {
        const p = trace(0)[i];
        ctx.beginPath(); ctx.arc(p[0], p[1], 2.4, 0, 7); ctx.fillStyle = "#FFD98A"; ctx.fill();
      }
      requestAnimationFrame(draw);
    }
    function paint(pts, col, lw) {
      ctx.strokeStyle = col;
      ctx.lineWidth = lw;
      ctx.shadowColor = "rgba(240,193,99,.55)";
      ctx.shadowBlur = 8;
      ctx.beginPath();
      ctx.moveTo(pts[0][0], pts[0][1]);
      for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
      ctx.stroke();
      ctx.shadowBlur = 0;
    }
    draw();
  })();

  /* ================= reveal on scroll ================= */
  const io = new IntersectionObserver(
    (entries) => {
      for (const e of entries) {
        if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); }
      }
    },
    { threshold: 0.12 }
  );
  $$(".reveal").forEach((el) => io.observe(el));

  /* ================= animated counters ================= */
  function fmtInt(n) { return Number(n || 0).toLocaleString("en-IN"); }
  function countTo(el, target, suffix) {
    if (!el) return;
    const from = parseFloat(el.dataset.val || "0") || 0;
    const to = parseFloat(target) || 0;
    const suf = suffix || "";
    if (el.dataset.val === String(to) && to !== 0) return;
    el.dataset.val = String(to);
    const t0 = performance.now();
    const dur = 700;
    (function tick(now) {
      const p = Math.min(1, (now - t0) / dur);
      const eased = 1 - Math.pow(1 - p, 3);
      const v = from + (to - from) * eased;
      el.textContent = (to >= 10000 ? Math.round(v) : Math.round(v * 10) / 10) + suf;
      if (p < 1) requestAnimationFrame(tick);
    })(t0);
  }
  function pulse(el) {
    if (!el) return;
    el.classList.remove("pulse-flash");
    void el.offsetWidth;
    el.classList.add("pulse-flash");
  }

  /* ================= gauge needles ================= */
  function setNeedle(name, value, max) {
    const el = document.querySelector('[data-gauge="' + name + '"]');
    if (!el) return;
    const ratio = Math.max(0, Math.min(1, value / max));
    el.style.transform = "rotate(" + (-90 + ratio * 180) + "deg)";
  }

  /* ================= index info ================= */
  async function loadIndexInfo() {
    try {
      const r = await fetch("/api/index-info");
      const info = await r.json();
      const st = info.strategies || {};
      let total = 0;
      $$("[data-strategy]").forEach((el) => {
        const key = el.dataset.strategy;
        if (key in st) total += Number(st[key]) || 0;
        el.textContent = key in st ? st[key] + " slips" : "–";
      });
      countTo($("#sChunks"), total);
    } catch (_) {}
  }

  /* ================= metrics ================= */
  async function pollMetrics() {
    try {
      const r = await fetch("/api/metrics");
      const m = await r.json();
      const f = (ms) => (typeof ms === "number" ? ms.toFixed(1) + " ms" : "–");
      countTo($("#sRequests"), m.total_requests);
      countTo($("#mP50"), m.p50_ms, " ms");
      countTo($("#mP70"), m.p70_ms, " ms");
      countTo($("#mP100"), m.p100_ms, " ms");
      countTo($("#mN"), m.total_requests);
      $("#sP70").textContent = typeof m.p70_ms === "number" ? m.p70_ms.toFixed(0) + " ms" : "–";
      pulse($("#sP70"));

      setNeedle("p50", m.p50_ms || 0, 200);
      setNeedle("p70", m.p70_ms || 0, 200);
      setNeedle("p100", m.p100_ms || 0, 200);
      const maxReq = Math.max(10, Math.ceil((m.total_requests || 0) / 10) * 10);
      setNeedle("n", m.total_requests || 0, maxReq);

      const stages = m.by_stage || {};
      const box = $("#stageBars");
      const maxP = Math.max(1, ...Object.values(stages).map((s) => s.p70_ms || 0));
      box.innerHTML = "";
      for (const [name, s] of Object.entries(stages)) {
        const w = Math.min(100, (100 * (s.p70_ms || 0)) / maxP);
        const row = document.createElement("div");
        row.className = "bar";
        row.innerHTML =
          '<span class="bar__name mono">' + name + "</span>" +
          '<div class="bar__track"><div class="bar__fill" style="width:' + w + '%"></div></div>' +
          '<span class="bar__val mono">' + f(s.p70_ms) + "</span>";
        box.appendChild(row);
      }
    } catch (_) {}
  }
  pollMetrics();
  setInterval(pollMetrics, 3000);

  /* ================= demo state ================= */
  const answerEl = $("#answer");
  const metaEl = $("#meta");
  const sttEl = $("#sttBadge");
  const sourcesEl = $("#sources");
  const statusEl = $("#status");
  const lane = $("#patchLane");
  const laneDot = $("#laneDot");
  const laneLamps = $$("[data-stage-line]");
  const slipNo = $(".slip__no");
  let slipCount = 0;

  function setStatus(msg, isErr) {
    statusEl.textContent = msg;
    statusEl.classList.toggle("error", !!isErr);
  }
  const stageOrder = ["guard", "embed", "retrieve", "gate", "generate", "verify"];
  function resetStage() {
    laneLamps.forEach((l) => l.classList.remove("lane__lamp--active", "lane__lamp--done"));
    lane.classList.remove("is-active");
  }
  function markStage(name, state) {
    const idx = stageOrder.indexOf(name);
    if (state === "active" && idx >= 0) {
      laneDot.style.left = 5 + (idx * 90) / (stageOrder.length - 1) + "%";
      lane.classList.add("is-active");
    }
    laneLamps.forEach((l) => {
      if (l.dataset.stageLine === name) {
        l.classList.remove("lane__lamp--active", "lane__lamp--done");
        if (state === "active") l.classList.add("lane__lamp--active");
        else if (state === "done") l.classList.add("lane__lamp--done");
      }
    });
  }
  let accumulatedAnswer = "";
  const ttsPlayBtn = $("#ttsPlayBtn");
  let currentAudio = null;

  function formatMarkdown(text) {
    if (!text) return "";
    let s = String(text);
    // Remove raw citation brackets e.g. [SOURCE 1]
    s = s.replace(/\[(?:SOURCE\s*\d+|\d+)\]/gi, "");
    // Bold: **text** or __text__
    s = s.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
    s = s.replace(/__(.*?)__/g, "<strong>$1</strong>");
    // Italic: *text* or _text_
    s = s.replace(/\*([^\*\n]+)\*/g, "<em>$1</em>");
    // Headers: ### Header
    s = s.replace(/^#{1,6}\s*(.*)$/gm, "<strong>$1</strong><br>");
    // Bullets: - bullet
    s = s.replace(/^\s*[\-\*]\s+(.*)$/gm, "• $1<br>");
    // Newlines
    s = s.replace(/\n\n+/g, "<br><br>");
    s = s.replace(/\n/g, "<br>");
    return s;
  }

  function setAnswer(text, isFinal) {
    accumulatedAnswer = text;
    if (isFinal) {
      answerEl.innerHTML = formatMarkdown(accumulatedAnswer);
    } else {
      const cursor = '<span class="answer__cursor"></span>';
      answerEl.innerHTML = formatMarkdown(accumulatedAnswer) + cursor;
    }
    answerEl.scrollTop = answerEl.scrollHeight;
  }

  function clearOut() {
    accumulatedAnswer = "";
    if (currentAudio) {
      currentAudio.pause();
      currentAudio = null;
    }
    if (ttsPlayBtn) {
      ttsPlayBtn.classList.remove("is-playing");
      ttsPlayBtn.textContent = "🔊 LISTEN VOICE";
    }
    answerEl.innerHTML = '<span class="answer__cursor"></span>';
    metaEl.textContent = "—";
    sttEl.textContent = "—";
    sourcesEl.innerHTML = "";
    slipCount += 1;
    slipNo.textContent = "#054-" + String(slipCount).padStart(3, "0");
    resetStage();
  }

  async function playVoice(text) {
    if (!text || !text.trim()) return;
    const plainText = text.replace(/<[^>]+>/g, "").replace(/[\*\#\_\[\]]/g, "").trim();
    if (!plainText) return;

    if (currentAudio) {
      currentAudio.pause();
      currentAudio = null;
    }

    if (ttsPlayBtn) {
      ttsPlayBtn.classList.add("is-playing");
      ttsPlayBtn.textContent = "🔊 SPEAKING…";
    }

    try {
      const resp = await fetch("/api/tts", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: plainText }),
      });
      if (!resp.ok) throw new Error("TTS HTTP " + resp.status);
      const blob = await resp.blob();
      const url = URL.createObjectURL(blob);
      currentAudio = new Audio(url);
      currentAudio.onended = () => {
        if (ttsPlayBtn) {
          ttsPlayBtn.classList.remove("is-playing");
          ttsPlayBtn.textContent = "🔊 LISTEN VOICE";
        }
        currentAudio = null;
      };
      currentAudio.onerror = () => {
        if (ttsPlayBtn) {
          ttsPlayBtn.classList.remove("is-playing");
          ttsPlayBtn.textContent = "🔊 LISTEN VOICE";
        }
        currentAudio = null;
      };
      await currentAudio.play();
    } catch (e) {
      console.error("TTS playback error:", e);
      if (ttsPlayBtn) {
        ttsPlayBtn.classList.remove("is-playing");
        ttsPlayBtn.textContent = "🔊 LISTEN VOICE";
      }
    }
  }

  if (ttsPlayBtn) {
    ttsPlayBtn.addEventListener("click", () => {
      const txt = accumulatedAnswer || answerEl.textContent || "";
      playVoice(txt);
    });
  }

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;")
      .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  }
  function renderSources(docs) {
    sourcesEl.innerHTML = "";
    (docs || []).forEach((d, i) => {
      const row = document.createElement("div");
      row.className = "src";
      row.innerHTML =
        '<span class="src__idx mono">S' + (i + 1) + "</span>" +
        '<span class="src__text">' + escapeHtml(d.text || "") + "</span>" +
        '<span class="src__tag mono">' + escapeHtml(d.strategy || "") + "</span>";
      sourcesEl.appendChild(row);
    });
  }

  /* ================= crossed-line easter egg ================= */
  const crossed = $("#crossedNote");
  let doneCount = 0;
  function maybeCrossedLine() {
    doneCount += 1;
    if (doneCount >= 3 && Math.random() < 0.7) {
      crossed.classList.add("is-on");
      setTimeout(() => crossed.classList.remove("is-on"), 2400);
    }
  }

  /* ================= SSE consumers ================= */
  async function runTextQuery(text) {
    if (!text.trim()) return;
    clearOut();
    setStatus("dialing · streaming the call …");
    const start = performance.now();
    await consumeStream("/api/ask/stream", { text }, (evt) => handleEvent(evt, start));
  }

  function handleEvent(evt, start, isVoice) {
    const t = evt.type;
    if (t === "stage") {
      markStage(evt.stage, "active");
    } else if (t === "guard_result") {
      markStage("guard", "done");
      if (!evt.ok) setStatus("blocked: " + (evt.reasons || []).join(", "), true);
    } else if (t === "sources") {
      renderSources(evt.docs || []);
      markStage("retrieve", "done");
    } else if (t === "answer_start") {
      markStage("generate", "active");
    } else if (t === "chunk") {
      accumulatedAnswer += (evt.delta || "");
      setAnswer(accumulatedAnswer, false);
    } else if (t === "refuse") {
      setAnswer(evt.reason || "refused", true);
      markStage("gate", "done");
      lane.classList.remove("is-active");
    } else if (t === "fallback") {
      setStatus("llm out of service → extractive fallback (" + evt.reason + ")");
    } else if (t === "stt") {
      sttEl.textContent = "STT · " + evt.provider + " · " + evt.language + " · " + (evt.stt_ms || 0).toFixed(0) + " ms";
      $("#askInput").value = evt.transcript || "";
    } else if (t === "done") {
      const ms = Math.round(performance.now() - start);
      const r = evt;
      const ragMs = (r.total_ms !== undefined) ? r.total_ms : ms;
      if (r.mode === "refused") {
        if (r.answer) setAnswer(r.answer, true);
        else setAnswer(accumulatedAnswer, true);
        setStatus("call refused in " + ragMs + " ms — " + ((r.guardrails && r.guardrails.reject_code) || "not in corpus"), true);
        lane.classList.remove("is-active");
        maybeCrossedLine();
        if (isVoice) playVoice(r.answer || accumulatedAnswer);
        return;
      }
      if (r.answer && !accumulatedAnswer) setAnswer(r.answer, true);
      else setAnswer(accumulatedAnswer, true);

      metaEl.textContent =
        "mode: " + r.mode + " · grounded: " + r.grounded +
        " · RAG: " + ragMs + " ms" +
        (isVoice ? " (voice STT + net: " + ms + " ms)" : "") +
        " · " + (r.pipeline || []).join(" → ");
      laneLamps.forEach((l) => markStage(l.dataset.stageLine, "done"));
      lane.classList.remove("is-active");
      setStatus("patched through in " + ragMs + " ms (RAG pipeline)" + (isVoice ? " · STT voice call" : ""));
      if (r.sources) renderSources(r.sources);
      maybeCrossedLine();
      if (isVoice) playVoice(accumulatedAnswer);
    } else if (t === "error") {
      setStatus("error: " + (evt.message || "unknown"), true);
    }
  }

  async function consumeStream(url, body, onEvent) {
    try {
      const resp = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!resp.ok || !resp.body) { setStatus("HTTP " + resp.status, true); return; }
      const reader = resp.body.getReader();
      const dec = new TextDecoder();
      let buf = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        let i;
        while ((i = buf.indexOf("\n\n")) !== -1) {
          const raw = buf.slice(0, i);
          buf = buf.slice(i + 2);
          for (const line of raw.split("\n")) {
            if (line.startsWith("data:")) {
              try { onEvent(JSON.parse(line.slice(5).trim())); } catch (_) {}
            }
          }
        }
      }
    } catch (e) {
      setStatus("network error: " + e.message, true);
    }
  }

  /* ================= text form ================= */
  $("#askForm").addEventListener("submit", (e) => {
    e.preventDefault();
    runTextQuery($("#askInput").value);
  });
  $$(".console__chips button").forEach((b) => {
    b.addEventListener("click", () => {
      $("#askInput").value = b.dataset.q;
      runTextQuery(b.dataset.q);
    });
  });

  /* ================= microphone (hold to talk) ================= */
  const talkBtn = $("#talkBtn");
  let recorder = null, chunks = [], recording = false;

  function micLabel(txt) {
    talkBtn.querySelector(".mic-glyph").textContent = txt;
    const t = talkBtn.querySelector(".talk__txt");
    if (t) t.textContent = txt === "● REC" ? "RELEASE TO ASK" : "HOLD & SPEAK";
  }
  async function startRecording() {
    if (recording) return;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mime = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
        ? "audio/webm;codecs=opus"
        : MediaRecorder.isTypeSupported("audio/webm") ? "audio/webm" : "";
      recorder = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
      chunks = [];
      recorder.ondataavailable = (e) => e.data.size && chunks.push(e.data);
      recorder.onstop = () => {
        stream.getTracks().forEach((t) => t.stop());
        const type = recorder.mimeType || "audio/webm";
        const blob = new Blob(chunks, { type });
        if (blob.size > 4096) sendVoice(blob);
        else setStatus("too short — say something first");
      };
      recorder.start();
      recording = true;
      talkBtn.classList.add("is-rec");
      micLabel("● REC");
      setStatus("listening … release to ask");
    } catch (e) {
      setStatus("mic blocked: " + e.message, true);
    }
  }
  function stopRecording() {
    if (!recording || !recorder) return;
    recording = false;
    talkBtn.classList.remove("is-rec");
    micLabel("●");
    setStatus("processing audio …");
    try { recorder.stop(); } catch (_) {}
  }

  talkBtn.addEventListener("pointerdown", startRecording);
  talkBtn.addEventListener("pointerup", stopRecording);
  talkBtn.addEventListener("pointerleave", stopRecording);
  talkBtn.addEventListener("touchstart", (e) => e.preventDefault(), { passive: false });

  async function sendVoice(blob) {
    clearOut();
    const start = performance.now();
    setStatus("STT + RAG streaming …");
    const fd = new FormData();
    fd.append("file", blob, "voice.webm");
    try {
      const resp = await fetch("/api/voice/stream", { method: "POST", body: fd });
      if (!resp.ok || !resp.body) { setStatus("voice error HTTP " + resp.status, true); return; }
      const reader = resp.body.getReader();
      const dec = new TextDecoder();
      let buf = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        let i;
        while ((i = buf.indexOf("\n\n")) !== -1) {
          const raw = buf.slice(0, i);
          buf = buf.slice(i + 2);
          for (const line of raw.split("\n")) {
            if (line.startsWith("data:")) {
              try { handleEvent(JSON.parse(line.slice(5).trim()), start, true); } catch (_) {}
            }
          }
        }
      }
    } catch (e) {
      setStatus("voice network error: " + e.message, true);
    }
  }

  /* ================= kickoff ================= */
  loadIndexInfo();
  clearOut();
})();