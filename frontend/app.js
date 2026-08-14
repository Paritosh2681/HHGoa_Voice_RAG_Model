/* HH GOA Voice RAG — frontend wiring */
(function () {
  "use strict";

  const $ = (s) => document.querySelector(s);

  /* ---------------- reveal on scroll ---------------- */
  const io = new IntersectionObserver(
    (entries) => {
      for (const e of entries) {
        if (e.isIntersecting) {
          e.target.classList.add("in");
          io.unobserve(e.target);
        }
      }
    },
    { threshold: 0.12 }
  );
  document.querySelectorAll(".reveal").forEach((el) => io.observe(el));

  /* ---------------- index info -> strategy counts ---------------- */
  async function loadIndexInfo() {
    try {
      const r = await fetch("/api/index-info");
      const info = await r.json();
      const st = info.strategies || {};
      document.querySelectorAll("[data-strategy]").forEach((el) => {
        const key = el.dataset.strategy;
        el.textContent = key in st ? st[key] + " chunks" : "–";
      });
    } catch (_) {}
  }

  /* ---------------- metrics ---------------- */
  let barsDrawn = false;
  async function pollMetrics() {
    try {
      const r = await fetch("/api/metrics");
      const m = await r.json();
      const f = (ms) => (typeof ms === "number" ? ms.toFixed(1) + " ms" : "–");
      $("#mP50").textContent = f(m.p50_ms);
      $("#mP70").textContent = f(m.p70_ms);
      $("#mP100").textContent = f(m.p100_ms);
      $("#mN").textContent = m.total_requests ?? "–";
      const stages = m.by_stage || {};
      const box = $("#stageBars");
      const maxP = Math.max(
        1,
        ...Object.values(stages).map((s) => s.p70_ms || 0)
      );
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
      barsDrawn = true;
    } catch (_) {}
  }
  pollMetrics();
  setInterval(pollMetrics, 3000);

  /* ---------------- demo state ---------------- */
  const answerEl = $("#answer");
  const metaEl = $("#meta");
  const sttEl = $("#sttBadge");
  const sourcesEl = $("#sources");
  const statusEl = $("#status");
  const stageLines = document.querySelectorAll("[data-stage-line]");

  function setStatus(msg, isErr) {
    statusEl.textContent = msg;
    statusEl.classList.toggle("error", !!isErr);
  }
  function resetStage() {
    stageLines.forEach((l) => {
      l.classList.remove("stage__line--active", "stage__line--done");
      l.classList.add("stage__line--idle");
    });
  }
  function markStage(name, state) {
    stageLines.forEach((l) => {
      if (l.dataset.stageLine === name) {
        l.classList.remove("stage__line--active", "stage__line--done");
        if (state === "active") l.classList.add("stage__line--active");
        else if (state === "done") l.classList.add("stage__line--done");
      }
    });
  }
  function setAnswer(text, append) {
    const cursor = '<span class="answer__cursor"></span>';
    answerEl.innerHTML = escapeHtml(text) + cursor;
    if (append) answerEl.scrollTop = answerEl.scrollHeight;
  }
  function clearOut() {
    answerEl.innerHTML = '<span class="answer__cursor"></span>';
    metaEl.textContent = "—";
    sttEl.textContent = "—";
    sourcesEl.innerHTML = "";
    resetStage();
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
        '<span class="src__tag">' + escapeHtml(d.strategy || "") + "</span>";
      sourcesEl.appendChild(row);
    });
  }

  /* ---------------- SSE consumers ---------------- */
  async function runTextQuery(text) {
    if (!text.trim()) return;
    clearOut();
    setStatus("streaming …");
    const start = performance.now();
    await consumeStream("/api/ask/stream", { text }, (evt) => {
      handleEvent(evt, start);
    });
  }

  function handleEvent(evt, start) {
    const t = evt.type;
    if (t === "stage") {
      markStage(evt.stage, "active");
    } else if (t === "guard_result") {
      if (!evt.ok) {
        setStatus("blocked: " + (evt.reasons || []).join(", "), true);
        markStage("guard", "done");
      } else {
        markStage("guard", "done");
      }
    } else if (t === "sources") {
      renderSources(evt.docs || []);
      markStage("retrieve", "done");
    } else if (t === "answer_start") {
      markStage("generate", "active");
    } else if (t === "chunk") {
      if (answerEl.textContent === "" || answerEl.textContent === " ") {
        setAnswer("");
      }
      answerEl.lastChild &&
        answerEl.lastChild.nodeType === 3 &&
        answerEl.removeChild(answerEl.lastChild);
      const cur = answerEl.querySelector(".answer__cursor");
      if (cur) {
        const t = document.createTextNode(evt.delta || "");
        answerEl.insertBefore(t, cur);
      }
    } else if (t === "refuse") {
      setAnswer(evt.reason || "refused");
      markStage("gate", "done");
    } else if (t === "fallback") {
      setStatus("llm unavailable → extractive fallback (" + evt.reason + ")");
    } else if (t === "stt") {
      sttEl.textContent = "STT · " + evt.provider + " · " + evt.language +
        " · " + (evt.stt_ms || 0).toFixed(0) + " ms";
      $("#askInput").value = evt.transcript || "";
    } else if (t === "done") {
      const ms = Math.round(performance.now() - start);
      const r = evt;
      metaEl.textContent =
        "mode: " + r.mode + " · grounded: " + r.grounded +
        " · total: " + (r.total_ms || ms) + " ms" +
        " · " + (r.pipeline || []).join(" → ");
      stageLines.forEach((l) => markStage(l.dataset.stageLine, "done"));
      setStatus("done in " + ms + " ms");
      if (r.sources) renderSources(r.sources);
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
      if (!resp.ok || !resp.body) {
        setStatus("HTTP " + resp.status, true);
        return;
      }
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
              try {
                onEvent(JSON.parse(line.slice(5).trim()));
              } catch (_) {}
            }
          }
        }
      }
    } catch (e) {
      setStatus("network error: " + e.message, true);
    }
  }

  /* ---------------- text form ---------------- */
  const form = $("#askForm");
  form.addEventListener("submit", (e) => {
    e.preventDefault();
    runTextQuery($("#askInput").value);
  });

  /* sample chips */
  document.querySelectorAll(".demo__chips button").forEach((b) => {
    b.addEventListener("click", () => {
      $("#askInput").value = b.dataset.q;
      runTextQuery(b.dataset.q);
    });
  });

  /* ---------------- microphone (hold to record) ---------------- */
  const micBtn = $("#micBtn");
  let recorder = null;
  let chunks = [];
  let recording = false;

  function micLabel(txt) {
    micBtn.querySelector(".mic-glyph").textContent = txt;
  }
  async function startRecording() {
    if (recording) return;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mime = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
        ? "audio/webm;codecs=opus"
        : MediaRecorder.isTypeSupported("audio/webm")
        ? "audio/webm"
        : "";
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
      micBtn.style.background = "#FF0080";
      micLabel("● REC");
      setStatus("listening … release to ask");
    } catch (e) {
      setStatus("mic blocked: " + e.message, true);
    }
  }
  function stopRecording() {
    if (!recording || !recorder) return;
    recording = false;
    micBtn.style.background = "";
    micLabel("●");
    setStatus("processing audio …");
    try {
      recorder.stop();
    } catch (_) {}
  }

  micBtn.addEventListener("pointerdown", startRecording);
  micBtn.addEventListener("pointerup", stopRecording);
  micBtn.addEventListener("pointerleave", stopRecording);
  micBtn.addEventListener("touchstart", (e) => e.preventDefault(), { passive: false });

  async function sendVoice(blob) {
    clearOut();
    const start = performance.now();
    setStatus("STT + RAG streaming …");
    const fd = new FormData();
    fd.append("file", blob, "voice.webm");
    try {
      const resp = await fetch("/api/voice/stream", { method: "POST", body: fd });
      if (!resp.ok || !resp.body) {
        setStatus("voice error HTTP " + resp.status, true);
        return;
      }
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
              try {
                handleEvent(JSON.parse(line.slice(5).trim()), start);
              } catch (_) {}
            }
          }
        }
      }
    } catch (e) {
      setStatus("voice network error: " + e.message, true);
    }
  }

  /* ---------------- kickoff ---------------- */
  loadIndexInfo();
  clearOut();
})();
