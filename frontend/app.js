(() => {
  "use strict";

  const MAX_BYTES = 15 * 1024 * 1024;
  const TYPES = ["image/jpeg", "image/png", "image/webp"];
  const MAX_CANVAS = 4096; // cap canvas size so very large photos don't exhaust phone memory
  // One fixed color per class (colors are also used in the legend).
  const CLASSES = [
    { label: "Class I", color: "#E53935" },
    { label: "Class II", color: "#FB8C00" },
    { label: "Class III", color: "#FDD835" },
    { label: "Class IV", color: "#43A047" },
    { label: "Class V", color: "#1E88E5" },
    { label: "Class VI", color: "#8E24AA" },
  ];
  const COLOR = Object.fromEntries(CLASSES.map(c => [c.label, c.color]));

  const $ = id => document.getElementById(id);
  const el = {
    file: $("file"), drop: $("drop"), previewWrap: $("preview-wrap"), preview: $("preview"),
    analyze: $("analyze"), spinner: $("spinner"), error: $("error"), results: $("results"),
    conf: $("conf"), confOut: $("conf-out"), legend: $("legend"), canvas: $("canvas"),
    download: $("download"), empty: $("empty"), list: $("list"),
  };
  const ctx = el.canvas.getContext("2d");

  let file = null;       // chosen File
  let objectUrl = null;  // URL for preview and canvas source
  let img = null;        // decoded HTMLImageElement (browser applies EXIF orientation)
  let detections = [];   // as returned by the server (normalized coordinates)
  const enabled = Object.fromEntries(CLASSES.map(c => [c.label, true]));
  const countEls = {};

  // ---- legend with checkboxes and counts ----
  CLASSES.forEach((c, i) => {
    const label = document.createElement("label");
    const cb = document.createElement("input");
    cb.type = "checkbox"; cb.checked = true; cb.id = "cls-" + i;
    cb.addEventListener("change", () => { enabled[c.label] = cb.checked; render(); });
    const sw = document.createElement("span");
    sw.className = "swatch"; sw.style.background = c.color; sw.setAttribute("aria-hidden", "true");
    const txt = document.createElement("span");
    txt.textContent = c.label + " (";
    const count = document.createElement("span");
    count.textContent = "0";
    countEls[c.label] = count;
    txt.append(count, ")");
    label.append(cb, sw, txt);
    el.legend.appendChild(label);
  });

  // ---- helpers ----
  function showError(msg) { el.error.textContent = msg; el.error.hidden = !msg; }

  // Pick black or white text for readable contrast on a given tag color.
  function textOn(hex) {
    const n = parseInt(hex.slice(1), 16);
    const lum = 0.2126 * (n >> 16) + 0.7152 * ((n >> 8) & 255) + 0.0722 * (n & 255);
    return lum > 150 ? "#000" : "#fff";
  }

  function rgba(hex, a) {
    const n = parseInt(hex.slice(1), 16);
    return `rgba(${n >> 16},${(n >> 8) & 255},${n & 255},${a})`;
  }

  // ---- file selection ----
  function setFile(f) {
    showError("");
    if (!f) return;
    if (!TYPES.includes(f.type)) { showError("Wrong file type. Please choose a JPG, PNG or WEBP image."); return; }
    if (f.size > MAX_BYTES) { showError("File too large. Maximum size is 15 MB."); return; }
    if (objectUrl) URL.revokeObjectURL(objectUrl);
    objectUrl = URL.createObjectURL(f);
    const im = new Image();
    im.onload = () => {
      file = f; img = im; detections = [];
      el.preview.src = objectUrl;
      el.previewWrap.hidden = false;
      el.results.hidden = true;
      el.analyze.disabled = false;
    };
    im.onerror = () => { file = null; img = null; el.analyze.disabled = true; showError("This image could not be opened. Try another file."); };
    im.src = objectUrl;
  }

  el.file.addEventListener("change", () => setFile(el.file.files[0]));
  ["dragenter", "dragover"].forEach(ev => el.drop.addEventListener(ev, e => { e.preventDefault(); el.drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach(ev => el.drop.addEventListener(ev, e => { e.preventDefault(); el.drop.classList.remove("over"); }));
  el.drop.addEventListener("drop", e => setFile(e.dataTransfer.files[0]));

  // ---- analyze ----
  el.analyze.addEventListener("click", async () => {
    if (!file) return;
    showError("");
    el.analyze.disabled = true; el.spinner.hidden = false;
    try {
      const fd = new FormData();
      fd.append("file", file);
      const res = await fetch("/predict", { method: "POST", body: fd });
      let data = null;
      try { data = await res.json(); } catch (_) { /* non-JSON error body */ }
      if (!res.ok) {
        const msg = data && data.error ? data.error : "Server error (" + res.status + "). Please try again.";
        throw new Error(msg);
      }
      detections = data.detections || [];
      el.results.hidden = false;
      render();
      el.results.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (e) {
      showError(e instanceof TypeError ? "Network error: could not reach the server." : e.message);
    } finally {
      el.spinner.hidden = true; el.analyze.disabled = false;
    }
  });

  // ---- drawing ----
  function draw(list) {
    const s = Math.min(1, MAX_CANVAS / Math.max(img.naturalWidth, img.naturalHeight));
    const W = Math.round(img.naturalWidth * s), H = Math.round(img.naturalHeight * s);
    el.canvas.width = W; el.canvas.height = H;
    ctx.drawImage(img, 0, 0, W, H);

    // Line width and font scale with canvas size so labels stay readable on big photos.
    const lw = Math.max(2, W / 500), fs = Math.max(12, Math.round(W / 60));
    ctx.lineWidth = lw; ctx.lineJoin = "round";
    ctx.font = `600 ${fs}px system-ui, sans-serif`; ctx.textBaseline = "middle";

    for (const d of list) {
      const color = COLOR[d.label];
      const [bx1, by1, bx2, by2] = d.box;
      ctx.fillStyle = rgba(color, 0.35); ctx.strokeStyle = color;
      if (d.polygon && d.polygon.length >= 3) {
        ctx.beginPath();
        d.polygon.forEach(([x, y], i) => i ? ctx.lineTo(x * W, y * H) : ctx.moveTo(x * W, y * H));
        ctx.closePath(); ctx.fill(); ctx.stroke();
      } else { // fallback: rectangle
        ctx.fillRect(bx1 * W, by1 * H, (bx2 - bx1) * W, (by2 - by1) * H);
        ctx.strokeRect(bx1 * W, by1 * H, (bx2 - bx1) * W, (by2 - by1) * H);
      }
      // Label tag above the shape (inside it if there is no room above), kept within the canvas.
      const text = `${d.label} ${Math.round(d.conf * 100)}%`;
      const pad = fs * 0.35, tw = ctx.measureText(text).width + pad * 2, th = fs + pad * 2;
      const tx = Math.min(Math.max(0, bx1 * W), W - tw);
      const ty = by1 * H - th >= 0 ? by1 * H - th : by1 * H;
      ctx.fillStyle = color; ctx.fillRect(tx, ty, tw, th);
      ctx.fillStyle = textOn(color); ctx.fillText(text, tx + pad, ty + th / 2);
    }
    el.canvas.setAttribute("aria-label",
      `Dental photo with ${list.length} caries detection${list.length === 1 ? "" : "s"} highlighted`);
  }

  // Re-filter the stored detections locally; no server call.
  function render() {
    if (!img) return;
    const th = parseFloat(el.conf.value);
    el.confOut.textContent = th.toFixed(2);
    const passing = detections.filter(d => d.conf >= th);
    const shown = passing.filter(d => enabled[d.label]);

    CLASSES.forEach(c => { countEls[c.label].textContent = passing.filter(d => d.label === c.label).length; });
    draw(shown);

    el.list.replaceChildren();
    for (const d of shown) {
      const li = document.createElement("li");
      const sw = document.createElement("span");
      sw.className = "swatch"; sw.style.background = COLOR[d.label]; sw.setAttribute("aria-hidden", "true");
      li.append(sw, `${d.label}: ${(d.conf * 100).toFixed(0)}% confidence`);
      el.list.appendChild(li);
    }
    el.empty.hidden = shown.length > 0;
    el.empty.textContent = passing.length === 0
      ? "No caries detected at this confidence"
      : "All detected classes are currently hidden.";
    el.download.disabled = false;
  }

  el.conf.addEventListener("input", render);

  el.download.addEventListener("click", () => {
    el.canvas.toBlob(blob => {
      if (!blob) { showError("Could not export the image."); return; }
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob); a.download = "caries-labeled.png";
      document.body.appendChild(a); a.click(); a.remove();
      setTimeout(() => URL.revokeObjectURL(a.href), 1000);
    }, "image/png");
  });
})();