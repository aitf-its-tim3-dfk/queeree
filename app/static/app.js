const searchWrapper = document.getElementById("searchWrapper");
const searchInput = document.getElementById("searchInput");
const searchBtn = document.getElementById("searchBtn");
const uploadBtn = document.getElementById("uploadBtn");
const imageInput = document.getElementById("imageInput");
const imagePreview = document.getElementById("imagePreview");
const settingsToggleBtn = document.getElementById("settingsToggleBtn");
const settingsPanel = document.getElementById("settingsPanel");
const loadingBar = document.getElementById("loadingBar");
const resultArea = document.getElementById("resultArea");
const resultContent = document.getElementById("resultContent");
const progressStepper = document.getElementById("progressStepper");
const statusMessage = document.getElementById("statusMessage");
const footer = document.getElementById("footer");
const urlToggleBtn = document.getElementById("urlToggleBtn");
const mediaUrlRow = document.getElementById("mediaUrlRow");
const mediaUrlInput = document.getElementById("mediaUrlInput");
const mediaUrlClear = document.getElementById("mediaUrlClear");

let hasResult = false;
let currentAbortController = null;
let selectedFile = null;

function setLoading(show) {
  loadingBar.classList.toggle("active", show);
  searchBtn.disabled = show;
  searchInput.disabled = show;
  uploadBtn.disabled = show;
  settingsToggleBtn.disabled = show;
}

function resetUI() {
  resultContent.innerHTML = "";
  progressStepper.classList.remove("done");
  progressStepper.style.display = "block";
  statusMessage.textContent = "Menunggu...";
}

// Image handling
uploadBtn.addEventListener("click", () => imageInput.click());

// Settings toggling
settingsToggleBtn.addEventListener("click", () => {
  if (settingsPanel.style.display === "none") {
    settingsPanel.style.display = "block";
    settingsToggleBtn.classList.add("active");
  } else {
    settingsPanel.style.display = "none";
    settingsToggleBtn.classList.remove("active");
  }
});

// URL toggle
urlToggleBtn.addEventListener("click", () => {
  if (mediaUrlRow.style.display === "none") {
    mediaUrlRow.style.display = "flex";
    urlToggleBtn.classList.add("active");
    mediaUrlInput.focus();
  } else {
    mediaUrlRow.style.display = "none";
    urlToggleBtn.classList.remove("active");
    mediaUrlInput.value = "";
  }
});

mediaUrlClear.addEventListener("click", () => {
  mediaUrlInput.value = "";
  mediaUrlInput.focus();
});

imageInput.addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (file) {
    showImagePreview(file);
  }
});

// Clipboard paste support — paste images directly into the textarea
searchInput.addEventListener("paste", (e) => {
  const items = e.clipboardData?.items;
  if (!items) return;
  for (const item of items) {
    if (item.type.startsWith("image/")) {
      e.preventDefault();
      const file = item.getAsFile();
      if (file) showImagePreview(file);
      return;
    }
  }
});

function showImagePreview(file) {
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = (re) => {
    imagePreview.style.display = "block";
    imagePreview.innerHTML = `
      <img src="${re.target.result}" alt="Preview">
      <button class="remove-img-btn" onclick="removeImage()">✕</button>
    `;
  };
  reader.readAsDataURL(file);
}

window.removeImage = function () {
  selectedFile = null;
  imageInput.value = "";
  imagePreview.style.display = "none";
  imagePreview.innerHTML = "";
};

function renderBadge(text, isDanger) {
  return `<span class="badge ${isDanger ? "danger" : "safe"}">${escapeHtml(
    text,
  )}</span>`;
}

function processMarkdown(text) {
  // Ultra-simple markdown -> html for laws response
  return escapeHtml(text)
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*(.*?)\*/g, "<em>$1</em>")
    .replace(/\n\n/g, "<br><br>")
    .replace(/\n- /g, "<br>• ");
}

function renderFinalResult(data) {
  progressStepper.classList.add("done");
  statusMessage.textContent = "Analisis selesai";

  if (!data) return;

  const card = document.createElement("div");
  card.className = "result-card";

  // Header & Badges
  const header = document.createElement("div");
  header.className = "moderation-header";

  const badgesContainer = document.createElement("div");
  badgesContainer.className = "badges-container";

  if (!data.is_flagged || !data.categories || data.categories.length === 0) {
    badgesContainer.innerHTML = renderBadge("Aman (No Violation)", false);
  } else {
    data.categories.forEach((cat) => {
      badgesContainer.innerHTML += renderBadge(cat, true);
    });
  }

  header.appendChild(badgesContainer);
  card.appendChild(header);

  // Final Summary Section (If any)
  if (data.final_summary) {
    const title = document.createElement("div");
    title.className = "result-section-title";
    title.textContent = "Ringkasan Eksekutif";
    card.appendChild(title);

    const panel = document.createElement("div");
    panel.style.padding = "1rem";
    panel.style.backgroundColor = "rgba(100, 116, 139, 0.05)";
    panel.style.borderRadius = "0.75rem";
    panel.style.marginBottom = "1.5rem";
    panel.innerHTML = `<p>${processMarkdown(data.final_summary)}</p>`;
    card.appendChild(panel);
  }

  // Analyzed Text Section with Highlights
  const contentInput = document.getElementById("searchInput").value.trim();
  if (contentInput && data.law_analysis && data.law_analysis.length > 0) {
    const title = document.createElement("div");
    title.className = "result-section-title";
    title.textContent = "Sorotan Bukti Teks";
    card.appendChild(title);

    // Aggregate all segments
    let allSegments = [];
    data.law_analysis.forEach((law) => {
      if (law.segments) {
        law.segments.forEach((seg) => {
          allSegments.push({
            start: seg.start,
            end: seg.end,
            score: seg.score,
            reason: seg.reason,
            pasal: law.pasal,
          });
        });
      }
    });

    const textPanel = document.createElement("div");
    textPanel.className = "analyzed-text-panel";

    if (allSegments.length === 0) {
      textPanel.textContent = contentInput;
    } else {
      // Create non-overlapping boundaries
      let boundaries = new Set([0, contentInput.length]);
      allSegments.forEach((seg) => {
        boundaries.add(seg.start);
        boundaries.add(seg.end);
      });
      let sortedBounds = Array.from(boundaries).sort((a, b) => a - b);

      let html = "";
      for (let i = 0; i < sortedBounds.length - 1; i++) {
        let spanStart = sortedBounds[i];
        let spanEnd = sortedBounds[i + 1];
        let textChunk = contentInput.substring(spanStart, spanEnd);

        // Find segments covering this chunk
        let coveringSegs = allSegments.filter(
          (s) => s.start <= spanStart && s.end >= spanEnd,
        );
        if (coveringSegs.length > 0) {
          // Take the highest score
          let maxScore = Math.max(...coveringSegs.map((s) => s.score));
          let bestSeg = coveringSegs.find((s) => s.score === maxScore);
          let tooltip = `Pasal: ${bestSeg.pasal}\nCertainty: ${bestSeg.score}/N\nReason: ${bestSeg.reason}`;
          html += `<span class="highlight-segment" data-score="${maxScore}" data-law-tooltip="${escapeHtml(
            tooltip,
          )}">${escapeHtml(textChunk)}</span>`;
        } else {
          html += escapeHtml(textChunk);
        }
      }
      textPanel.innerHTML = html;
    }
    card.appendChild(textPanel);
  }

  // Laws Section (If any)
  if (data.laws_summary) {
    const title = document.createElement("div");
    title.className = "result-section-title";
    title.textContent = "Pasal & Hukum Terkait";
    card.appendChild(title);

    const panel = document.createElement("div");
    panel.className = "laws-panel";

    // Check if we have detailed law_analysis
    if (data.law_analysis && data.law_analysis.length > 0) {
      data.law_analysis.forEach((law) => {
        const item = document.createElement("div");
        item.className = "law-analysis-item";

        let htmlContext = `<h4>${escapeHtml(law.pasal)}</h4>`;
        htmlContext += `<p class="overall-reason">${escapeHtml(
          law.overall_reason || "",
        )}</p>`;

        if (
          law.clustered_reason_counts &&
          Object.keys(law.clustered_reason_counts).length > 0
        ) {
          htmlContext += `<ul class="image-reasons-list">`;
          Object.entries(law.clustered_reason_counts).forEach(([r, count]) => {
            htmlContext += `<li>${escapeHtml(
              r,
            )} <strong>(votes: ${count})</strong></li>`;
          });
          htmlContext += `</ul>`;
        }
        item.innerHTML = htmlContext;
        panel.appendChild(item);
      });
    } else {
      panel.innerHTML = processMarkdown(data.laws_summary);
    }

    card.appendChild(panel);
  }

  // Fact Check Section (If any)
  if (data.fact_check) {
    const title = document.createElement("div");
    title.className = "result-section-title";
    title.textContent = "Verifikasi Fakta";
    card.appendChild(title);

    const panel = document.createElement("div");
    panel.className = "fact-check-panel";

    const statusText = document.createElement("p");
    if (data.fact_check.status?.toLowerCase() === "true") {
      statusText.innerHTML = "<strong>Status:</strong> Terverifikasi Fakta";
      statusText.style.color = "#4ade80";
    } else if (data.fact_check.status?.toLowerCase() === "false") {
      statusText.innerHTML = "<strong>Status:</strong> Terbukti Hoaks / Salah";
      statusText.style.color = "var(--red)";
    } else {
      statusText.innerHTML = "<strong>Status:</strong> Bukti Tidak Cukup";
      statusText.style.color = "var(--text-secondary)";
    }

    const reasoning = document.createElement("p");
    reasoning.style.marginTop = "0.5rem";
    reasoning.textContent = data.fact_check.reasoning;

    panel.appendChild(statusText);
    panel.appendChild(reasoning);

    if (data.fact_check.sources && data.fact_check.sources.length > 0) {
      const srcTitle = document.createElement("div");
      srcTitle.style.marginTop = "1rem";
      srcTitle.style.fontWeight = "600";
      srcTitle.textContent = "Sumber Referensi:";
      panel.appendChild(srcTitle);

      data.fact_check.sources.forEach((src) => {
        const item = document.createElement("div");
        item.className = "source-item";
        item.innerHTML = `
          <a href="${escapeHtml(
            src.url,
          )}" target="_blank" rel="noopener noreferrer">${escapeHtml(
          src.title,
        )}</a>
          <p>${escapeHtml(src.description).substring(0, 150)}...</p>
        `;
        panel.appendChild(item);
      });
    }
    card.appendChild(panel);
  }

  resultContent.appendChild(card);
  footer.style.display = "block";
}

function escapeHtml(str) {
  if (!str) return "";
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

function mapStage(stage) {
  const map = {
    classifying: "Klasifikasi",
    fact_checking: "Cek Fakta",
    law_retrieval: "Yurisprudensi",
    processing: "Analisis",
    done: "Selesai",
    error: "Error",
  };
  return map[stage] || stage;
}

async function doAnalyze() {
  const content = searchInput.value.trim();
  const mediaUrl = mediaUrlInput.value.trim();
  if (!content && !selectedFile && !mediaUrl) return;

  if (currentAbortController) {
    currentAbortController.abort();
  }
  currentAbortController = new AbortController();

  if (!hasResult) {
    searchWrapper.classList.add("has-result");
    resultArea.style.display = "block";
    hasResult = true;
  }

  setLoading(true);
  resetUI();

  try {
    const formData = new FormData();
    formData.append("content", content);
    if (selectedFile) {
      formData.append("image", selectedFile);
    }

    // Include media URL if provided
    if (mediaUrl && !selectedFile) {
      formData.append("media_url", mediaUrl);
    }

    // Read config overrides
    const configData = {};
    const cfgClassifierModel = document
      .getElementById("cfgClassifierModel")
      .value.trim();
    if (cfgClassifierModel)
      configData.classifier_model_name = cfgClassifierModel;

    const cfgFactCheckerModel = document
      .getElementById("cfgFactCheckerModel")
      .value.trim();
    if (cfgFactCheckerModel)
      configData.fact_checker_model_name = cfgFactCheckerModel;

    const cfgLawRetrieverModel = document
      .getElementById("cfgLawRetrieverModel")
      .value.trim();
    if (cfgLawRetrieverModel)
      configData.law_retriever_model_name = cfgLawRetrieverModel;

    const cfgFactLoops = document.getElementById("cfgFactLoops").value;
    if (cfgFactLoops)
      configData.fact_checker_max_loops = parseInt(cfgFactLoops);

    const cfgClassifierSamples = document.getElementById(
      "cfgClassifierSamples",
    ).value;
    if (cfgClassifierSamples)
      configData.classifier_n_samples = parseInt(cfgClassifierSamples);

    const cfgFactSamples = document.getElementById("cfgFactSamples").value;
    if (cfgFactSamples)
      configData.fact_checker_n_samples = parseInt(cfgFactSamples);

    if (Object.keys(configData).length > 0) {
      formData.append("config", JSON.stringify(configData));
    }

    const response = await fetch("/api/analyze", {
      method: "POST",
      body: formData,
      signal: currentAbortController.signal,
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n\n");
      buffer = lines.pop(); // Keep incomplete chunk

      for (const line of lines) {
        if (line.startsWith("data: ")) {
          try {
            const payload = JSON.parse(line.substring(6));
            if (payload.type === "progress") {
              const { stage, message } = payload.data;
              const stageLabel = `<span class="stage-label">${mapStage(
                stage,
              )}</span>`;
              statusMessage.innerHTML = `${stageLabel} ${escapeHtml(message)}`;
            } else if (payload.type === "result") {
              progressStepper.style.display = "none";
              renderFinalResult(payload.data);
            } else if (payload.type === "error") {
              statusMessage.textContent = "Error: " + payload.data;
              progressStepper.classList.add("done");
              progressStepper.style.color = "var(--red)";
            }
          } catch (e) {
            console.error("Failed to parse SSE", e);
          }
        }
      }
    }
  } catch (error) {
    if (error.name !== "AbortError") {
      statusMessage.textContent = `Network Error: ${error.message}`;
      progressStepper.classList.add("done");
      progressStepper.style.color = "var(--red)";
    }
  } finally {
    setLoading(false);
  }
}

searchBtn.addEventListener("click", doAnalyze);
searchInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) doAnalyze();
});

searchInput.focus();
