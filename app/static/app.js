const searchWrapper = document.getElementById("searchWrapper");
const searchInput = document.getElementById("searchInput");
const searchBtn = document.getElementById("searchBtn");
const uploadBtn = document.getElementById("uploadBtn");
const imageInput = document.getElementById("imageInput");
const imagePreview = document.getElementById("imagePreview");
const loadingBar = document.getElementById("loadingBar");
const resultArea = document.getElementById("resultArea");
const resultContent = document.getElementById("resultContent");
const progressStepper = document.getElementById("progressStepper");
const statusMessage = document.getElementById("statusMessage");
const footer = document.getElementById("footer");

let hasResult = false;
let currentAbortController = null;
let selectedFile = null;

function setLoading(show) {
  loadingBar.classList.toggle("active", show);
  searchBtn.disabled = show;
  searchInput.disabled = show;
  uploadBtn.disabled = show;
}

function resetUI() {
  resultContent.innerHTML = "";
  progressStepper.classList.remove("done");
  progressStepper.style.display = "block";
  statusMessage.textContent = "Menunggu...";
}

// Image handling
uploadBtn.addEventListener("click", () => imageInput.click());

imageInput.addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (file) {
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
});

window.removeImage = function() {
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

  // Laws Section (If any)
  if (data.laws_summary) {
    const title = document.createElement("div");
    title.className = "result-section-title";
    title.textContent = "Pasal & Hukum Terkait";
    card.appendChild(title);

    const panel = document.createElement("div");
    panel.className = "laws-panel";
    panel.innerHTML = processMarkdown(data.laws_summary);
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
    if (data.fact_check.verified === true) {
      statusText.innerHTML = "<strong>Status:</strong> Terverifikasi Fakta";
      statusText.style.color = "#4ade80";
    } else if (data.fact_check.verified === false) {
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
    error: "Error"
  };
  return map[stage] || stage;
}

async function doAnalyze() {
  const content = searchInput.value.trim();
  if (!content && !selectedFile) return;

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
              const stageLabel = `<span class="stage-label">${mapStage(stage)}</span>`;
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

