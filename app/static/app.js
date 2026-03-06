const searchWrapper = document.getElementById("searchWrapper");
const searchInput = document.getElementById("searchInput");
const searchBtn = document.getElementById("searchBtn");
const loadingBar = document.getElementById("loadingBar");
const resultArea = document.getElementById("resultArea");
const footer = document.getElementById("footer");

let hasResult = false;

function setLoading(show) {
  loadingBar.classList.toggle("active", show);
  searchBtn.disabled = show;
  searchInput.disabled = show;
}

function showResult(query, text, isError = false) {
  const existing = resultArea.querySelector(".result-card");
  if (existing) {
    existing.classList.add("fade-out");
    setTimeout(() => renderCard(query, text, isError), 280);
  } else {
    renderCard(query, text, isError);
  }
}

function renderCard(query, text, isError) {
  resultArea.innerHTML = "";

  const card = document.createElement("div");
  card.className = "result-card";

  const queryLabel = document.createElement("div");
  queryLabel.className = "result-query";
  queryLabel.innerHTML = `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg> <strong>${escapeHtml(
    query,
  )}</strong>`;

  const body = document.createElement("div");
  body.className = "result-body" + (isError ? " error-text" : "");

  card.appendChild(queryLabel);
  card.appendChild(body);
  resultArea.appendChild(card);

  typewrite(body, text);
  footer.style.display = "block";
}

function typewrite(el, text) {
  let i = 0;
  el.classList.add("typing");
  const speed = Math.max(5, Math.min(20, 2000 / text.length));

  function tick() {
    if (i < text.length) {
      const chunk = text.slice(i, i + 2);
      el.textContent += chunk;
      i += chunk.length;
      setTimeout(tick, speed);
    } else {
      el.classList.remove("typing");
    }
  }
  tick();
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

async function doSearch() {
  const query = searchInput.value.trim();
  if (!query) return;

  if (!hasResult) {
    searchWrapper.classList.add("has-result");
    hasResult = true;
  }

  setLoading(true);

  try {
    const response = await fetch("/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    });

    const data = await response.json();
    setLoading(false);

    if (data.success) {
      showResult(query, data.response);
    } else {
      showResult(query, `Error: ${data.error}`, true);
    }
  } catch (error) {
    setLoading(false);
    showResult(query, `Network Error: ${error.message}`, true);
  }

  searchInput.focus();
}

searchBtn.addEventListener("click", doSearch);
searchInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") doSearch();
});

searchInput.focus();
