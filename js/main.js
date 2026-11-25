/* AnnotationLens — main.js */

// ── Sidebar toggle (mobile) ──────────────────
function toggleSidebar() {
  document.getElementById("sidebar").classList.toggle("open");
}

// ── Status indicator ─────────────────────────
async function updateStatus() {
  const dot  = document.getElementById("status-dot");
  const text = document.getElementById("status-text");
  if (!dot || !text) return;
  try {
    const data = await fetch("/api/status").then(r => r.json());
    if (data.loaded) {
      dot.className  = "status-dot loaded";
      text.textContent = `${data.stats.n_matched} responses loaded`;
    } else {
      dot.className  = "status-dot ready";
      text.textContent = "No data loaded";
    }
  } catch {
    dot.className  = "status-dot";
    text.textContent = "Server not reachable";
  }
}

// Run on every page load
updateStatus();
