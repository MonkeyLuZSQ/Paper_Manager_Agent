const state = {
  busy: false,
};

const els = {
  runtimeLine: document.querySelector("#runtimeLine"),
  refreshBtn: document.querySelector("#refreshBtn"),
  indexBtn: document.querySelector("#indexBtn"),
  paperList: document.querySelector("#paperList"),
  outputList: document.querySelector("#outputList"),
  chunkCount: document.querySelector("#chunkCount"),
  embeddingState: document.querySelector("#embeddingState"),
  activePaper: document.querySelector("#activePaper"),
  messageInput: document.querySelector("#messageInput"),
  sendBtn: document.querySelector("#sendBtn"),
  chatLog: document.querySelector("#chatLog"),
  reportPreview: document.querySelector("#reportPreview"),
};

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const data = await response.json();
  if (!response.ok || data.error) {
    throw new Error(data.error || `HTTP ${response.status}`);
  }
  return data;
}

function setBusy(value) {
  state.busy = value;
  els.sendBtn.disabled = value;
  els.indexBtn.disabled = value;
  els.refreshBtn.disabled = value;
}

function itemButton(label, onClick) {
  const button = document.createElement("button");
  button.className = "list-item";
  button.type = "button";
  button.textContent = label;
  button.addEventListener("click", onClick);
  return button;
}

function renderList(container, items, onClick) {
  container.innerHTML = "";
  if (!items.length) {
    const empty = document.createElement("div");
    empty.className = "empty";
    empty.textContent = "No records.";
    container.appendChild(empty);
    return;
  }
  for (const item of items) {
    container.appendChild(itemButton(item, () => onClick(item)));
  }
}

function addMessage(role, content) {
  const article = document.createElement("article");
  article.className = `message ${role === "User" ? "user" : "agent"}`;

  const label = document.createElement("div");
  label.className = "message-role";
  label.textContent = role === "Agent" ? "小智" : role;

  const body = document.createElement("div");
  body.className = "message-body";
  body.textContent = content;

  article.append(label, body);
  els.chatLog.appendChild(article);
  els.chatLog.scrollTop = els.chatLog.scrollHeight;
}

async function refreshState() {
  const data = await api("/api/state");
  els.runtimeLine.textContent = `${data.model} · ${data.base_url}`;
  els.chunkCount.textContent = String(data.indexed_chunks || 0);
  els.activePaper.textContent = data.active_paper || "-";

  if (data.embedding && data.embedding.model) {
    els.embeddingState.innerHTML = `<span class="status-ok">${data.embedding.backend}</span> · ${data.embedding.chunks}`;
  } else {
    els.embeddingState.textContent = "-";
  }

  renderList(els.paperList, data.papers || [], (name) => {
    els.messageInput.value = `总结 ${name}`;
    els.messageInput.focus();
  });
  renderList(els.outputList, data.outputs || [], loadOutput);
}

async function loadOutput(name) {
  try {
    const data = await api(`/api/output?name=${encodeURIComponent(name)}`);
    els.reportPreview.textContent = data.content || "";
  } catch (error) {
    els.reportPreview.textContent = error.message;
  }
}

async function sendMessage() {
  const message = els.messageInput.value.trim();
  if (!message || state.busy) {
    return;
  }
  els.messageInput.value = "";
  addMessage("User", message);
  addMessage("Agent", "Working...");
  const pending = els.chatLog.lastElementChild.querySelector(".message-body");

  try {
    setBusy(true);
    const data = await api("/api/chat", {
      method: "POST",
      body: JSON.stringify({ message }),
    });
    pending.textContent = data.answer || "";
    await refreshState();
  } catch (error) {
    pending.textContent = error.message;
  } finally {
    setBusy(false);
    els.messageInput.focus();
  }
}

async function rebuildIndex() {
  if (state.busy) {
    return;
  }
  try {
    setBusy(true);
    addMessage("Agent", "Indexing...");
    const data = await api("/api/index", { method: "POST", body: "{}" });
    els.chatLog.lastElementChild.querySelector(".message-body").textContent = `Indexed ${data.indexed_chunk_count} chunk(s).`;
    await refreshState();
  } catch (error) {
    addMessage("Agent", error.message);
  } finally {
    setBusy(false);
  }
}

els.sendBtn.addEventListener("click", sendMessage);
els.indexBtn.addEventListener("click", rebuildIndex);
els.refreshBtn.addEventListener("click", refreshState);
els.messageInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    sendMessage();
  }
});

refreshState().catch((error) => addMessage("Agent", error.message));
