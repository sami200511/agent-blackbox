const log = document.getElementById("log");
const form = document.getElementById("form");
const q = document.getElementById("q");
const send = document.getElementById("send");
const modelSel = document.getElementById("model");
const customModel = document.getElementById("customModel");
const rerankPill = document.getElementById("rerankPill");
const keysPill = document.getElementById("keysPill");
const keyModal = document.getElementById("keyModal");
const keyText = document.getElementById("keyText");
const keyInput = document.getElementById("keyInput");

let pendingQuestion = null;   // retried after a key is saved
let pendingEnvVar = null;

async function loadModels() {
  const r = await fetch("/api/models");
  const data = await r.json();
  modelSel.innerHTML = "";
  for (const m of data.models) {
    const opt = document.createElement("option");
    opt.value = m.id;
    const hasKey = data.keys[m.provider];
    opt.textContent = (hasKey ? "" : "🔑 ") + m.label;
    modelSel.appendChild(opt);
  }
  rerankPill.textContent = data.rerank.enabled ? "RERANK: ON" : "RERANK: OFF";
  rerankPill.className = "pill " + (data.rerank.enabled ? "on" : "off");
  const okKeys = Object.entries(data.keys).filter(([,v]) => v).map(([k]) => k);
  keysPill.textContent = "KEYS: " + (okKeys.join(", ") || "none");
  modelSel.onchange = () =>
    customModel.style.display = modelSel.value === "__custom__" ? "block" : "none";
  modelSel.onchange();
}

function add(cls, text) {
  const row = document.createElement("div");
  row.className = "msg " + cls;
  const b = document.createElement("div");
  b.className = "bubble";
  b.textContent = text;
  row.appendChild(b);
  log.appendChild(row);
  row.scrollIntoView({behavior: "smooth", block: "end"});
  return b;
}

async function loadHistory() {
  const r = await fetch("/api/history");
  const sessions = await r.json();
  const box = document.getElementById("history");
  box.innerHTML = "";
  if (!sessions.length) {
    box.innerHTML = "<span class='sub'>no sessions yet</span>";
    return;
  }
  for (const s of sessions.slice(0, 25)) {
    const div = document.createElement("div");
    div.className = "logitem";
    let badges = "";
    if (s.has_error) badges += "<span class='badge err'>error</span>";
    if (s.has_annotation) badges += "<span class='badge flag'>flagged</span>";
    div.innerHTML = `<a href="/session/${s.session_id}" target="_blank">${s.session_id}</a>${badges}
      <div class="sub">${s.pilot}</div>
      <div class="sub">${(s.started_at || "").slice(0, 19).replace("T", " ")} · ${s.n_events} events</div>`;
    box.appendChild(div);
  }
}

async function askQuestion(question) {
  send.disabled = true;
  const wait = add("agent", "… thinking (the flight recorder is running) …");
  try {
    const r = await fetch("/api/ask", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({question,
                            model_id: modelSel.value,
                            custom_model: customModel.value})
    });
    const data = await r.json();
    if (data.need_key) {
      wait.textContent = "This model needs an API key (" + data.need_key + ").";
      pendingQuestion = question;
      pendingEnvVar = data.need_key;
      keyText.textContent = "The provider '" + data.provider +
        "' has no key. Paste your " + data.need_key + " to continue.";
      keyInput.value = "";
      keyModal.style.display = "flex";
      keyInput.focus();
      return;
    }
    wait.textContent = data.answer;
    if (data.session_id && data.session_id !== "?") {
      const m = document.createElement("div");
      m.className = "meta";
      m.innerHTML = "🟠 recorded as <a href='/session/" + data.session_id +
                    "' target='_blank'>" + data.session_id +
                    "</a> — open the investigation report";
      wait.appendChild(m);
    }
  } catch (err) {
    wait.textContent = "Request failed: " + err;
  } finally {
    send.disabled = false; q.focus();
    loadHistory();
  }
}

form.addEventListener("submit", (e) => {
  e.preventDefault();
  const question = q.value.trim();
  if (!question) return;
  q.value = "";
  add("user", question);
  askQuestion(question);
});

document.getElementById("keyCancel").onclick = () => {
  keyModal.style.display = "none";
  pendingQuestion = null;
};

document.getElementById("keySave").onclick = async () => {
  const key = keyInput.value.trim();
  if (!key) return;
  const r = await fetch("/api/set_key", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({env_var: pendingEnvVar, key})
  });
  const data = await r.json();
  keyModal.style.display = "none";
  await loadModels();
  if (data.ok && pendingQuestion) {
    const question = pendingQuestion;
    pendingQuestion = null;
    askQuestion(question);       // automatic retry with the new key
  }
};

loadModels();
loadHistory();
