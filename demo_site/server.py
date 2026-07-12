"""
DemoShop demo website: chat with a recorded multi-model agent.

    python demo_site/server.py            # -> http://127.0.0.1:8700/

Routes:
    /                 chat UI — model switcher, key prompt, session history
    GET  /api/models  model registry + which providers have keys + rerank state
    POST /api/ask     {"question", "model_id", "custom_model"} -> answer/need_key
    POST /api/set_key {"env_var", "key"} -> stores key (process env + .env)
    GET  /api/history recorded sessions as JSON (newest first)
    /sessions         black-box index page
    /session/<id>     full investigation report for one session
"""

import http.server
import json
import sys

import agent
from agent_blackbox.viewer import build_index, build_report

HOST, PORT = "127.0.0.1", 8700

CHAT_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DemoShop · Recorded Agent</title>
<style>
  :root { --panel:#14161B; --panel2:#1C1F26; --line:#2C313B; --ink:#E8E6DF;
          --dim:#9BA0AA; --orange:#FF4F00; --green:#5FD08A; --red:#FF5A5A;
          --amber:#FFB000;
          --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--panel); color:var(--ink);
         font:15px/1.6 -apple-system,Segoe UI,Roboto,sans-serif;
         display:flex; flex-direction:column; min-height:100vh; }
  .plate { background:var(--orange); color:#16181D; padding:16px 20px 12px;
           border-bottom:6px solid #16181D; }
  .plate-inner { max-width:1140px; margin:0 auto; display:flex;
                 justify-content:space-between; align-items:baseline;
                 gap:12px; flex-wrap:wrap; }
  .plate h1 { margin:0; font-family:var(--mono); font-size:18px;
              letter-spacing:.16em; text-transform:uppercase; font-weight:800; }
  .plate a { color:#16181D; font-family:var(--mono); font-size:12px;
             letter-spacing:.12em; text-transform:uppercase; }
  .stripes { height:8px; background:repeating-linear-gradient(45deg,
             #16181D 0 12px, var(--orange) 12px 24px); }
  .rec { display:inline-flex; align-items:center; gap:6px;
         font-family:var(--mono); font-size:11px; letter-spacing:.14em; }
  .rec .dot { width:9px; height:9px; border-radius:50%; background:#B3220A;
              animation:blink 1.4s infinite; }
  @keyframes blink { 50% { opacity:.25; } }

  .toolbar { max-width:1140px; margin:14px auto 0; padding:0 20px; width:100%;
             display:flex; gap:10px; align-items:center; flex-wrap:wrap;
             font-family:var(--mono); font-size:12px; color:var(--dim); }
  select, .toolbar input[type=text] { background:var(--panel2); color:var(--ink);
          border:1px solid var(--line); border-radius:6px; padding:8px 10px;
          font:13px var(--mono); }
  #customModel { display:none; width:260px; }
  .pill { border:1px solid var(--line); border-radius:12px; padding:3px 10px;
          letter-spacing:.08em; }
  .pill.on { color:var(--green); border-color:var(--green); }
  .pill.off { color:var(--dim); }

  .layout { flex:1; width:100%; max-width:1140px; margin:0 auto;
            display:flex; gap:18px; padding:14px 20px 0; align-items:stretch; }
  main { flex:1; min-width:0; }
  @media (max-width:900px) { .layout { flex-direction:column; } aside { width:100%!important; } }

  .msg { margin:14px 0; display:flex; }
  .msg .bubble { max-width:88%; padding:12px 16px; border-radius:10px;
                 white-space:pre-wrap; overflow-wrap:break-word; }
  .msg.user { justify-content:flex-end; }
  .msg.user .bubble { background:#2A3550; border:1px solid #3A4A6E; }
  .msg.agent .bubble { background:var(--panel2); border:1px solid var(--line);
                       border-left:4px solid var(--orange); }
  .msg.agent .meta { margin-top:8px; font-family:var(--mono); font-size:11px;
                     color:var(--dim); letter-spacing:.06em; }
  .msg.agent .meta a { color:var(--orange); }
  .hint { color:var(--dim); font-size:13.5px; background:var(--panel2);
          border:1px dashed var(--line); border-radius:8px; padding:14px 16px; }
  .hint code { font-family:var(--mono); color:var(--ink); }

  aside { width:290px; flex-shrink:0; }
  .logbox { background:var(--panel2); border:1px solid var(--line);
            border-radius:8px; padding:12px 14px; position:sticky; top:12px; }
  .logbox h3 { margin:0 0 10px; font-family:var(--mono); font-size:11px;
               letter-spacing:.2em; color:var(--dim); text-transform:uppercase; }
  .logitem { border-top:1px solid var(--line); padding:8px 0; font-size:12.5px; }
  .logitem:first-of-type { border-top:0; }
  .logitem a { color:var(--orange); font-family:var(--mono); font-size:12px;
               text-decoration:none; }
  .logitem .sub { color:var(--dim); font-family:var(--mono); font-size:10.5px; }
  .badge { font-family:var(--mono); font-size:10px; padding:1px 7px;
           border-radius:9px; border:1px solid var(--line); color:var(--dim);
           margin-left:5px; }
  .badge.err { color:var(--red); border-color:var(--red); }
  .badge.flag { color:var(--amber); border-color:var(--amber); }

  form.askbar { position:sticky; bottom:0; background:var(--panel);
         padding:14px 20px 20px; border-top:1px solid var(--line); }
  .bar { max-width:1100px; margin:0 auto; display:flex; gap:10px; }
  .bar input[type=text] { flex:1; background:var(--panel2); color:var(--ink);
        border:1px solid var(--line); border-radius:8px; padding:12px 14px;
        font:15px inherit; }
  .bar input[type=text]:focus { outline:2px solid var(--orange); border-color:transparent; }
  button { background:var(--orange); color:#16181D; border:0; border-radius:8px;
           padding:12px 22px; font-weight:800; cursor:pointer;
           font-family:var(--mono); letter-spacing:.08em; }
  button:disabled { opacity:.4; cursor:wait; }

  /* key modal */
  #keyModal { display:none; position:fixed; inset:0; background:rgba(0,0,0,.65);
              z-index:10; align-items:center; justify-content:center; }
  #keyModal .box { background:var(--panel2); border:1px solid var(--orange);
              border-radius:10px; padding:24px; width:min(480px, 92vw); }
  #keyModal h2 { margin:0 0 6px; font-family:var(--mono); font-size:14px;
              letter-spacing:.14em; text-transform:uppercase; color:var(--orange); }
  #keyModal p { color:var(--dim); font-size:13.5px; margin:6px 0 14px; }
  #keyModal input { width:100%; background:var(--panel); color:var(--ink);
              border:1px solid var(--line); border-radius:6px; padding:11px 12px;
              font:13px var(--mono); margin-bottom:14px; }
  #keyModal .row { display:flex; gap:10px; justify-content:flex-end; }
  #keyModal .ghost { background:transparent; color:var(--dim);
              border:1px solid var(--line); }
</style>
</head>
<body>
  <div class="plate"><div class="plate-inner">
    <h1>DemoShop Support</h1>
    <span class="rec"><span class="dot"></span> BLACK BOX RECORDING</span>
    <a href="/sessions">Full session index →</a>
  </div></div>
  <div class="stripes"></div>

  <div class="toolbar">
    <label for="model">MODEL</label>
    <select id="model"></select>
    <input id="customModel" type="text"
           placeholder="e.g. mistralai/mistral-small-3.1">
    <span id="rerankPill" class="pill off">RERANK: OFF</span>
    <span class="pill" id="keysPill"></span>
  </div>

  <div class="layout">
    <main id="log">
      <div class="hint">Pick a model above, then ask the agent anything about
DemoShop — or tell it to <b>do</b> something. Every question becomes a
tamper-evident recording. Try:
<br>• <code>I bought a laptop 45 days ago and it broke. Can I get a refund?</code>
<br>• <code>Where is my order DS-1002?</code>
<br>• <code>Open a support ticket for me, I'm Sam and my charger melted.</code>
<br><br>Add an <b>OpenRouter</b> key (pick an OpenRouter model once and the
site will ask you) to unlock more models <i>and</i> the retrieval reranker.</div>
    </main>

    <aside>
      <div class="logbox">
        <h3>Flight log — recorded sessions</h3>
        <div id="history"><span class="sub" style="color:var(--dim)">loading…</span></div>
      </div>
    </aside>
  </div>

  <form class="askbar" id="form">
    <div class="bar">
      <input id="q" type="text" autocomplete="off"
             placeholder="Ask the agent, or tell it to do something…" autofocus>
      <button id="send" type="submit">ASK</button>
    </div>
  </form>

  <div id="keyModal">
    <div class="box">
      <h2>API key needed</h2>
      <p id="keyText"></p>
      <input id="keyInput" type="password" placeholder="paste the key here">
      <div class="row">
        <button class="ghost" type="button" id="keyCancel">Cancel</button>
        <button type="button" id="keySave">Save & retry</button>
      </div>
      <p style="margin:12px 0 0">The key is stored in the project's <code>.env</code>
      file (gitignored) and used from the environment — never shown again.</p>
    </div>
  </div>

<script>
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
</script>
</body>
</html>"""


class Handler(http.server.BaseHTTPRequestHandler):
    def _send(self, body: str, status: int = 200, ctype: str = "text/html"):
        data = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", f"{ctype}; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _json(self, obj, status: int = 200):
        self._send(json.dumps(obj), status, "application/json")

    def do_GET(self):
        try:
            if self.path in ("/", "/index.html"):
                self._send(CHAT_HTML)
            elif self.path == "/api/models":
                self._json(agent.models_status())
            elif self.path == "/api/history":
                self._json([
                    {"session_id": s.session_id, "pilot": s.pilot,
                     "started_at": s.started_at, "n_events": s.n_events,
                     "has_error": s.has_error, "has_annotation": s.has_annotation}
                    for s in agent.bb.list_sessions()
                ])
            elif self.path == "/sessions":
                self._send(build_index(agent.bb.list_sessions(),
                                       link_fn=lambda sid: f"/session/{sid}"))
            elif self.path.startswith("/session/"):
                sid = self.path[len("/session/"):].split("?")[0]
                events = agent.bb.load(sid)
                self._send(build_report(events, agent.bb.verify(sid)))
            else:
                self._send("<h1>404</h1>", 404)
        except FileNotFoundError:
            self._send("<h1>Session not found</h1>", 404)
        except Exception as exc:
            self._send(f"<h1>500</h1><pre>{exc}</pre>", 500)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length) or b"{}")

    def do_POST(self):
        try:
            if self.path == "/api/ask":
                payload = self._body()
                question = str(payload.get("question", "")).strip()
                if not question:
                    return self._json({"error": "empty question"}, 400)
                try:
                    answer, session_id = agent.ask(
                        question,
                        model_id=str(payload.get("model_id", "gemini-3.5-flash")),
                        custom_model=str(payload.get("custom_model", "")),
                        user_id=self.client_address[0],
                    )
                except agent.MissingKeyError as exc:
                    return self._json({"need_key": exc.env_var,
                                       "provider": exc.provider})
                return self._json({"answer": answer, "session_id": session_id})

            if self.path == "/api/set_key":
                payload = self._body()
                try:
                    provider = agent.set_api_key(
                        str(payload.get("env_var", "")),
                        str(payload.get("key", "")))
                except ValueError as exc:
                    return self._json({"ok": False, "error": str(exc)}, 400)
                return self._json({"ok": True, "provider": provider})

            self._json({"error": "not found"}, 404)
        except Exception as exc:
            self._json({"answer": f"server error: {exc}", "session_id": "?"}, 500)

    def log_message(self, fmt, *args):
        sys.stderr.write("site: " + fmt % args + "\n")


def main() -> None:
    print(f"DemoShop recorded-agent site: http://{HOST}:{PORT}/")
    print(f"Session investigations:       http://{HOST}:{PORT}/sessions")
    with http.server.ThreadingHTTPServer((HOST, PORT), Handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
