"""
DemoShop demo website: chat with the recorded Gemini agent in your browser.

    python demo_site/server.py            # -> http://127.0.0.1:8700/

Routes:
    /                 chat UI — ask the agent to answer or do things
    POST /api/ask     {"question": "..."} -> {"answer", "session_id"}
    /sessions         black-box index of every recorded session
    /session/<id>     full investigation report for one session
"""

from __future__ import annotations

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
          --dim:#9BA0AA; --orange:#FF4F00; --green:#5FD08A;
          --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--panel); color:var(--ink);
         font:15px/1.6 -apple-system,Segoe UI,Roboto,sans-serif;
         display:flex; flex-direction:column; min-height:100vh; }
  .plate { background:var(--orange); color:#16181D; padding:18px 20px 14px;
           border-bottom:6px solid #16181D; }
  .plate-inner { max-width:820px; margin:0 auto; display:flex;
                 justify-content:space-between; align-items:baseline; gap:12px; }
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

  main { flex:1; width:100%; max-width:820px; margin:0 auto; padding:20px; }
  .msg { margin:14px 0; display:flex; }
  .msg .bubble { max-width:85%; padding:12px 16px; border-radius:10px;
                 white-space:pre-wrap; }
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

  form { position:sticky; bottom:0; background:var(--panel);
         padding:14px 20px 20px; border-top:1px solid var(--line); }
  .bar { max-width:780px; margin:0 auto; display:flex; gap:10px; }
  input[type=text] { flex:1; background:var(--panel2); color:var(--ink);
        border:1px solid var(--line); border-radius:8px; padding:12px 14px;
        font:15px inherit; }
  input[type=text]:focus { outline:2px solid var(--orange); border-color:transparent; }
  button { background:var(--orange); color:#16181D; border:0; border-radius:8px;
           padding:12px 22px; font-weight:800; cursor:pointer;
           font-family:var(--mono); letter-spacing:.08em; }
  button:disabled { opacity:.4; cursor:wait; }
</style>
</head>
<body>
  <div class="plate"><div class="plate-inner">
    <h1>DemoShop Support</h1>
    <span class="rec"><span class="dot"></span> BLACK BOX RECORDING</span>
    <a href="/sessions">Investigate sessions →</a>
  </div></div>
  <div class="stripes"></div>

  <main id="log">
    <div class="hint">Ask the agent anything about DemoShop, or tell it to
<b>do</b> something. Every question becomes a tamper-evident recording. Try:
<br>• <code>I bought a laptop 45 days ago and it broke. Can I get a refund?</code>
<br>• <code>Where is my order DS-1002?</code>
<br>• <code>Open a support ticket for me, I'm Sam and my charger melted.</code></div>
  </main>

  <form id="form">
    <div class="bar">
      <input id="q" type="text" autocomplete="off"
             placeholder="Ask the agent, or tell it to do something…" autofocus>
      <button id="send" type="submit">ASK</button>
    </div>
  </form>

<script>
const log = document.getElementById("log");
const form = document.getElementById("form");
const q = document.getElementById("q");
const send = document.getElementById("send");

function add(cls, text, meta) {
  const row = document.createElement("div");
  row.className = "msg " + cls;
  const b = document.createElement("div");
  b.className = "bubble";
  b.textContent = text;
  if (meta) { const m = document.createElement("div"); m.className = "meta";
              m.innerHTML = meta; b.appendChild(m); }
  row.appendChild(b);
  log.appendChild(row);
  row.scrollIntoView({behavior: "smooth", block: "end"});
  return b;
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = q.value.trim();
  if (!question) return;
  q.value = ""; send.disabled = true;
  add("user", question);
  const wait = add("agent", "… thinking (the flight recorder is running) …");
  try {
    const r = await fetch("/api/ask", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({question})
    });
    const data = await r.json();
    wait.textContent = data.answer;
    const m = document.createElement("div");
    m.className = "meta";
    m.innerHTML = "🟠 recorded as <a href='/session/" + data.session_id +
                  "' target='_blank'>" + data.session_id +
                  "</a> — open the investigation report";
    wait.appendChild(m);
  } catch (err) {
    wait.textContent = "Request failed: " + err;
  } finally {
    send.disabled = false; q.focus();
  }
});
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

    def do_GET(self):
        try:
            if self.path in ("/", "/index.html"):
                self._send(CHAT_HTML)
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

    def do_POST(self):
        if self.path != "/api/ask":
            return self._send('{"error": "not found"}', 404, "application/json")
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
            question = str(payload.get("question", "")).strip()
            if not question:
                return self._send('{"error": "empty question"}', 400,
                                  "application/json")
            answer, session_id = agent.ask(question,
                                           user_id=self.client_address[0])
            self._send(json.dumps({"answer": answer, "session_id": session_id}),
                       200, "application/json")
        except Exception as exc:
            self._send(json.dumps({"answer": f"server error: {exc}",
                                   "session_id": "?"}), 500, "application/json")

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
