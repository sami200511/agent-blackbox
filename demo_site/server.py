"""
DemoShop demo website: chat with a recorded multi-model agent.

    python demo_site/server.py            # -> http://127.0.0.1:8700/

Routes:
    /                 chat UI — served from static/index.html
    /static/<file>    chat page assets (index.html, style.css, app.js)
    GET  /api/models  model registry + which providers have keys + rerank state
    POST /api/ask     {"question", "model_id", "custom_model"} -> answer/need_key
    POST /api/set_key {"env_var", "key"} -> stores key (process env + .env)
    GET  /api/history recorded sessions as JSON (newest first)
    /sessions         black-box index page
    /session/<id>     full investigation report for one session
"""

import http.server
import json
import os
import sys

import agent
from agent_blackbox.viewer import build_index, build_report

HOST, PORT = "127.0.0.1", 8700

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

# The chat page's HTML, CSS and JS live as plain files under static/ so each
# can be read and edited on its own; the server just serves them as-is.
_STATIC_TYPES = {
    ".html": "text/html", ".css": "text/css",
    ".js": "application/javascript", ".json": "application/json",
    ".svg": "image/svg+xml",
}


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

    def _static(self, name: str):
        """Serve a file from static/ (the chat page's html/css/js)."""
        safe = os.path.normpath(name).lstrip("/")
        path = os.path.join(STATIC_DIR, safe)
        if safe.startswith("..") or not os.path.isfile(path):
            return self._send("<h1>404</h1>", 404)
        ctype = _STATIC_TYPES.get(os.path.splitext(path)[1], "application/octet-stream")
        with open(path, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", f"{ctype}; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        try:
            if self.path in ("/", "/index.html"):
                self._static("index.html")
            elif self.path.startswith("/static/"):
                self._static(self.path[len("/static/"):].split("?")[0])
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
