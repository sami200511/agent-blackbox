"""
The `blackbox` command-line tool.

    blackbox list                         # sessions in ./recordings (or --db)
    blackbox verify SESSION_OR_PATH       # recompute the hash chain
    blackbox report SESSION [-o out.html] # write the investigation report
    blackbox serve [-p 8686]              # browse all sessions + reports
    blackbox demo                         # record the built-in demo incident

Storage selection (all subcommands):
    --dir recordings      JSONL directory (default; env BLACKBOX_DIR)
    --db blackbox.db      SQLite database instead
"""

from __future__ import annotations

import argparse
import os
import sys

from .core import verify_recording
from .storage import open_store
from .viewer import build_index, build_report, write_report


def _store(args):
    return open_store(directory=args.dir, db=args.db)


def cmd_list(args) -> int:
    store = _store(args)
    sessions = store.list_sessions()
    if not sessions:
        print("No recordings found.")
        return 0
    w = max(len(s.session_id) for s in sessions)
    print(f"{'SESSION':<{w}}  {'PILOT':<24} {'STARTED (UTC)':<20} {'EVENTS':>6}  FLAGS")
    for s in sessions:
        flags = []
        if s.has_error:
            flags.append("error")
        if s.has_annotation:
            flags.append("flagged")
        if not s.closed:
            flags.append("open")
        print(f"{s.session_id:<{w}}  {s.pilot:<24} {s.started_at[:19]:<20} "
              f"{s.n_events:>6}  {','.join(flags) or '-'}")
    return 0


def cmd_verify(args) -> int:
    target = args.session
    if os.path.exists(target) and target.endswith(".jsonl"):
        ok, msg = verify_recording(target)
    else:
        ok, msg = _store(args).verify(target)
    lamp = "✓" if ok else "✗"
    print(f"{lamp} {msg}")
    return 0 if ok else 1


def cmd_report(args) -> int:
    store = _store(args)
    if os.path.exists(args.session) and args.session.endswith(".jsonl"):
        from .core import load_recording, verify_events
        events = load_recording(args.session)
        out = args.output or args.session.replace(".jsonl", "_report.html")
        with open(out, "w", encoding="utf-8") as f:
            f.write(build_report(events, verify_events(events)))
    else:
        out = write_report(store, args.session, args.output)
    print(f"Incident report written: {out}")
    return 0


def cmd_demo(args) -> int:
    from .demo import run_demo
    bb_kwargs = {}
    if args.db:
        bb_kwargs["db"] = args.db
    else:
        bb_kwargs["directory"] = args.dir or "recordings"
    from .context import BlackBox
    bb = BlackBox(**bb_kwargs)
    sid = run_demo(bb)
    out = bb.report(sid)
    print(f"Demo incident recorded: session {sid}")
    print(f"Investigation report:   {out}")
    print(f"Verify integrity:       blackbox verify {sid}"
          + (f" --db {args.db}" if args.db else (f" --dir {args.dir}" if args.dir else "")))
    return 0


def cmd_serve(args) -> int:
    import http.server

    store = _store(args)

    class Handler(http.server.BaseHTTPRequestHandler):
        def _send(self, body: str, status: int = 200):
            data = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            try:
                if self.path in ("/", "/index.html"):
                    self._send(build_index(store.list_sessions(),
                                           link_fn=lambda sid: f"/session/{sid}"))
                elif self.path.startswith("/session/"):
                    sid = self.path[len("/session/"):].split("?")[0]
                    events = store.load(sid)
                    self._send(build_report(events, store.verify(sid)))
                else:
                    self._send("<h1>404</h1>", 404)
            except FileNotFoundError:
                self._send("<h1>Session not found</h1>", 404)
            except Exception as exc:  # keep the server alive
                self._send(f"<h1>500</h1><pre>{exc}</pre>", 500)

        def log_message(self, fmt, *a):
            sys.stderr.write("serve: " + fmt % a + "\n")

    addr = (args.host, args.port)
    print(f"Black box viewer on http://{args.host}:{args.port}/  (Ctrl-C to stop)")
    with http.server.ThreadingHTTPServer(addr, Handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="blackbox",
        description="Flight recorder for AI agents — record, verify, investigate.",
    )
    parser.add_argument("--dir", default=None,
                        help="recordings directory (default: ./recordings or $BLACKBOX_DIR)")
    parser.add_argument("--db", default=None, help="SQLite database path instead of --dir")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="list recorded sessions").set_defaults(fn=cmd_list)

    p = sub.add_parser("verify", help="verify a session's hash chain")
    p.add_argument("session", help="session id, or path to a .jsonl recording")
    p.set_defaults(fn=cmd_verify)

    p = sub.add_parser("report", help="write the HTML investigation report")
    p.add_argument("session", help="session id, or path to a .jsonl recording")
    p.add_argument("-o", "--output", default=None)
    p.set_defaults(fn=cmd_report)

    p = sub.add_parser("serve", help="serve an HTML index of all sessions")
    p.add_argument("-p", "--port", type=int, default=8686)
    p.add_argument("--host", default="127.0.0.1")
    p.set_defaults(fn=cmd_serve)

    sub.add_parser("demo", help="record the built-in demo incident").set_defaults(fn=cmd_demo)

    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
