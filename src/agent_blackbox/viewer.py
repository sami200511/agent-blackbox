"""
Incident-investigation report generation.

Turns a black box recording into a single self-contained HTML file:

  * Integrity check   — was the recording tampered with? (hash chain)
  * Incident flags    — annotations and errors
  * In vs out         — what the user asked vs what the agent answered
  * Model calls (CVR) — each LLM call with its stated reasoning
  * Decision analysis — the agent's STATED reason next to the EVIDENCE it
                        actually had, including evidence it ignored
  * Full timeline     — every event, like a flight data recorder printout

Also renders an index page over all sessions, used by `blackbox serve`.
"""

from __future__ import annotations

import html
import json
from typing import Optional

from .core import verify_events

ORANGE = "#FF4F00"  # international orange — the colour of real flight recorders

_CSS = """
  :root {
    --panel:#14161B; --panel2:#1C1F26; --line:#2C313B;
    --ink:#E8E6DF; --dim:#9BA0AA; --orange:%(orange)s;
    --amber:#FFB000; --green:#5FD08A; --red:#FF5A5A;
    --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--panel); color:var(--ink);
         font:15px/1.6 -apple-system,Segoe UI,Roboto,sans-serif; }
  .mono { font-family:var(--mono); }
  .wrap { max-width:980px; margin:0 auto; padding:0 20px 80px; }
  a { color:var(--orange); }

  .plate { background:var(--orange); color:#16181D; padding:26px 20px 22px;
            border-bottom:6px solid #16181D; }
  .plate-inner { max-width:980px; margin:0 auto; display:flex;
                  justify-content:space-between; flex-wrap:wrap; gap:12px; }
  .plate h1 { margin:0; font-family:var(--mono); font-size:22px;
               letter-spacing:.18em; text-transform:uppercase; font-weight:800; }
  .plate .sub { font-family:var(--mono); font-size:12px; letter-spacing:.14em;
                 text-transform:uppercase; opacity:.85; }
  .stripes { height:10px;
    background:repeating-linear-gradient(45deg,#16181D 0 14px,var(--orange) 14px 28px); }

  .verify { display:flex; align-items:center; gap:10px; margin:26px 0 6px;
             font-family:var(--mono); font-size:13px; letter-spacing:.1em; }
  .verify .lamp { width:11px; height:11px; border-radius:50%%; }
  .verify.ok .lamp { background:var(--green); box-shadow:0 0 8px var(--green); }
  .verify.bad .lamp { background:var(--red); box-shadow:0 0 8px var(--red); }
  .verify.ok { color:var(--green); } .verify.bad { color:var(--red); }
  .verify-detail { color:var(--dim); letter-spacing:0; }

  h2 { font-family:var(--mono); font-size:13px; letter-spacing:.22em;
        text-transform:uppercase; color:var(--dim); margin:40px 0 14px;
        border-bottom:1px solid var(--line); padding-bottom:8px; }

  .io-grid { display:grid; grid-template-columns:1fr 1fr; gap:14px; }
  @media (max-width:720px) { .io-grid,.dec-grid { grid-template-columns:1fr!important; } }
  .card { background:var(--panel2); border:1px solid var(--line);
           border-radius:6px; padding:16px 18px; margin-bottom:12px; }
  .card .label { font-family:var(--mono); font-size:11px; letter-spacing:.18em;
                  color:var(--dim); text-transform:uppercase; margin-bottom:8px; }
  .card.output { border-left:4px solid var(--orange); }
  .card.input  { border-left:4px solid var(--dim); }
  .card p { margin:0; white-space:pre-wrap; }

  .flag { background:var(--panel2); border:1px solid var(--amber);
           border-left:6px solid var(--amber); border-radius:6px;
           padding:12px 16px; margin:10px 0; }
  .flag.err { border-color:var(--red); border-left-color:var(--red); }
  .flag-author { font-family:var(--mono); font-size:11px; color:var(--amber); letter-spacing:.14em; }
  .flag.err .flag-author { color:var(--red); }
  .flag p { margin:6px 0 0; white-space:pre-wrap; }

  .llm { background:var(--panel2); border:1px solid var(--line);
          border-radius:6px; padding:16px 18px; margin:12px 0; }
  .llm-head { display:flex; gap:14px; flex-wrap:wrap; align-items:baseline;
               font-family:var(--mono); font-size:12px; color:var(--dim); }
  .llm-head .model { color:var(--orange); font-weight:700; font-size:14px; }
  .llm .ev-label { margin-top:12px; }
  .llm p { margin:4px 0 0; white-space:pre-wrap; }
  .llm .reason { font-style:italic; color:#D8D4C8; }
  .toolreq { font-family:var(--mono); font-size:12.5px; color:var(--amber); margin-top:6px; }

  .decision { background:var(--panel2); border:1px solid var(--line);
               border-radius:6px; padding:18px; margin:14px 0; }
  .dec-head { display:flex; justify-content:space-between; align-items:baseline;
               margin-bottom:12px; }
  .dec-choice { color:var(--orange); font-weight:700; font-size:15px; }
  .conf { font-family:var(--mono); font-size:12px; color:var(--dim); }
  .dec-grid { display:grid; grid-template-columns:1fr 1.3fr; gap:18px; }
  .ev-label { font-family:var(--mono); font-size:10px; letter-spacing:.16em;
               color:var(--dim); text-transform:uppercase; margin:12px 0 6px; }
  .ev-label:first-child { margin-top:0; }
  .reasoning { margin:0 0 6px; font-style:italic; white-space:pre-wrap; }
  .alts { margin:4px 0 0 18px; padding:0; color:var(--dim); font-size:14px; }
  .evidence { border:1px dashed var(--line); border-radius:6px;
               padding:12px 14px; margin-bottom:10px; }
  .chunk { border-top:1px solid var(--line); padding-top:10px; margin-top:10px; }
  .chunk:first-of-type { border-top:0; margin-top:4px; padding-top:0; }
  .chunk-head { display:flex; justify-content:space-between; font-size:12px; }
  .chunk-head .score { color:var(--amber); font-family:var(--mono); }
  .chunk-src { font-size:11px; color:var(--dim); margin:2px 0 4px; }
  .chunk p { margin:4px 0 0; font-size:14px; }
  .investigator-note { margin:14px 0 0; font-size:12.5px; color:var(--dim);
                        border-top:1px solid var(--line); padding-top:10px; }

  .ev-row { border:1px solid var(--line); border-radius:4px; margin:6px 0;
             background:var(--panel2); }
  .ev-row summary { display:flex; gap:16px; align-items:baseline; cursor:pointer;
                     padding:9px 14px; font-size:12.5px; list-style:none; }
  .ev-row summary::-webkit-details-marker { display:none; }
  .ev-row .seq { color:var(--dim); }
  .ev-row .etype { min-width:120px; }
  .ev-row .etime { color:var(--dim); margin-left:auto; }
  .ev-row .ehash { color:#4E5560; }
  .ev-row pre { margin:0; padding:12px 16px; border-top:1px solid var(--line);
                 overflow-x:auto; font-size:12px; color:#C9CDD6;
                 font-family:var(--mono); white-space:pre-wrap; }
  .t-in  summary .etype { color:var(--green); }
  .t-out summary .etype { color:var(--orange); }
  .t-dec summary .etype { color:var(--amber); }
  .t-llm summary .etype { color:#7FB4FF; }
  .t-ret summary .etype { color:#B99BFF; }
  .t-tool summary .etype { color:#5FD0C8; }
  .evidence.err { border-color:var(--red); }
  .t-err summary .etype,.t-flag summary .etype { color:var(--red); }
  pre { font-family:var(--mono); }

  table.idx { width:100%%; border-collapse:collapse; margin-top:18px; }
  table.idx th, table.idx td { text-align:left; padding:10px 12px;
    border-bottom:1px solid var(--line); font-size:14px; }
  table.idx th { font-family:var(--mono); font-size:11px; letter-spacing:.16em;
    text-transform:uppercase; color:var(--dim); }
  .badge { font-family:var(--mono); font-size:11px; padding:2px 8px;
    border-radius:10px; border:1px solid var(--line); color:var(--dim); }
  .badge.err { color:var(--red); border-color:var(--red); }
  .badge.flag { color:var(--amber); border-color:var(--amber); }
  .badge.open { color:#7FB4FF; border-color:#7FB4FF; }
""" % {"orange": ORANGE}


def esc(s) -> str:
    if s is None:
        return ""
    if not isinstance(s, str):
        s = json.dumps(s, ensure_ascii=False, indent=2)
    return html.escape(s)


def _rerank_label(ev: dict) -> str:
    p = ev["payload"]
    model = esc(p.get("model", "reranker"))
    if p.get("error"):
        return f"RERANK ({model}) — FAILED, fell back to keyword order"
    return f"RERANK ({model}) — re-scored relevance"


def _rerank_rows(ev: dict) -> str:
    p = ev["payload"]
    if p.get("error"):
        return (f"<p class='reasoning'>{esc(p.get('error'))}</p>"
                f"<p style='color:var(--dim);font-size:12px'>{esc(p.get('note'))}</p>")
    rows = ""
    for r in p.get("results", []):
        rows += (
            f"<div class='chunk'><div class='chunk-head'>"
            f"<span class='mono'>{esc(r.get('doc_id'))}</span>"
            f"<span class='score'>relevance {esc(r.get('relevance_score'))}</span></div>"
            f"<div class='chunk-src mono'>{esc(r.get('source'))}</div></div>"
        )
    return rows or "<p style='color:var(--dim)'>no rerank results</p>"


def _page(title: str, header_right: str, body: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)}</title>
<style>{_CSS}</style>
</head>
<body>
  <div class="plate">
    <div class="plate-inner">
      <div>
        <h1>Flight Recorder</h1>
        <div class="sub">Agent incident investigation unit</div>
      </div>
      <div style="text-align:right">{header_right}</div>
    </div>
  </div>
  <div class="stripes"></div>
  <div class="wrap">
{body}
  </div>
</body>
</html>"""


def build_report(events: list[dict], verify_result: Optional[tuple] = None) -> str:
    ok, verify_msg = verify_result if verify_result else verify_events(events)
    by_seq = {e["seq"]: e for e in events}

    session = next((e for e in events if e["type"] == "session_start"), {})
    user_inputs = [e for e in events if e["type"] == "user_input"]
    finals = [e for e in events if e["type"] == "final_output"]
    decisions = [e for e in events if e["type"] == "decision"]
    llm_calls = [e for e in events if e["type"] == "llm_call"]
    retrievals = [e for e in events if e["type"] == "retrieval"]
    reranks = [e for e in events if e["type"] == "rerank"]
    tool_calls = [e for e in events if e["type"] == "tool_call"]
    annotations = [e for e in events if e["type"] == "annotation"]
    errors = [e for e in events if e["type"] == "error"]

    meta = session.get("payload", {})
    pilot = esc(meta.get("pilot", "unknown-agent"))
    sid = esc(session.get("session_id", events[0]["session_id"] if events else "?"))

    # ---------- model calls (the CVR) ----------
    llm_html = ""
    for e in llm_calls:
        p = e["payload"]
        usage = p.get("usage") or {}
        usage_bits = " · ".join(
            f"{k.replace('_tokens','')} {v}" for k, v in usage.items() if v is not None
        )
        head_bits = [
            f"<span class='model'>{esc(p.get('model'))}</span>",
            f"<span>SEQ {e['seq']}</span>",
        ]
        if p.get("provider"):
            head_bits.append(f"<span>{esc(p['provider'])}</span>")
        if p.get("stop_reason"):
            head_bits.append(f"<span>stop: {esc(p['stop_reason'])}</span>")
        if p.get("latency_ms") is not None:
            head_bits.append(f"<span>{p['latency_ms']} ms</span>")
        if p.get("streamed"):
            head_bits.append("<span>streamed</span>")
        if usage_bits:
            head_bits.append(f"<span>{esc(usage_bits)}</span>")
        reasoning_html = ""
        if p.get("reasoning"):
            reasoning_html = (
                "<div class='ev-label'>Model reasoning (CVR — what it "
                "<em>said</em>, not proof of why)</div>"
                f"<p class='reason'>{esc(p['reasoning'])}</p>"
            )
        tools_html = "".join(
            f"<div class='toolreq'>→ tool requested: {esc(tc.get('name'))}"
            f"({esc(tc.get('arguments'))})</div>"
            for tc in (p.get("tool_calls") or [])
        )
        llm_html += f"""
        <div class="llm">
          <div class="llm-head">{' '.join(head_bits)}</div>
          {reasoning_html}
          <div class="ev-label">Response</div>
          <p>{esc(p.get('response') or '(no text)')}</p>
          {tools_html}
        </div>"""

    # ---------- decision analysis blocks ----------
    decision_html = ""
    for d in decisions:
        p = d["payload"]
        alts = "".join(f"<li>{esc(a)}</li>" for a in p.get("alternatives", []))
        ev_blocks = ""
        for ref in p.get("evidence_refs", []):
            ev = by_seq.get(ref)
            if not ev:
                continue
            if ev["type"] == "retrieval":
                rows = ""
                for c in ev["payload"].get("chunks", []):
                    rows += (
                        f"<div class='chunk'><div class='chunk-head'>"
                        f"<span class='mono'>{esc(c.get('doc_id'))}</span>"
                        f"<span class='score'>score {esc(c.get('score'))}</span></div>"
                        f"<div class='chunk-src mono'>{esc(c.get('source'))}</div>"
                        f"<p>{esc(c.get('text'))}</p></div>"
                    )
                ev_blocks += (
                    f"<div class='evidence'><div class='ev-label'>SEQ {ref} · RETRIEVAL "
                    f"— everything the agent could see</div>{rows}</div>"
                )
            elif ev["type"] == "rerank":
                ev_blocks += (
                    f"<div class='evidence'><div class='ev-label'>SEQ {ref} · "
                    f"{_rerank_label(ev)}</div>{_rerank_rows(ev)}</div>"
                )
            elif ev["type"] == "llm_call":
                ev_blocks += (
                    f"<div class='evidence'><div class='ev-label'>SEQ {ref} · MODEL "
                    f"REASONING (CVR — what it <em>said</em>, not proof of why)</div>"
                    f"<p class='reasoning'>{esc(ev['payload'].get('reasoning') or ev['payload'].get('response'))}</p></div>"
                )
            else:
                ev_blocks += (
                    f"<div class='evidence'><div class='ev-label'>SEQ {ref} · "
                    f"{esc(ev['type']).upper()}</div>"
                    f"<pre>{esc(ev['payload'])}</pre></div>"
                )
        conf = p.get("confidence")
        conf_html = f"<span class='conf'>confidence {conf}</span>" if conf is not None else ""
        decision_html += f"""
        <div class="decision">
          <div class="dec-head">
            <span class="dec-choice mono">{esc(p.get('choice'))}</span>{conf_html}
          </div>
          <div class="dec-grid">
            <div>
              <div class="ev-label">STATED REASON (agent's own why)</div>
              <p class="reasoning">"{esc(p.get('stated_reason'))}"</p>
              <div class="ev-label">PATHS NOT TAKEN</div>
              <ul class="alts">{alts}</ul>
            </div>
            <div>{ev_blocks}</div>
          </div>
          <p class="investigator-note">Investigator: compare the stated reason with the
          full evidence on the right. Anything present in the evidence but absent from
          the stated reason is a lead.</p>
        </div>"""

    # ---------- annotations / incident flags ----------
    ann_html = "".join(
        f"<div class='flag'><span class='flag-author mono'>{esc(a['payload'].get('author'))}</span>"
        f"<p>{esc(a['payload'].get('text'))}</p></div>"
        for a in annotations
    )
    err_html = "".join(
        f"<div class='flag err'><span class='flag-author mono'>ERROR"
        f"{(' · ' + esc(e['payload'].get('exception_type'))) if e['payload'].get('exception_type') else ''}</span>"
        f"<p>{esc(e['payload'].get('message'))}</p></div>"
        for e in errors
    )

    # ---------- in vs out ----------
    io_html = ""
    for i in range(max(len(user_inputs), len(finals))):
        u = user_inputs[i] if i < len(user_inputs) else None
        f = finals[i] if i < len(finals) else None
        io_html += f"""
    <div class="io-grid">
      <div class="card input">
        <div class="label">User said{f" · SEQ {u['seq']}" if u else ''}</div>
        <p>{esc(u['payload']['text']) if u else '—'}</p>
      </div>
      <div class="card output">
        <div class="label">Agent answered{f" · SEQ {f['seq']}" if f else ''}</div>
        <p>{esc(f['payload']['text']) if f else '—'}</p>
      </div>
    </div>"""

    # ---------- retrieval, reranking & tool actions ----------
    retrieval_html = ""
    for e in retrievals:
        p = e["payload"]
        rows_r = ""
        for c in p.get("chunks", []):
            rows_r += (
                f"<div class='chunk'><div class='chunk-head'>"
                f"<span class='mono'>{esc(c.get('doc_id'))}</span>"
                f"<span class='score'>keyword {esc(c.get('score'))}</span></div>"
                f"<div class='chunk-src mono'>{esc(c.get('source'))}</div>"
                f"<p>{esc(c.get('text'))}</p></div>"
            )
        retrieval_html += (
            f"<div class='evidence'><div class='ev-label'>SEQ {e['seq']} · RETRIEVAL "
            f"— query \"{esc(p.get('query'))}\" — everything the agent could see</div>"
            f"{rows_r or '<p style=\"color:var(--dim)\">no documents matched</p>'}</div>"
        )
    for e in reranks:
        retrieval_html += (
            f"<div class='evidence'><div class='ev-label'>SEQ {e['seq']} · "
            f"{_rerank_label(e)}</div>{_rerank_rows(e)}</div>"
        )

    tool_html = ""
    for e in tool_calls:
        p = e["payload"]
        ok_cls = "" if p.get("ok", True) else " err"
        lat = f" · {p['latency_ms']} ms" if p.get("latency_ms") is not None else ""
        tool_html += (
            f"<div class='evidence{ok_cls}'><div class='ev-label'>SEQ {e['seq']} · "
            f"TOOL {esc(p.get('name'))} · {'ok' if p.get('ok', True) else 'FAILED'}{lat}</div>"
            f"<div class='ev-label'>arguments</div><pre>{esc(p.get('arguments'))}</pre>"
            f"<div class='ev-label'>result</div><pre>{esc(p.get('result'))}</pre></div>"
        )

    # ---------- timeline ----------
    type_class = {
        "user_input": "t-in", "final_output": "t-out", "decision": "t-dec",
        "llm_call": "t-llm", "retrieval": "t-ret", "rerank": "t-ret",
        "tool_call": "t-tool", "error": "t-err", "annotation": "t-flag",
    }
    rows = ""
    for e in events:
        cls = type_class.get(e["type"], "")
        payload_str = json.dumps(e["payload"], ensure_ascii=False, indent=2)
        rows += f"""
        <details class="ev-row {cls}">
          <summary>
            <span class="seq mono">{e['seq']:03d}</span>
            <span class="etype mono">{esc(e['type'])}</span>
            <span class="etime mono">+{e.get('t_rel_ms', 0)} ms</span>
            <span class="ehash mono">{e.get('hash', '')[:12]}…</span>
          </summary>
          <pre>{esc(payload_str)}</pre>
        </details>"""

    verify_class = "ok" if ok else "bad"
    verify_word = "CHAIN INTACT" if ok else "TAMPERING DETECTED"

    header_right = (
        f"<div class='sub'>Session {sid}</div>"
        f"<div class='sub'>Pilot: {pilot}</div>"
    )
    body = f"""
    <div class="verify {verify_class}">
      <span class="lamp"></span> {verify_word}
      <span class="verify-detail">— {esc(verify_msg)}</span>
    </div>

    <h2>01 · Incident flags</h2>
    {ann_html or ''}{err_html or ''}
    {"<p style='color:var(--dim)'>No flags recorded.</p>" if not (ann_html or err_html) else ''}

    <h2>02 · What went in vs what came out</h2>
    {io_html or "<p style='color:var(--dim)'>No user input / final output recorded.</p>"}

    <h2>03 · Retrieval &amp; reranking — what the agent could see</h2>
    {retrieval_html or "<p style='color:var(--dim)'>No retrieval recorded.</p>"}

    <h2>04 · Model calls — cockpit voice recorder ({len(llm_calls)})</h2>
    {llm_html or "<p style='color:var(--dim)'>No LLM calls recorded.</p>"}

    <h2>05 · Actions taken — tool calls ({len(tool_calls)})</h2>
    {tool_html or "<p style='color:var(--dim)'>No tool calls recorded.</p>"}

    <h2>06 · Decision analysis — stated reason vs evidence</h2>
    {decision_html or "<p style='color:var(--dim)'>No decision events recorded.</p>"}

    <h2>07 · Full timeline ({len(events)} events — click any row)</h2>
    {rows}
"""
    return _page(f"Incident Report · {session.get('session_id', '?')}", header_right, body)


def build_index(sessions, link_fn=None) -> str:
    """Index page over all sessions. link_fn(session_id) -> href."""
    link_fn = link_fn or (lambda sid: f"{sid}.html")
    rows = ""
    for s in sessions:
        badges = []
        if s.has_error:
            badges.append("<span class='badge err'>error</span>")
        if s.has_annotation:
            badges.append("<span class='badge flag'>flagged</span>")
        if not s.closed:
            badges.append("<span class='badge open'>open</span>")
        rows += (
            f"<tr><td><a href='{esc(link_fn(s.session_id))}' class='mono'>"
            f"{esc(s.session_id)}</a></td>"
            f"<td class='mono'>{esc(s.pilot)}</td>"
            f"<td class='mono'>{esc(s.started_at[:19])}</td>"
            f"<td>{s.n_events}</td><td>{' '.join(badges)}</td></tr>"
        )
    body = f"""
    <h2>Recorded sessions ({len(sessions)})</h2>
    <table class="idx">
      <tr><th>Session</th><th>Pilot</th><th>Started (UTC)</th><th>Events</th><th>Flags</th></tr>
      {rows or "<tr><td colspan='5' style='color:var(--dim)'>No recordings yet.</td></tr>"}
    </table>
"""
    return _page("Agent Black Box · Sessions", "<div class='sub'>Session index</div>", body)


def write_report(store, session_id: str, out_path: Optional[str] = None) -> str:
    events = store.load(session_id)
    doc = build_report(events, store.verify(session_id))
    out = out_path or f"{session_id}_report.html"
    with open(out, "w", encoding="utf-8") as f:
        f.write(doc)
    return out
