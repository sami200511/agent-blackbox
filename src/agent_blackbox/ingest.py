"""
Turn arbitrary data files into retrievable text documents.

The retriever (`rank`, `attribute`, `bm25`) works on documents shaped like
``{"doc_id", "source", "text"}``. This module produces exactly that shape from
whatever a user drops into a folder — a spreadsheet, a CSV export, a SQLite
database, a SQL dump, JSON, XML/HTML, a Word file — so any of them becomes
searchable knowledge for a recorded agent. No vector store, no services.

Supported out of the box, PURE STDLIB (matching the recorder's zero-dep core):

    .txt .md .markdown .rst .log .text   plain text, read as-is
    .csv .tsv                            tabular -> one labelled line per row
    .json .jsonl .ndjson                 records flattened to key: value lines
    .xml .html .htm                      tags stripped to readable text
    .ini .cfg .conf .toml .yaml .yml .env   config, read as text
    .sql .pgsql .ddl                     replayed into a temp DB where possible;
                                         pg_dump COPY blocks parsed into tables;
                                         the raw script kept too
    .xlsx                                every sheet, every row (the OOXML zip
                                         is unpacked with stdlib — no openpyxl)
    .docx                                document body text (stdlib zip + XML)
    .sqlite .sqlite3 .db                 every table dumped to labelled rows

A LIVE database is knowledge too — `load_database(url_or_connection)` reflects
a PostgreSQL / MySQL / SQLite server's tables and renders their rows. It speaks
DB-API 2.0, so any driver works; pass your own connection and the core stays
dependency-free.

Optional (used only if the library is importable, never required):

    .pdf                via `pypdf`                    (pip install ...[pdf])
    .xls                via `xlrd`  (old binary Excel)  (...[xls])
    postgresql:// url   via `psycopg` / `psycopg2`      (...[postgres])
    mysql:// url        via `pymysql`                   (...[mysql])

A workbook or database yields ONE document per sheet/table (doc_id like
``sales.xlsx#Summary``) so retrieval can point at the exact tab or table an
answer came from. Anything else that decodes as text is read as plain text;
genuinely binary files are skipped and reported, never guessed at.

    from agent_blackbox import load_documents
    docs = load_documents("knowledge/")     # mixed folder -> retrievable docs
    ranked = rank("refunds over $50", docs)
"""

from __future__ import annotations

import csv
import json
import os
import re
import sqlite3
import zipfile
from xml.etree import ElementTree as ET

# ------------------------------------------------------------ configuration ---
# How many rows of a table/sheet/db to render before truncating. Retrieval only
# needs enough distinctive text to match on; unbounded dumps bloat every prompt.
DEFAULT_MAX_ROWS = 1000

# Text-ish extensions read verbatim, grouped by how we label them.
_PLAINTEXT = {".txt", ".md", ".markdown", ".rst", ".log", ".text", ""}
_CONFIG = {".ini", ".cfg", ".conf", ".toml", ".yaml", ".yml", ".env", ".properties"}
_TABULAR = {".csv", ".tsv"}
_JSON = {".json", ".jsonl", ".ndjson"}
_MARKUP = {".xml", ".html", ".htm", ".xhtml"}
_SQLITE = {".sqlite", ".sqlite3", ".db"}

#: Every extension this module knows how to turn into text without extra deps.
SUPPORTED_EXTENSIONS = (
    _PLAINTEXT | _CONFIG | _TABULAR | _JSON | _MARKUP | _SQLITE
    | {".sql", ".pgsql", ".ddl", ".xlsx", ".docx"}
) - {""}
#: Extensions handled only when an optional library is importable.
OPTIONAL_EXTENSIONS = {".pdf", ".xls"}


class IngestError(Exception):
    """A file matched a known type but could not be parsed."""


# ------------------------------------------------------------ table rendering ---
def _render_table(rows: list, title: str, max_rows: int = DEFAULT_MAX_ROWS) -> str:
    """Render rows (first row = header) as retrieval-friendly ``key: value`` text.

    Keeping each column name next to its value on every row is what lets BM25
    and grounding match on "salary" or "region" instead of a bare number, and
    lets an investigation point at the column an answer leaned on.
    """
    rows = [["" if c is None else str(c).strip() for c in r] for r in rows if r is not None]
    rows = [r for r in rows if any(c for c in r)]        # drop fully blank rows
    if not rows:
        return f"{title}\n(empty)"

    header = rows[0]
    body = rows[1:]
    # Synthesize column names when the sheet has no real header row.
    if not any(header):
        header = [f"col{i + 1}" for i in range(len(header))]

    ncol = max(len(r) for r in rows)
    header = (header + [f"col{i + 1}" for i in range(len(header), ncol)])[:ncol]

    lines = [title, f"Columns: {', '.join(h or f'col{i + 1}' for i, h in enumerate(header))}",
             f"{len(body)} data row(s)."]
    shown = body[:max_rows]
    for r in shown:
        cells = (r + [""] * ncol)[:ncol]
        pairs = [f"{header[i] or f'col{i + 1}'}: {cells[i]}"
                 for i in range(ncol) if cells[i] != ""]
        if pairs:
            lines.append(" | ".join(pairs))
    if len(body) > max_rows:
        lines.append(f"... ({len(body) - max_rows} more row(s) omitted)")
    return "\n".join(lines)


# ------------------------------------------------------------ CSV / TSV ---------
def _read_csv(path: str, ext: str, max_rows: int) -> list:
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        sample = f.read(8192)
        f.seek(0)
        delimiter = "\t" if ext == ".tsv" else None
        if delimiter is None:
            try:
                delimiter = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
            except csv.Error:
                delimiter = ","
        rows = list(csv.reader(f, delimiter=delimiter))
    name = os.path.basename(path)
    return [{"text": _render_table(rows, f"Table from {name}", max_rows)}]


# ------------------------------------------------------------ JSON / JSONL ------
def _flatten(obj, prefix: str = "") -> list:
    """Flatten nested JSON into ``dotted.key: value`` lines for retrieval."""
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            out += _flatten(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out += _flatten(v, f"{prefix}[{i}]")
    else:
        out.append(f"{prefix}: {obj}" if prefix else str(obj))
    return out


def _read_json(path: str, ext: str, max_rows: int) -> list:
    name = os.path.basename(path)
    with open(path, encoding="utf-8", errors="replace") as f:
        raw = f.read()
    if ext in (".jsonl", ".ndjson"):
        records = []
        for line in raw.splitlines():
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    records.append(line)
        obj = records
    else:
        obj = json.loads(raw)

    # A list of flat records reads best as a table; anything else, flatten.
    if isinstance(obj, list) and obj and all(isinstance(r, dict) for r in obj):
        cols = list(dict.fromkeys(k for r in obj for k in r))
        rows = [cols] + [[r.get(c, "") for c in cols] for r in obj]
        text = _render_table(rows, f"Records from {name}", max_rows)
    else:
        lines = _flatten(obj)
        if len(lines) > max_rows:
            lines = lines[:max_rows] + [f"... ({len(lines) - max_rows} more omitted)"]
        text = f"JSON from {name}\n" + "\n".join(lines)
    return [{"text": text}]


# ------------------------------------------------------------ XML / HTML --------
def _strip_markup(path: str) -> list:
    from html.parser import HTMLParser

    class _Text(HTMLParser):
        def __init__(self):
            super().__init__()
            self.parts = []
            self._skip = 0

        def handle_starttag(self, tag, attrs):
            if tag in ("script", "style"):
                self._skip += 1

        def handle_endtag(self, tag):
            if tag in ("script", "style") and self._skip:
                self._skip -= 1

        def handle_data(self, data):
            if not self._skip and data.strip():
                self.parts.append(data.strip())

    with open(path, encoding="utf-8", errors="replace") as f:
        parser = _Text()
        parser.feed(f.read())
    name = os.path.basename(path)
    return [{"text": f"Text extracted from {name}\n" + "\n".join(parser.parts)}]


# ------------------------------------------------------------ SQL script --------
_COPY_RE = re.compile(
    r"^\s*COPY\s+([^\s(]+)\s*\(([^)]*)\)\s+FROM\s+stdin\s*;", re.IGNORECASE)


def _parse_pg_copy(script: str) -> list:
    r"""Parse pg_dump ``COPY table (cols) FROM stdin;`` … ``\.`` blocks.

    Postgres' default plain-text dumps carry their DATA in COPY blocks
    (tab-separated, ``\N`` for NULL) — which SQLite cannot replay — so a
    pg_dump would otherwise land as opaque text. Reading the blocks directly
    turns each one into a searchable table. Returns [(table, header, rows)].
    """
    out = []
    lines = script.splitlines()
    i = 0
    while i < len(lines):
        m = _COPY_RE.match(lines[i])
        if not m:
            i += 1
            continue
        table = m.group(1).strip().replace('"', "").split(".")[-1]
        cols = [c.strip().strip('"') for c in m.group(2).split(",")]
        i += 1
        rows = []
        while i < len(lines) and lines[i] != r"\.":
            cells = ["" if v == r"\N" else v for v in lines[i].split("\t")]
            rows.append(cells)
            i += 1
        out.append((table, cols, rows))
        i += 1  # step past the closing "\."
    return out


def _read_sql(path: str, max_rows: int) -> list:
    with open(path, encoding="utf-8", errors="replace") as f:
        script = f.read()
    name = os.path.basename(path)
    # Best-effort: run the script into an in-memory DB and dump the result, so a
    # dump of INSERTs becomes searchable rows — not just the raw statements.
    # (Works for SQLite-compatible dumps and generic INSERT scripts.)
    docs = []
    try:
        conn = sqlite3.connect(":memory:")
        conn.executescript(script)
        docs = _dump_sqlite_conn(conn, name, max_rows, source_note="rebuilt from SQL script")
        conn.close()
    except Exception:
        docs = []
    # Postgres (pg_dump) carries data in COPY blocks SQLite can't run — parse
    # those directly so a Postgres dump still yields searchable tables.
    have = {d.get("suffix") for d in docs}
    for table, header, rows in _parse_pg_copy(script):
        suffix = f"#{table}"
        if suffix in have:
            continue
        have.add(suffix)
        title = f'Table "{table}" from {name} (pg_dump COPY block)'
        docs.append({"suffix": suffix,
                     "text": _render_table([header] + rows, title, max_rows)})
    # Always keep the script text itself too (DDL, comments, dialect-specific SQL
    # that neither path could execute) so nothing is lost.
    head = script if len(script) <= 20000 else script[:20000] + "\n... (truncated)"
    docs.append({"suffix": "#script", "text": f"SQL script {name}\n{head}"})
    return docs


# ------------------------------------------------------------ SQLite database ---
def _dump_sqlite_conn(conn: sqlite3.Connection, name: str, max_rows: int,
                      source_note: str = "") -> list:
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name")
    tables = [r[0] for r in cur.fetchall()]
    docs = []
    for table in tables:
        try:
            cur.execute(f'SELECT * FROM "{table}" LIMIT {max_rows + 1}')
        except sqlite3.Error:
            continue
        fetched = cur.fetchall()
        cols = [d[0] for d in cur.description] if cur.description else []
        rows = [cols] + [[row[c] for c in cols] for row in fetched[:max_rows]]
        title = f'Table "{table}" from {name}'
        if source_note:
            title += f" ({source_note})"
        text = _render_table(rows, title, max_rows)
        if len(fetched) > max_rows:
            text += f"\n... (more than {max_rows} rows; showing first {max_rows})"
        docs.append({"suffix": f"#{table}", "text": text})
    return docs


def _read_sqlite(path: str, max_rows: int) -> list:
    name = os.path.basename(path)
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        raise IngestError(f"cannot open sqlite database: {exc}") from exc
    try:
        docs = _dump_sqlite_conn(conn, name, max_rows)
    finally:
        conn.close()
    return docs or [{"text": f"SQLite database {name} (no user tables found)"}]


# ------------------------------------------------------ LIVE database (any) -----
# A live PostgreSQL / MySQL / SQLite database becomes retrievable knowledge the
# same way a file does: reflect its tables, SELECT each one, render rows. We
# speak DB-API 2.0, so ANY driver works — bring your own connection and the
# core stays dependency-free; a URL is a convenience that imports the driver.
def _dialect_of(conn) -> str:
    mod = type(conn).__module__.split(".")[0].lower()
    return {"sqlite3": "sqlite", "psycopg2": "postgres", "psycopg": "postgres",
            "pymysql": "mysql", "mysqldb": "mysql",
            "mysql": "mysql"}.get(mod, "generic")


def _connect_url(url: str):
    """Open a connection from a SQLAlchemy-style URL, importing the driver."""
    import importlib
    import urllib.parse

    low = url.lower()
    if low.startswith(("postgres://", "postgresql://")):
        for modname in ("psycopg", "psycopg2"):
            try:
                mod = importlib.import_module(modname)
            except ImportError:
                continue
            return mod.connect(url), "postgres"
        raise IngestError("PostgreSQL URL needs psycopg or psycopg2 "
                          "(pip install 'agent-blackbox[postgres]')")
    if low.startswith("sqlite://"):
        rest = url[len("sqlite://"):]
        path = rest[1:] if rest.startswith("/") else rest      # sqlite:///x -> x
        return sqlite3.connect(path or ":memory:"), "sqlite"
    if low.startswith(("mysql://", "mysql+pymysql://", "mariadb://")):
        try:
            import pymysql
        except ImportError as exc:
            raise IngestError("MySQL URL needs pymysql "
                              "(pip install 'agent-blackbox[mysql]')") from exc
        u = urllib.parse.urlparse(url)
        return pymysql.connect(
            host=u.hostname or "localhost", port=u.port or 3306,
            user=u.username or "", password=u.password or "",
            database=u.path.lstrip("/") or None), "mysql"
    raise IngestError(f"unsupported database URL scheme in {url!r}")


def _reflect_tables(cur, dialect: str, schema: str = None) -> list:
    """List (schema, table) pairs for user tables in a live database."""
    if dialect == "sqlite":
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' "
                    "AND name NOT LIKE 'sqlite_%' ORDER BY name")
        return [(None, r[0]) for r in cur.fetchall()]
    # Postgres, MySQL and most engines expose information_schema.
    q = ("SELECT table_schema, table_name FROM information_schema.tables "
         "WHERE table_type='BASE TABLE' AND table_schema NOT IN "
         "('pg_catalog','information_schema','mysql','performance_schema','sys')")
    if schema:
        cur.execute(q + " AND table_schema = %s ORDER BY table_schema, table_name",
                    (schema,))
    else:
        cur.execute(q + " ORDER BY table_schema, table_name")
    return [(r[0], r[1]) for r in cur.fetchall()]


def _quote_table(dialect: str, schema: str, table: str) -> str:
    if dialect == "mysql":
        q = lambda s: "`" + str(s).replace("`", "``") + "`"
    else:
        q = lambda s: '"' + str(s).replace('"', '""') + '"'
    return f"{q(schema)}.{q(table)}" if schema else q(table)


def load_database(source, tables: list = None, schema: str = None,
                  label: str = None, max_rows: int = DEFAULT_MAX_ROWS) -> list:
    """Load a live SQL database's tables as retrieval documents.

    `source` is either a connection URL — ``postgresql://user:pw@host/db``,
    ``mysql://…``, ``sqlite:///path.db`` — or an already-open DB-API 2.0
    connection (psycopg, psycopg2, pymysql, sqlite3, …); with a connection you
    supply the driver, so the recorder core stays dependency-free. A URL is
    opened (and closed) for you; a connection you pass is left open.

    Returns one ``{"doc_id", "source", "text"}`` per table, doc_id like
    ``shop-db#public.orders``, ready for `rank()` / `attribute()`. Pass
    `tables` (names or ``"schema.table"``) to skip reflection and pick exactly
    what to expose; `schema` to restrict reflection to one schema.

        from agent_blackbox import load_database, rank
        docs = load_database("postgresql://ro@db.internal/shop", schema="public")
        hits = rank("orders refunded last week", docs)
    """
    if isinstance(source, str):
        conn, dialect = _connect_url(source)
        owned = True
        if label is None:
            import urllib.parse
            label = urllib.parse.urlparse(source).path.lstrip("/") or "database"
    else:
        conn, dialect, owned = source, _dialect_of(source), False
        label = label or "database"

    try:
        cur = conn.cursor()
        if tables:
            pairs = []
            for t in tables:
                if isinstance(t, (tuple, list)):
                    pairs.append((t[0], t[1]))
                elif "." in t and dialect != "sqlite":
                    s, _, name = t.partition(".")
                    pairs.append((s, name))
                else:
                    pairs.append((None, t))
        else:
            pairs = _reflect_tables(cur, dialect, schema)

        docs = []
        for sch, table in pairs:
            try:
                cur.execute(f"SELECT * FROM {_quote_table(dialect, sch, table)} "
                            f"LIMIT {max_rows + 1}")
                fetched = cur.fetchall()
            except Exception:
                continue        # a table we can't read must not sink the rest
            cols = [d[0] for d in cur.description] if cur.description else []
            body = [list(r) for r in fetched[:max_rows]]
            qualified = f"{sch}.{table}" if sch else table
            title = f'Table "{qualified}" from {label} ({dialect})'
            text = _render_table([cols] + body, title, max_rows)
            if len(fetched) > max_rows:
                text += f"\n... (more than {max_rows} rows; showing first {max_rows})"
            docs.append({"doc_id": f"{label}#{qualified}",
                         "source": f"{label}#{qualified}", "text": text})
    finally:
        if owned:
            conn.close()
    return docs


# ------------------------------------------------------------ XLSX (stdlib) -----
def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _col_index(ref: str) -> int:
    """'B7' -> 1 (0-based column index), from the leading letters of a cell ref."""
    idx = 0
    for ch in ref:
        if ch.isalpha():
            idx = idx * 26 + (ord(ch.upper()) - 64)
        else:
            break
    return idx - 1


def _read_xlsx(path: str, max_rows: int) -> list:
    """Read every sheet of an .xlsx by unzipping the OOXML — no openpyxl needed.

    An .xlsx is a zip of XML parts: shared strings live in xl/sharedStrings.xml,
    sheet order/names in xl/workbook.xml, and the r:id -> file map in the rels.
    """
    name = os.path.basename(path)
    try:
        zf = zipfile.ZipFile(path)
    except zipfile.BadZipFile as exc:
        raise IngestError(f"not a valid .xlsx (zip) file: {exc}") from exc

    with zf:
        names = set(zf.namelist())

        shared: list = []
        if "xl/sharedStrings.xml" in names:
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in root:
                shared.append("".join(t.text or "" for t in si.iter()
                                      if _local(t.tag) == "t"))

        # sheet name -> r:id, then r:id -> part path via the workbook rels.
        rid_to_target = {}
        if "xl/_rels/workbook.xml.rels" in names:
            for rel in ET.fromstring(zf.read("xl/_rels/workbook.xml.rels")):
                target = rel.attrib.get("Target", "")
                if not target.startswith("/"):
                    target = "xl/" + target.lstrip("./")
                rid_to_target[rel.attrib.get("Id")] = target.lstrip("/")

        sheets = []   # (sheet_name, part_path)
        if "xl/workbook.xml" in names:
            wb = ET.fromstring(zf.read("xl/workbook.xml"))
            for sh in wb.iter():
                if _local(sh.tag) != "sheet":
                    continue
                title = sh.attrib.get("name", f"Sheet{len(sheets) + 1}")
                rid = next((v for k, v in sh.attrib.items()
                            if _local(k) == "id" or k.endswith("}id")), None)
                part = rid_to_target.get(rid)
                if part and part in names:
                    sheets.append((title, part))
        if not sheets:  # fall back to file order if the workbook part is odd
            sheets = [(os.path.splitext(os.path.basename(p))[0], p)
                      for p in sorted(names) if p.startswith("xl/worksheets/")
                      and p.endswith(".xml")]

        docs = []
        for title, part in sheets:
            grid = _xlsx_sheet_rows(zf.read(part), shared, max_rows)
            docs.append({"suffix": f"#{title}",
                         "text": _render_table(grid, f'Sheet "{title}" from {name}',
                                               max_rows)})
    return docs or [{"text": f"Workbook {name} (no sheets found)"}]


def _xlsx_sheet_rows(sheet_xml: bytes, shared: list, max_rows: int) -> list:
    """Turn one worksheet's XML into a rectangular list-of-lists (values)."""
    grid = []
    root = ET.fromstring(sheet_xml)
    for row in root.iter():
        if _local(row.tag) != "row":
            continue
        cells = {}
        for c in row:
            if _local(c.tag) != "c":
                continue
            ref = c.attrib.get("r", "")
            col = _col_index(ref) if ref else len(cells)
            ctype = c.attrib.get("t")
            value = ""
            if ctype == "inlineStr":
                value = "".join(t.text or "" for t in c.iter() if _local(t.tag) == "t")
            else:
                v = next((ch for ch in c if _local(ch.tag) == "v"), None)
                raw = v.text if v is not None else None
                if raw is None:
                    value = ""
                elif ctype == "s":                       # shared-string index
                    try:
                        value = shared[int(raw)]
                    except (ValueError, IndexError):
                        value = raw
                elif ctype == "b":                       # boolean
                    value = "TRUE" if raw == "1" else "FALSE"
                else:
                    value = raw
            cells[col] = value
        width = (max(cells) + 1) if cells else 0
        grid.append([cells.get(i, "") for i in range(width)])
        if len(grid) > max_rows + 1:                     # +1 for the header row
            break
    return grid


# ------------------------------------------------------------ DOCX (stdlib) -----
def _read_docx(path: str) -> list:
    name = os.path.basename(path)
    try:
        with zipfile.ZipFile(path) as zf:
            xml = zf.read("word/document.xml")
    except (zipfile.BadZipFile, KeyError) as exc:
        raise IngestError(f"not a valid .docx file: {exc}") from exc
    root = ET.fromstring(xml)
    paras = []
    for p in root.iter():
        if _local(p.tag) != "p":
            continue
        text = "".join(t.text or "" for t in p.iter() if _local(t.tag) == "t")
        if text.strip():
            paras.append(text.strip())
    return [{"text": f"Document {name}\n" + "\n".join(paras)}]


# ------------------------------------------------------------ optional deps -----
def _read_pdf(path: str) -> list:
    try:
        import pypdf
    except ImportError as exc:  # pragma: no cover - depends on env
        raise IngestError("PDF support needs `pypdf` (pip install pypdf)") from exc
    name = os.path.basename(path)
    reader = pypdf.PdfReader(path)
    pages = [(pg.extract_text() or "").strip() for pg in reader.pages]
    return [{"text": f"PDF {name}\n" + "\n\n".join(p for p in pages if p)}]


def _read_xls(path: str, max_rows: int) -> list:
    try:
        import xlrd
    except ImportError as exc:  # pragma: no cover - depends on env
        raise IngestError("legacy .xls support needs `xlrd` (pip install xlrd)") from exc
    name = os.path.basename(path)
    book = xlrd.open_workbook(path)
    docs = []
    for sheet in book.sheets():
        rows = [[sheet.cell_value(r, c) for c in range(sheet.ncols)]
                for r in range(sheet.nrows)]
        docs.append({"suffix": f"#{sheet.name}",
                     "text": _render_table(rows, f'Sheet "{sheet.name}" from {name}',
                                           max_rows)})
    return docs


# ------------------------------------------------------------ plain text --------
def _read_text(path: str, label: str) -> list:
    with open(path, "rb") as f:
        raw = f.read()
    if b"\x00" in raw[:4096]:
        raise IngestError("looks binary (NUL bytes) — not text")
    text = raw.decode("utf-8", errors="replace")
    name = os.path.basename(path)
    prefix = f"{label} {name}\n" if label else ""
    return [{"text": prefix + text}]


# ------------------------------------------------------------ dispatch ----------
def _handlers(ext: str):
    if ext in _TABULAR:
        return lambda p, mr: _read_csv(p, ext, mr)
    if ext in _JSON:
        return lambda p, mr: _read_json(p, ext, mr)
    if ext in _MARKUP:
        return lambda p, mr: _strip_markup(p)
    if ext in _SQLITE:
        return lambda p, mr: _read_sqlite(p, mr)
    if ext in (".sql", ".pgsql", ".ddl"):
        return lambda p, mr: _read_sql(p, mr)
    if ext == ".xlsx":
        return lambda p, mr: _read_xlsx(p, mr)
    if ext == ".docx":
        return lambda p, mr: _read_docx(p)
    if ext == ".pdf":
        return lambda p, mr: _read_pdf(p)
    if ext == ".xls":
        return lambda p, mr: _read_xls(p, mr)
    if ext in _CONFIG:
        return lambda p, mr: _read_text(p, "Config file")
    if ext in _PLAINTEXT:
        return lambda p, mr: _read_text(p, "")
    return None


def load_file(path: str, max_rows: int = DEFAULT_MAX_ROWS,
              doc_id: str = None, source: str = None) -> list:
    """Convert one file into a list of retrieval documents.

    Returns ``[{"doc_id", "source", "text"}, ...]`` — usually one document, but
    a workbook/database/multi-table SQL file yields one per sheet/table with a
    ``doc_id`` like ``sales.xlsx#Summary``. A file whose extension isn't
    recognised is read as plain text if it decodes, else skipped (empty list).

    Raises IngestError only when a KNOWN type is present but unparseable.
    """
    ext = os.path.splitext(path)[1].lower()
    base_id = doc_id or os.path.basename(path)
    base_src = source if source is not None else path

    handler = _handlers(ext)
    if handler is None:                       # unknown extension: try as text
        try:
            parts = _read_text(path, "")
        except (IngestError, OSError):
            return []
    else:
        parts = handler(path, max_rows)

    docs = []
    for part in parts:
        suffix = part.get("suffix", "")
        docs.append({
            "doc_id": base_id + suffix,
            "source": base_src + suffix,
            "text": part["text"],
        })
    return docs


def load_documents(directory: str, recursive: bool = True,
                   max_rows: int = DEFAULT_MAX_ROWS,
                   skip_hidden: bool = True) -> list:
    """Load every readable file under `directory` as retrieval documents.

    Walks the folder, converts each supported file (and anything else that
    decodes as text), and returns a flat list of ``{"doc_id", "source", "text"}``
    ready to hand straight to `rank()` / `attribute()`. `doc_id`/`source` are
    made relative to `directory` so reports show tidy paths. Unreadable or truly
    binary files are skipped, not errored on, so one bad file never blocks the
    rest.
    """
    docs = []
    for root, dirs, files in os.walk(directory):
        if skip_hidden:
            dirs[:] = [d for d in dirs if not d.startswith(".")]
        for fname in sorted(files):
            if skip_hidden and fname.startswith("."):
                continue
            path = os.path.join(root, fname)
            rel = os.path.relpath(path, directory)
            try:
                docs.extend(load_file(path, max_rows=max_rows, doc_id=rel, source=rel))
            except (IngestError, OSError, UnicodeError):
                continue        # one unreadable file must not sink the folder
        if not recursive:
            break
    docs.sort(key=lambda d: d["doc_id"])
    return docs
