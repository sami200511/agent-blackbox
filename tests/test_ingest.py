"""Ingesting arbitrary data files (CSV, Excel, SQLite, SQL, JSON, ...) into
retrievable documents, and searching the resulting knowledge base."""

import sqlite3
import zipfile

from agent_blackbox import (
    SUPPORTED_EXTENSIONS,
    load_database,
    load_documents,
    load_file,
    rank,
)

# --------------------------------------------------------------- builders -----
_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


def _make_xlsx(path):
    """Write a minimal but real .xlsx (OOXML zip) with one 'Sales' sheet."""
    shared = ["Region", "Product", "Revenue", "North", "Widget", "South"]
    sst = (f'<sst xmlns="{_NS}" count="6" uniqueCount="6">'
           + "".join(f"<si><t>{s}</t></si>" for s in shared) + "</sst>")
    workbook = (f'<workbook xmlns="{_NS}" xmlns:r="{_R}"><sheets>'
                '<sheet name="Sales" sheetId="1" r:id="rId1"/></sheets></workbook>')
    rels = ('<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/'
            'relationships"><Relationship Id="rId1" Type="http://schemas.'
            'openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            'Target="worksheets/sheet1.xml"/></Relationships>')
    sheet = (f'<worksheet xmlns="{_NS}"><sheetData>'
             '<row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c>'
             '<c r="C1" t="s"><v>2</v></c></row>'
             '<row r="2"><c r="A2" t="s"><v>3</v></c><c r="B2" t="s"><v>4</v></c>'
             '<c r="C2"><v>1200</v></c></row>'
             '<row r="3"><c r="A3" t="s"><v>5</v></c><c r="B3" t="s"><v>4</v></c>'
             '<c r="C3"><v>900</v></c></row>'
             '</sheetData></worksheet>')
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("xl/sharedStrings.xml", sst)
        z.writestr("xl/workbook.xml", workbook)
        z.writestr("xl/_rels/workbook.xml.rels", rels)
        z.writestr("xl/worksheets/sheet1.xml", sheet)


def _make_docx(path):
    body = (f'<w:document xmlns:w="{_R}"><w:body>'
            '<w:p><w:r><w:t>Refund window is 30 days from delivery.</w:t></w:r></w:p>'
            '<w:p><w:r><w:t>Warranty covers repairs after that.</w:t></w:r></w:p>'
            '</w:body></w:document>')
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("word/document.xml", body)


# ---------------------------------------------------------------- per-type ----
def test_csv_becomes_labelled_rows(tmp_path):
    p = tmp_path / "orders.csv"
    p.write_text("order_id,item,price\nDS-1001,laptop,1200\nDS-1002,phone,800\n")
    (doc,) = load_file(str(p))
    assert doc["doc_id"] == "orders.csv"
    # column names stay attached to values so retrieval can match on them
    assert "item: laptop" in doc["text"]
    assert "price: 1200" in doc["text"]
    assert "Columns: order_id, item, price" in doc["text"]


def test_tsv_delimiter(tmp_path):
    p = tmp_path / "t.tsv"
    p.write_text("name\tregion\nAlice\tNorth\n")
    (doc,) = load_file(str(p))
    assert "name: Alice" in doc["text"] and "region: North" in doc["text"]


def test_json_records_render_as_table(tmp_path):
    p = tmp_path / "people.json"
    p.write_text('[{"name": "Alice", "dept": "Eng"}, {"name": "Bob", "dept": "Sales"}]')
    (doc,) = load_file(str(p))
    assert "name: Alice" in doc["text"] and "dept: Sales" in doc["text"]


def test_jsonl_records(tmp_path):
    p = tmp_path / "log.jsonl"
    p.write_text('{"event": "login", "user": "amy"}\n{"event": "logout", "user": "amy"}\n')
    (doc,) = load_file(str(p))
    assert "event: login" in doc["text"] and "user: amy" in doc["text"]


def test_html_tags_are_stripped(tmp_path):
    p = tmp_path / "page.html"
    p.write_text("<html><head><style>.x{color:red}</style></head>"
                 "<body><h1>Shipping</h1><p>Free over $50.</p>"
                 "<script>alert(1)</script></body></html>")
    (doc,) = load_file(str(p))
    assert "Shipping" in doc["text"] and "Free over $50." in doc["text"]
    assert "color:red" not in doc["text"] and "alert" not in doc["text"]


def test_sqlite_yields_one_doc_per_table(tmp_path):
    p = tmp_path / "shop.db"
    conn = sqlite3.connect(str(p))
    conn.execute("CREATE TABLE products (sku TEXT, name TEXT, price REAL)")
    conn.execute("INSERT INTO products VALUES ('SKU1', 'Nimbus laptop', 1200)")
    conn.execute("CREATE TABLE staff (id INTEGER, name TEXT)")
    conn.execute("INSERT INTO staff VALUES (1, 'Dana')")
    conn.commit()
    conn.close()

    docs = load_file(str(p))
    ids = {d["doc_id"] for d in docs}
    assert ids == {"shop.db#products", "shop.db#staff"}
    prod = next(d for d in docs if d["doc_id"].endswith("#products"))
    assert "name: Nimbus laptop" in prod["text"] and "price: 1200" in prod["text"]


def test_sql_script_is_rebuilt_and_kept(tmp_path):
    p = tmp_path / "seed.sql"
    p.write_text(
        "CREATE TABLE warranty (item TEXT, months INTEGER);\n"
        "INSERT INTO warranty VALUES ('laptop', 24);\n"
        "INSERT INTO warranty VALUES ('phone', 12);\n"
    )
    docs = load_file(str(p))
    ids = {d["doc_id"] for d in docs}
    # one doc per rebuilt table + the raw script itself
    assert "seed.sql#warranty" in ids and "seed.sql#script" in ids
    table = next(d for d in docs if d["doc_id"].endswith("#warranty"))
    assert "item: laptop" in table["text"] and "months: 24" in table["text"]


def test_postgres_dump_copy_blocks_become_tables(tmp_path):
    # pg_dump plain format: data lives in COPY ... FROM stdin blocks (tab-sep,
    # \N = NULL), which SQLite cannot replay — parsed directly here.
    p = tmp_path / "pgdump.sql"
    p.write_text(
        "--\n-- PostgreSQL database dump\n--\n"
        "SET statement_timeout = 0;\n"
        "CREATE TABLE public.orders (id integer, customer text, total numeric);\n"
        "COPY public.orders (id, customer, total) FROM stdin;\n"
        "1\tAlice\t120.50\n"
        "2\tBob\t\\N\n"
        "\\.\n"
        "ALTER TABLE ONLY public.orders ADD CONSTRAINT orders_pkey PRIMARY KEY (id);\n"
    )
    docs = load_file(str(p))
    ids = {d["doc_id"] for d in docs}
    assert "pgdump.sql#orders" in ids and "pgdump.sql#script" in ids
    table = next(d for d in docs if d["doc_id"].endswith("#orders"))
    assert "customer: Alice" in table["text"]
    assert "total: 120.50" in table["text"]
    assert "Columns: id, customer, total" in table["text"]


def test_load_database_from_connection_object(tmp_path):
    # A live database (here SQLite, but the path is driver-agnostic DB-API):
    # bring your own connection, get one document per table back.
    p = tmp_path / "shop.sqlite"
    conn = sqlite3.connect(str(p))
    conn.execute("CREATE TABLE customers (id INTEGER, name TEXT, city TEXT)")
    conn.execute("INSERT INTO customers VALUES (1, 'Carla', 'Berlin')")
    conn.execute("CREATE TABLE invoices (num TEXT, amount REAL)")
    conn.execute("INSERT INTO invoices VALUES ('INV-9', 42.0)")
    conn.commit()

    docs = load_database(conn, label="shop")
    conn.close()
    ids = {d["doc_id"] for d in docs}
    assert ids == {"shop#customers", "shop#invoices"}
    cust = next(d for d in docs if d["doc_id"].endswith("#customers"))
    assert "name: Carla" in cust["text"] and "city: Berlin" in cust["text"]


def test_load_database_from_url_and_table_filter(tmp_path):
    p = tmp_path / "kb.db"
    conn = sqlite3.connect(str(p))
    conn.execute("CREATE TABLE faq (q TEXT, a TEXT)")
    conn.execute("INSERT INTO faq VALUES ('return window?', '30 days')")
    conn.execute("CREATE TABLE secret (x TEXT)")
    conn.execute("INSERT INTO secret VALUES ('hidden')")
    conn.commit()
    conn.close()

    # sqlite:///<path> URL, and expose only the faq table
    docs = load_database(f"sqlite:///{p}", tables=["faq"], label="kb")
    assert {d["doc_id"] for d in docs} == {"kb#faq"}
    assert "a: 30 days" in docs[0]["text"]

    ranked = rank("what is the return window", docs)
    assert ranked[0]["doc_id"] == "kb#faq"


def test_xlsx_read_with_stdlib_only(tmp_path):
    p = tmp_path / "sales.xlsx"
    _make_xlsx(str(p))
    (doc,) = load_file(str(p))
    assert doc["doc_id"] == "sales.xlsx#Sales"
    assert "Region: North" in doc["text"]
    assert "Product: Widget" in doc["text"]
    assert "Revenue: 1200" in doc["text"]


def test_docx_body_text(tmp_path):
    p = tmp_path / "policy.docx"
    _make_docx(str(p))
    (doc,) = load_file(str(p))
    assert "Refund window is 30 days from delivery." in doc["text"]
    assert "Warranty covers repairs" in doc["text"]


def test_plain_and_config_files(tmp_path):
    (tmp_path / "note.md").write_text("# Heading\nSome policy text about refunds.")
    (tmp_path / "app.yaml").write_text("service: shop\nrefund_days: 30\n")
    md = load_file(str(tmp_path / "note.md"))[0]
    yml = load_file(str(tmp_path / "app.yaml"))[0]
    assert "policy text about refunds" in md["text"]
    assert "refund_days: 30" in yml["text"]


def test_binary_file_is_skipped(tmp_path):
    p = tmp_path / "blob.bin"
    p.write_bytes(b"\x00\x01\x02\x03PNG\x00\x00")
    assert load_file(str(p)) == []


def test_unknown_extension_read_as_text(tmp_path):
    p = tmp_path / "data.weird"
    p.write_text("just some plain readable text")
    (doc,) = load_file(str(p))
    assert "plain readable text" in doc["text"]


# ---------------------------------------------------------- whole folder ------
def test_load_documents_mixed_folder_is_searchable(tmp_path):
    (tmp_path / "refunds.md").write_text(
        "Refund policy: 30 day window from delivery for electronics.")
    (tmp_path / "orders.csv").write_text(
        "order_id,item,price\nDS-1001,ergonomic keyboard,90\n")
    _make_xlsx(tmp_path / "sales.xlsx")
    sub = tmp_path / "db"
    sub.mkdir()
    conn = sqlite3.connect(str(sub / "shop.db"))
    conn.execute("CREATE TABLE warranty (item TEXT, months INTEGER)")
    conn.execute("INSERT INTO warranty VALUES ('laptop', 24)")
    conn.commit()
    conn.close()

    docs = load_documents(str(tmp_path))
    ids = {d["doc_id"] for d in docs}
    assert "refunds.md" in ids
    assert "orders.csv" in ids
    assert "sales.xlsx#Sales" in ids
    assert "db/shop.db#warranty" in ids            # nested + relative doc_id

    # the CSV is the doc that answers a keyword-only query about its contents
    ranked = rank("ergonomic keyboard price", docs)
    assert ranked[0]["doc_id"] == "orders.csv"

    # and the spreadsheet wins a query about its own rows
    ranked = rank("widget revenue north region", docs)
    assert ranked[0]["doc_id"] == "sales.xlsx#Sales"


def test_supported_extensions_advertised():
    for ext in (".csv", ".tsv", ".json", ".jsonl", ".xlsx", ".docx",
                ".sqlite", ".db", ".sql", ".xml", ".html", ".yaml"):
        assert ext in SUPPORTED_EXTENSIONS


def test_max_rows_truncates(tmp_path):
    rows = "order_id,item\n" + "".join(f"DS-{i},item{i}\n" for i in range(50))
    p = tmp_path / "big.csv"
    p.write_text(rows)
    (doc,) = load_file(str(p), max_rows=10)
    assert "more row(s) omitted" in doc["text"]
