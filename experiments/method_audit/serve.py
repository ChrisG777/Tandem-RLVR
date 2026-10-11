# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Local method-pair audit UI with durable SQLite reviews and family-safe exports."""

from __future__ import annotations
import argparse
import hashlib
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import sqlite3
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Audit:
    """Human judgments only; no verdict or empirical capability claim is inferred."""

    status: str = "pending"
    elegant: str = ""
    elementary: str = ""
    correct_elegant: bool = False
    correct_elementary: bool = False
    distinct: bool = False
    methods_reviewed: bool = False
    answer: str = ""
    notes: str = ""
    capability: str = "untested"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "outputs/method-audit")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    store = AuditStore(args.data)
    handler = make_handler(store, ROOT / "ui/method-audit", args.port)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    print(f"Method audit: http://127.0.0.1:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


class AuditStore:
    """Immutable source catalog plus transactional, revision-checked user audits."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        catalog_bytes = (root / "catalog.json").read_bytes()
        self.catalog = json.loads(catalog_bytes)
        self.catalog_sha256 = hashlib.sha256(catalog_bytes).hexdigest()
        self.problems = {p["id"]: p for p in self.catalog["problems"]}
        self.methods = {m["id"]: m for m in self.catalog["methods"]}
        taxonomy_path = root / "machine-review/taxonomy.json"
        taxonomy = (
            json.loads(taxonomy_path.read_text()) if taxonomy_path.exists() else {}
        )
        self.method_definitions = {m["id"]: m for m in taxonomy.get("methods", [])}
        self.db = root / "audits.sqlite3"
        with self.connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS audits (id TEXT PRIMARY KEY, revision INTEGER NOT NULL, body TEXT NOT NULL, updated TEXT NOT NULL)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS settings (id INTEGER PRIMARY KEY CHECK(id=1), revision INTEGER NOT NULL, body TEXT NOT NULL)"
            )
            conn.execute(
                "INSERT OR IGNORE INTO settings VALUES (1,0,?)",
                (json.dumps({k: v["split"] for k, v in self.methods.items()}),),
            )
            revision, body = conn.execute(
                "SELECT revision,body FROM settings WHERE id=1"
            ).fetchone()
            splits = json.loads(body)
            if set(splits) - set(self.methods):
                raise ValueError(
                    "Taxonomy removed saved methods; resolve the migration explicitly."
                )
            new_methods = set(self.methods) - set(splits)
            if new_methods:
                splits.update({key: self.methods[key]["split"] for key in new_methods})
                conn.execute(
                    "UPDATE settings SET revision=?,body=? WHERE id=1",
                    (revision + 1, json.dumps(splits)),
                )

    def connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db, timeout=20)

    def get_audit(self, pid: str) -> dict:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT revision,body,updated FROM audits WHERE id=?", (pid,)
            ).fetchone()
        if row:
            audit = dict(json.loads(row[1]), revision=row[0], updated=row[2])
            if audit.get("source_signature") != source_signature(self.problems[pid]):
                for key in (
                    "methods_reviewed",
                    "correct_elegant",
                    "correct_elementary",
                    "distinct",
                ):
                    audit[key] = False
            return audit
        return dict(
            asdict(Audit()),
            revision=0,
            methods={},
            additional_methods=[],
            route_text={},
        )

    def settings(self) -> dict:
        with self.connect() as conn:
            revision, body = conn.execute(
                "SELECT revision,body FROM settings WHERE id=1"
            ).fetchone()
        return {"revision": revision, "splits": json.loads(body)}

    def save_settings(self, payload: dict) -> dict:
        splits = payload.get("splits")
        if (
            not isinstance(splits, dict)
            or set(splits) != set(self.methods)
            or any(v not in ("train", "validation", "test") for v in splits.values())
        ):
            raise ValueError("Supply exactly one valid split for every method.")
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            rev = conn.execute("SELECT revision FROM settings WHERE id=1").fetchone()[0]
            if payload.get("revision") != rev:
                raise Conflict("Holdouts changed in another tab. Reload before saving.")
            conn.execute(
                "UPDATE settings SET revision=?,body=? WHERE id=1",
                (rev + 1, json.dumps(splits)),
            )
        return self.settings()

    def save_audit(self, pid: str, payload: dict) -> dict:
        """Validate a complete audit and atomically save, refusing stale revisions."""
        p = self.problems[pid]
        validate_audit(payload, p, set(self.methods))
        body = {k: v for k, v in payload.items() if k not in ("revision", "updated")}
        body["source_signature"] = source_signature(p)
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT revision FROM audits WHERE id=?", (pid,)
            ).fetchone()
            rev = row[0] if row else 0
            if payload.get("revision") != rev:
                raise Conflict(
                    "This audit changed in another tab. Reload to preserve both edits."
                )
            now = datetime.now(timezone.utc).isoformat()
            conn.execute(
                "INSERT OR REPLACE INTO audits VALUES (?,?,?,?)",
                (pid, rev + 1, json.dumps(body), now),
            )
        return dict(body, revision=rev + 1, updated=now)

    def machine_reviews(self) -> dict:
        """Read AI evidence separately from user audits; refreshed on each request."""
        try:
            from .review import load_reviews, review_blockers
        except ImportError:
            from review import load_reviews, review_blockers
        reviews = load_reviews(self.root, self.problems)
        for pid, review in reviews.items():
            review["export_blockers"] = (
                review_blockers(self.problems[pid], review, set(self.methods))
                if pid in self.problems
                else ["Problem absent from current catalog"]
            )
        return reviews

    def machine_export(self) -> dict:
        """Export AI-reviewed pairs through the independent strict evidence gate."""
        try:
            from .review import machine_export
        except ImportError:
            from review import machine_export
        settings = self.settings()
        result = machine_export(self.root, self.problems, settings["splits"])
        result.update(
            schema_version=1, catalog_sha256=self.catalog_sha256, settings=settings
        )
        return result

    def elegance_review(self, pid: str, review: dict | None = None) -> dict | None:
        """Read all-route style labels independently of mathematical approval."""
        try:
            from .elegance import load_judgment
        except ImportError:
            from elegance import load_judgment
        review = review or self.machine_reviews().get(pid)
        if not review:
            return None
        try:
            return load_judgment(self.root, self.problems[pid], review)
        except ValueError as exc:
            return {"error": str(exc)}

    def elegance_export(self) -> dict:
        """Return every labeled path; fail until all accepted problems are labeled."""
        try:
            from .elegance import export_dataset
        except ImportError:
            from elegance import export_dataset
        result = export_dataset(self.root, self.problems, self.machine_reviews())
        result["catalog_sha256"] = self.catalog_sha256
        return result

    def index(self) -> dict:
        reviews = self.machine_reviews()
        style_reviews = [
            self.elegance_review(pid, review)
            for pid, review in reviews.items()
            if review.get("status") == "accepted"
        ]
        audits = self.saved_audits()
        splits = self.settings()
        rows = []
        for p in self.problems.values():
            a = audits.get(p["id"], {})
            machine = reviews.get(p["id"], {})
            if machine.get("route_assessments"):
                reviewed_tags = {
                    r["route_id"]: r.get("methods", [])
                    for r in machine["route_assessments"]
                }
                tags = set(a.get("additional_methods", []))
                for route in p["routes"]:
                    tags.update(
                        a.get("methods", {}).get(
                            route["id"],
                            reviewed_tags.get(route["id"], route["methods"]),
                        )
                    )
            else:
                tags = effective_methods(p, a)
            rows.append(
                {
                    k: p[k]
                    for k in (
                        "id",
                        "title",
                        "subject",
                        "flags",
                        "priority",
                        "shortlist",
                    )
                }
                | {
                    "preview": p["question"][:280],
                    "sources": list(dict.fromkeys(s["dataset"] for s in p["sources"])),
                    "methods": sorted(tags),
                    "route_count": len(p["routes"]),
                    "full_routes": sum(r["kind"] != "summary" for r in p["routes"]),
                    "status": a.get("status", "pending"),
                    "split": assign_split(tags, splits["splits"]),
                    "eligible": not export_blockers(p, a),
                    "updated": a.get("updated", ""),
                    "machine_status": reviews.get(p["id"], {}).get(
                        "status", "unreviewed"
                    ),
                    "machine_eligible": reviews.get(p["id"], {}).get("status")
                    == "accepted"
                    and not reviews.get(p["id"], {}).get(
                        "export_blockers", ["Missing gate evidence"]
                    ),
                }
            )
        return {
            "problems": rows,
            "methods": [
                dict(
                    m,
                    **{
                        key: self.method_definitions.get(m["id"], {}).get(key, "")
                        for key in (
                            "definition",
                            "classification",
                            "related_holdout_groups",
                        )
                    },
                )
                for m in self.catalog["methods"]
            ],
            "sources": self.catalog["sources"],
            "notes": self.catalog["notes"],
            "summary": self.catalog["summary"],
            "settings": splits,
            "elegance_summary": {
                "expected": len(style_reviews),
                "complete": sum(bool(r and not r.get("error")) for r in style_reviews),
            },
            "machine_summary": {
                "total": sum(p["shortlist"] for p in self.problems.values()),
                "reviewed": len(reviews),
                "accepted": sum(
                    r.get("status") == "accepted" for r in reviews.values()
                ),
                "export_ready": sum(
                    r.get("status") == "accepted"
                    and not r.get("export_blockers", ["Missing gate evidence"])
                    for r in reviews.values()
                ),
            },
        }

    def export(self) -> dict:
        """Export only explicitly accepted, complete pair audits; quarantine all others.

        Every reviewed route contributes to exclusion from train, including unused
        alternate solutions. Unreviewed/unknown methods block export. Held-out test
        takes precedence over validation. Lexical misses still require human review.
        """
        settings = self.settings()
        buckets = {"train": [], "validation": [], "test": []}
        withheld = []
        for pid, a in self.saved_audits().items():
            if pid not in self.problems:
                withheld.append(
                    {"id": pid, "reasons": ["Problem absent from current catalog"]}
                )
                continue
            p = self.problems[pid]
            if a["status"] != "accepted":
                continue
            blockers = export_blockers(p, a)
            tags = effective_methods(p, a)
            if blockers:
                withheld.append({"id": p["id"], "reasons": blockers})
                continue
            split = assign_split(tags, settings["splits"])
            buckets[split].append(
                {
                    "problem": p,
                    "audit": a,
                    "methods": sorted(tags),
                    "split": split,
                    "capability_evidence": a.get("capability", "untested"),
                }
            )
        return {
            "schema_version": 1,
            "catalog_sha256": self.catalog_sha256,
            "settings": settings,
            "partitions": buckets,
            "blocked_accepted": withheld,
            "notes": [
                "Human-audited pairs; this export does not certify senior/junior capability differences.",
                "No training item has a currently labeled validation/test method in any retained route.",
                "Near-duplicate and semantic method leakage require review beyond lexical proposals.",
            ],
        }

    def saved_audits(self) -> dict[str, dict]:
        """Read saved records in one snapshot, retaining audits of removed problems."""
        with self.connect() as conn:
            return {
                pid: dict(json.loads(body), revision=rev, updated=updated)
                for pid, rev, body, updated in conn.execute("SELECT * FROM audits")
            }


class Conflict(ValueError):
    pass


def validate_audit(a: dict, p: dict, method_ids: set[str]) -> None:
    allowed = set(asdict(Audit())) | {
        "revision",
        "updated",
        "methods",
        "additional_methods",
        "route_text",
        "source_signature",
    }
    if set(a) - allowed:
        raise ValueError("Unknown audit field")
    if a.get("status") not in ("pending", "accepted", "rejected", "needs_work"):
        raise ValueError("Invalid verdict")
    if a.get("capability", "untested") not in ("untested", "gap_observed", "no_gap"):
        raise ValueError("Invalid capability status")
    routes = {r["id"] for r in p["routes"]}
    for key in ("elegant", "elementary"):
        if a.get(key, "") and a[key] not in routes:
            raise ValueError("Unknown selected route")
    for key in (
        "correct_elegant",
        "correct_elementary",
        "distinct",
        "methods_reviewed",
    ):
        if not isinstance(a.get(key, False), bool):
            raise ValueError("Invalid checkbox value")
    for key in ("answer", "notes"):
        if not isinstance(a.get(key, ""), str) or len(a.get(key, "")) > 100000:
            raise ValueError("Invalid text")
    for key in ("methods", "route_text"):
        if not isinstance(a.get(key, {}), dict) or set(a.get(key, {})) - routes:
            raise ValueError("Unknown route in corrections")
    lists = list(a.get("methods", {}).values()) + [a.get("additional_methods", [])]
    if any(
        not isinstance(tags, list)
        or any(not isinstance(t, str) or t not in method_ids for t in tags)
        for tags in lists
    ):
        raise ValueError("Unknown method label")
    if any(
        not isinstance(v, str) or len(v) > 200000
        for v in a.get("route_text", {}).values()
    ):
        raise ValueError("Invalid corrected solution")


def effective_methods(p: dict, a: dict) -> set[str]:
    tags = set(a.get("additional_methods", []))
    overrides = a.get("methods", {})
    for r in p["routes"]:
        tags.update(overrides.get(r["id"], r["methods"]))
    return tags


def assign_split(methods: set[str], splits: dict) -> str:
    values = {splits[m] for m in methods if m in splits}
    if "test" in values:
        return "test"
    if "validation" in values:
        return "validation"
    return "train" if methods else "unassigned"


def source_signature(p: dict) -> str:
    import hashlib

    value = [
        p["question"],
        p["answer"],
        [(r["id"], r["text"], r["methods"]) for r in p["routes"]],
    ]
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def export_blockers(p: dict, a: dict) -> list[str]:
    reasons = []
    if a.get("revision", 0) and a.get("source_signature") != source_signature(p):
        reasons.append("Source catalog changed: review and save again")
    if a.get("status") != "accepted":
        reasons.append("Not accepted")
    routes = {r["id"]: r for r in p["routes"]}
    chosen = [a.get("elegant"), a.get("elementary")]
    if any(r not in routes for r in chosen) or chosen[0] == chosen[1]:
        reasons.append("Choose two different routes")
    for key in (
        "correct_elegant",
        "correct_elementary",
        "distinct",
        "methods_reviewed",
    ):
        if not a.get(key):
            reasons.append("Missing review: " + key)
    if not (a.get("answer", "").strip() or p["answer"].strip()):
        reasons.append("Missing standalone answer")
    for r in p["routes"]:
        if not a.get("methods", {}).get(r["id"], r["methods"]):
            reasons.append("Unclassified route: " + r["title"])
    for rid in chosen:
        if (
            rid in routes
            and routes[rid]["kind"] == "summary"
            and not a.get("route_text", {}).get(rid, "").strip()
        ):
            reasons.append("Selected summary needs a complete derivation")
    if (
        any("Conflicting reference" in f for f in p["flags"])
        and not a.get("answer", "").strip()
    ):
        reasons.append("Resolve conflicting reference answers")
    return reasons


def make_handler(
    store: AuditStore, ui: Path, port: int
) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def send_json(self, value: object, status: int = 200) -> None:
            data = json.dumps(value, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def valid_host(self) -> bool:
            return self.headers.get("Host") in (
                f"127.0.0.1:{port}",
                f"localhost:{port}",
            )

        def do_GET(self):
            if not self.valid_host():
                self.send_json({"error": "Invalid host"}, 403)
                return
            path = urlparse(self.path).path
            try:
                if path == "/api/index":
                    self.send_json(store.index())
                    return
                if path.startswith("/api/problem/"):
                    pid = path.rsplit("/", 1)[-1]
                    p = store.problems[pid]
                    a = store.get_audit(pid)
                    review = store.machine_reviews().get(pid)
                    self.send_json(
                        {
                            "problem": p,
                            "audit": a,
                            "blockers": export_blockers(p, a),
                            "machine_review": review,
                            "elegance_review": store.elegance_review(pid, review),
                        }
                    )
                    return
                if path == "/api/machine-export":
                    self.send_json(store.machine_export())
                    return
                if path == "/api/elegance-export":
                    self.send_json(store.elegance_export())
                    return
                if path == "/api/export":
                    self.send_json(store.export())
                    return
                if path == "/api/audits":
                    self.send_json(
                        {
                            "catalog_sha256": store.catalog_sha256,
                            "settings": store.settings(),
                            "audits": store.saved_audits(),
                            "machine_reviews": store.machine_reviews(),
                            "elegance_reviews": {
                                pid: store.elegance_review(pid, review)
                                for pid, review in store.machine_reviews().items()
                                if review.get("status") == "accepted"
                            },
                        }
                    )
                    return
                base = store.root / "assets" if path.startswith("/assets/") else ui
                relative = (
                    path[len("/assets/") :]
                    if path.startswith("/assets/")
                    else (path.lstrip("/") or "index.html")
                )
                target = (base / relative).resolve()
                if not target.is_relative_to(base.resolve()) or not target.is_file():
                    self.send_error(404)
                    return
                data = target.read_bytes()
                self.send_response(200)
                self.send_header(
                    "Content-Type",
                    mimetypes.guess_type(target)[0] or "application/octet-stream",
                )
                self.send_header("Content-Length", str(len(data)))
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                self.wfile.write(data)
            except KeyError:
                self.send_json({"error": "Unknown problem"}, 404)
            except ValueError as exc:
                self.send_json({"error": str(exc)}, 409)

        def do_POST(self):
            if not self.valid_host():
                self.send_json({"error": "Invalid host"}, 403)
                return
            origin = self.headers.get("Origin")
            if origin and origin not in (
                f"http://127.0.0.1:{port}",
                f"http://localhost:{port}",
            ):
                self.send_json({"error": "Cross-origin write refused"}, 403)
                return
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                self.send_json({"error": "JSON required"}, 415)
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size < 2000000:
                    raise ValueError("Invalid body size")
                payload = json.loads(self.rfile.read(size))
                if not isinstance(payload, dict):
                    raise ValueError("Expected an object")
                path = urlparse(self.path).path
                if path == "/api/settings":
                    self.send_json(store.save_settings(payload))
                    return
                if path.startswith("/api/audit/"):
                    self.send_json(store.save_audit(path.rsplit("/", 1)[-1], payload))
                    return
                self.send_json({"error": "Unknown endpoint"}, 404)
            except Conflict as exc:
                self.send_json({"error": str(exc)}, 409)
            except (ValueError, TypeError) as exc:
                self.send_json({"error": str(exc)}, 400)
            except KeyError:
                self.send_json({"error": "Unknown problem"}, 404)

    return Handler


if __name__ == "__main__":
    main()
