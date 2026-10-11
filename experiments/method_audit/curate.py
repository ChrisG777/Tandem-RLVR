# /// script
# requires-python = ">=3.11"
# dependencies = ["pyarrow==19.0.1"]
# ///
"""Build a provenance-preserving pool of alternate-method candidates; no inference."""

from __future__ import annotations
import argparse
import csv
from dataclasses import asdict, dataclass, field
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "outputs/method-audit"
SOURCES = {
    "harp.zip": (
        "HARP",
        "dac2734ff6443bcaf3bbdcb10f13cf21ae9729c2",
        "https://raw.githubusercontent.com/aadityasingh/HARP/{rev}/HARP.jsonl.zip",
    ),
    "proofrank.parquet": (
        "ProofRank",
        "9a6ec576e2de7929d9b72cc7964eccf6577adb35",
        "https://huggingface.co/datasets/INSAIT-Institute/ProofRank/resolve/{rev}/data/main-00000-of-00001.parquet",
    ),
    "hm.jsonl": (
        "HM-ReasoningBench",
        "8da44ed052ae6bcdafe5bb6e1c930484e358a571",
        "https://huggingface.co/datasets/jasonlwd/HM-ReasoningBench/resolve/{rev}/data/train.jsonl",
    ),
    "braingle-math.csv": (
        "Braingle Math",
        "aa242baa9bae20c8841e1cc8eacd442c4e281d31",
        "https://raw.githubusercontent.com/stephenxia1/brainteasers/{rev}/data/braingle/braingle_math_annotated.csv",
    ),
    "braingle-logic.csv": (
        "Braingle Logic",
        "aa242baa9bae20c8841e1cc8eacd442c4e281d31",
        "https://raw.githubusercontent.com/stephenxia1/brainteasers/{rev}/data/braingle/braingle_Logic_annotated.csv",
    ),
}


@dataclass
class SolutionRoute:
    """Original source text; method matches are proposals, never correctness evidence."""

    id: str
    title: str
    text: str
    kind: str
    source: str
    methods: list[str] = field(default_factory=list)
    evidence: dict[str, str] = field(default_factory=dict)
    verification: str = "Not independently verified"


@dataclass
class Problem:
    """One normalized question with all retained routes and source references."""

    id: str
    title: str
    question: str
    answer: str
    subject: str
    sources: list[dict]
    routes: list[dict] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    curator_note: str = ""
    suggested_pair: list[str] = field(default_factory=list)
    priority: int = 0
    shortlist: bool = False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    result = build_catalog(args.out)
    print(json.dumps(result["summary"], indent=2))


def build_catalog(out: Path) -> dict:
    """Fetch pinned public releases and export candidates + hashes into out.

    Exact normalized-question duplicates are merged; original solutions stay intact.
    Rebuilds preserve audits.sqlite3. No model calls, training, or correctness claims.
    """
    import pyarrow.parquet as pq

    sources = out / "sources"
    sources.mkdir(parents=True, exist_ok=True)
    methods = load_methods()
    manifest = []
    for name, (label, revision, template) in SOURCES.items():
        url = template.format(rev=revision)
        path = sources / name
        if not path.exists():
            download(url, path)
        manifest.append(
            {
                "name": label,
                "file": name,
                "revision": revision,
                "url": url,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    catalog: dict[str, Problem] = {}
    with zipfile.ZipFile(sources / "harp.zip") as archive:
        for line in archive.read("HARP.jsonl").splitlines():
            row = json.loads(line)
            title = f"{row['year']} {row['contest']} · {row['number']}"
            refs = {
                "dataset": "HARP",
                "key": title,
                "url": "https://github.com/aadityasingh/HARP",
                "source_file": "harp.zip",
            }
            routes = [
                (f"Human solution {i}", row[f"solution_{i}"], "human")
                for i in range(1, row["num_solutions"] + 1)
            ]
            add_problem(
                catalog,
                title,
                row["problem"],
                row["answer"],
                row["subject"],
                refs,
                routes,
                methods,
            )
    for line in (sources / "hm.jsonl").read_text().splitlines():
        row = json.loads(line)
        # HARP-derived entries join only on exact normalized question, not guessed IDs.
        refs = {
            "dataset": "HM-ReasoningBench",
            "key": row["problem_id"],
            "url": "https://huggingface.co/datasets/jasonlwd/HM-ReasoningBench",
            "source_file": "hm.jsonl",
        }
        routes = [
            ("Human reference", row["human_solution"], "human"),
            ("GPT-5.1 solution", row["model_solution"], "model"),
        ]
        add_problem(
            catalog,
            row["problem_id"],
            row["problem"],
            "",
            str(row["subject"]),
            refs,
            routes,
            methods,
        )
    for row in pq.read_table(sources / "proofrank.parquet").to_pylist():
        refs = {
            "dataset": "ProofRank",
            "key": row["problem_id"],
            "url": "https://huggingface.co/datasets/INSAIT-Institute/ProofRank",
            "source_file": "proofrank.parquet",
        }
        routes = [
            (f"Method summary {i + 1}", s, "summary")
            for i, s in enumerate(row["human_solution_summaries"])
        ]
        add_problem(
            catalog,
            row["problem_id"],
            row["problem"],
            row["gold_answer"],
            "Competition mathematics",
            refs,
            routes,
            methods,
        )
    for name in ["braingle-math.csv", "braingle-logic.csv"]:
        for row in csv.DictReader(io.StringIO((sources / name).read_text())):
            if not row.get("Question") or not row.get("Answer"):
                continue
            refs = {
                "dataset": "Braingle",
                "key": f"{name}:{row['ID']}",
                "url": "https://github.com/stephenxia1/brainteasers",
                "source_file": name,
            }
            p = add_problem(
                catalog,
                row["Title"],
                row["Question"],
                "",
                row.get("Category", ""),
                refs,
                [("Human reference", row["Answer"], "human")],
                methods,
            )
            p.curator_note = f"Source human-brute-force annotation: {row.get('Human Bruteforce', 'not supplied')}. A second route must be supplied and verified."
    items = []
    for p in catalog.values():
        # Single-route HARP records add little to this audit; retain other sources as leads.
        if len(p.routes) < 2 and all(s["dataset"] == "HARP" for s in p.sources):
            continue
        classify_candidate(p)
        p.shortlist = bool(
            p.answer
            and sum(r["kind"] == "human" for r in p.routes) >= 2
            and p.priority >= 55
            and not any("Diagram" in f or "choices" in f for f in p.flags)
        )
        items.append(asdict(p))
    items.sort(key=lambda p: (-p["priority"], p["id"]))
    coverage = {
        m["id"]: sum(any(m["id"] in r["methods"] for r in p["routes"]) for p in items)
        for m in methods
    }
    summary = {
        "problems": len(items),
        "with_multiple_routes": sum(len(p["routes"]) >= 2 for p in items),
        "with_reference_answer": sum(bool(p["answer"]) for p in items),
        "screened_shortlist": sum(p["shortlist"] for p in items),
        "method_families": len(methods),
        "families_with_candidates": sum(v > 0 for v in coverage.values()),
        "independently_verified_pairs": 0,
        "coverage": coverage,
    }
    result = {
        "schema_version": 1,
        "summary": summary,
        "methods": methods,
        "sources": manifest,
        "notes": [
            "Lexical method proposals require manual review. They do not prove distinctness, elegance, or correctness.",
            "Family holdouts are provisional until every route’s method labels are audited. Unknown methods block export.",
            "Human references and model outputs are not mechanically certified. No senior/junior calibration has run.",
            "Exact normalized-question deduplication only; near duplicates still require review.",
        ],
        "problems": items,
    }
    write_json(out / "catalog.json", result)
    install_math_assets(out / "assets/katex")
    return result


def load_methods() -> list[dict]:
    rows = json.loads(Path(__file__).with_name("methods.json").read_text())
    return [
        dict(zip(("id", "label", "domain", "split", "pattern"), row)) for row in rows
    ]


def normalized(text: str) -> str:
    return (
        re.sub(r"\s+", "", text).casefold().replace("\\left", "").replace("\\right", "")
    )


def propose_methods(text: str, methods: list[dict]) -> tuple[list[str], dict[str, str]]:
    evidence = {}
    for m in methods:
        match = re.search(m["pattern"], text, re.I)
        if match:
            evidence[m["id"]] = text[
                max(0, match.start() - 65) : min(len(text), match.end() + 95)
            ].strip()
    return list(evidence), evidence


def add_problem(
    catalog: dict[str, Problem],
    title: str,
    question: str,
    answer: str,
    subject: str,
    source: dict,
    routes: list[tuple[str, str, str]],
    methods: list[dict],
) -> Problem:
    key = hashlib.sha256(normalized(question).encode()).hexdigest()[:20]
    p = catalog.setdefault(key, Problem(key, title, question, answer, subject, []))
    p.sources.append(source)
    if not p.answer and answer:
        p.answer = answer
    elif answer and p.answer and normalized(answer) != normalized(p.answer):
        p.flags.append("Conflicting reference answers: review required")
    texts = {normalized(r["text"]) for r in p.routes}
    for label, text, kind in routes:
        if not text or normalized(text) in texts:
            continue
        tags, evidence = propose_methods(text, methods)
        rid = hashlib.sha256(normalized(text).encode()).hexdigest()[:16]
        route = SolutionRoute(rid, label, text, kind, source["dataset"], tags, evidence)
        p.routes.append(asdict(route))
        texts.add(normalized(text))
    return p


def classify_candidate(p: Problem) -> None:
    if not p.answer:
        p.flags.append("Standalone answer not supplied")
    if re.search(r"which of the following|(?:is|are|then):\s*$", p.question, re.I):
        p.flags.append("May require omitted answer choices")
    if "[asy]" in p.question or "\\includegraphics" in p.question:
        p.flags.append("Diagram requires source inspection")
    if re.search(r"\b(?:prove|show that)\b", p.question, re.I) and not p.answer:
        p.flags.append("Proof task: no exact-answer reward yet")
    if any(r["kind"] == "summary" for r in p.routes):
        p.flags.append("Summaries are not complete derivations")
    if len(p.routes) < 2:
        p.flags.append("Second route needed")
    if not all(r["methods"] for r in p.routes):
        p.flags.append("Some routes have no proposed method")
    best = (-1, [])
    elementary = {
        "casework",
        "enumeration",
        "direct_calculation",
        "quadratic_formula",
        "coordinates",
        "dynamic_programming",
    }
    for i, a in enumerate(p.routes):
        for b in p.routes[i + 1 :]:
            ma, mb = set(a["methods"]), set(b["methods"])
            # Ranking only: labels may be false positives, and do not establish route distinctions.
            score = (
                2 * bool(ma - mb) + 2 * bool(mb - ma) + 2 * bool((ma ^ mb) & elementary)
            )
            score += int(a["kind"] == "human" and b["kind"] == "human")
            score -= 3 * int(a["kind"] == "summary" or b["kind"] == "summary")
            if score > best[0]:
                best = (score, [a["id"], b["id"]])
    p.suggested_pair = best[1]
    p.priority = (
        max(0, best[0]) * 10 + 5 * bool(p.answer) - 10 * bool("[asy]" in p.question)
    )
    if not p.curator_note:
        p.curator_note = "Candidate pair ranked by differing method mentions; inspect both arguments. Left/right ordering does not imply elegance."
    if p.title == "1984 AIME · 11":
        p.priority = 120
        p.suggested_pair = [p.routes[0]["id"], p.routes[1]["id"]]
        p.curator_note = "Research seed: gap selection versus explicit distribution casework. Both yield 56 valid birch-position sets out of 792. Junior accessibility remains untested."
    if p.title == "1983 AIME · 7":
        p.priority = 115
        p.suggested_pair = [p.routes[0]["id"], p.routes[3]["id"]]
        p.curator_note = "Research seed: complementary placement argument versus conditioning on the second knight’s distance. Both obtain 11/46. Audit the sample-space reasoning and junior accessibility."
    if p.title == "2015 AIME_I · 3":
        p.priority = 110
        p.curator_note = "Cautionary seed: one brute-force route uses the AIME answer-range convention, while several other routes omit justification. Do not approve the pair on answer agreement alone."


def install_math_assets(path: Path) -> None:
    if (path / "katex.min.js").exists() and (
        path / "contrib/auto-render.min.js"
    ).exists():
        return
    url = "https://registry.npmjs.org/katex/-/katex-0.16.22.tgz"
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read()
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        for entry in archive.getmembers():
            if not entry.isfile():
                continue
            if entry.name.startswith("package/dist/"):
                relative = entry.name[len("package/dist/") :]
            elif entry.name == "package/LICENSE":
                relative = "LICENSE"
            else:
                continue
            if ".." in Path(relative).parts:
                raise ValueError("Unsafe archive path")
            if not (
                relative.endswith((".js", ".css", ".woff2")) or relative == "LICENSE"
            ):
                continue
            target = path / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.extractfile(entry).read())


def download(url: str, path: Path) -> None:
    temp = path.with_suffix(path.suffix + ".partial")
    try:
        with (
            urllib.request.urlopen(url, timeout=120) as source,
            temp.open("wb") as target,
        ):
            import shutil

            shutil.copyfileobj(source, target)
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":")))
    temp.replace(path)


if __name__ == "__main__":
    main()
