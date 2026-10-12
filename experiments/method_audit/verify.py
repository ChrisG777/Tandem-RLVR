# /// script
# requires-python = ">=3.11"
# dependencies = ["sympy==1.14.0", "antlr4-python3-runtime==4.11.1"]
# ///
"""Execute proposed answer checks separately from model judgments (macOS sandbox)."""

from __future__ import annotations

import argparse
import ast
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
import resource
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
ALLOWED_MODULES = {
    "math",
    "itertools",
    "fractions",
    "decimal",
    "collections",
    "functools",
    "operator",
    "statistics",
    "json",
    "sympy",
}
FORBIDDEN_CALLS = {
    "open",
    "exec",
    "eval",
    "compile",
    "__import__",
    "getattr",
    "setattr",
    "delattr",
    "globals",
    "locals",
    "vars",
    "input",
    "breakpoint",
    "help",
    "exit",
    "quit",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "outputs/method-audit")
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument(
        "--retry",
        action="store_true",
        help="Re-run unchanged checks, including previous failures.",
    )
    args = parser.parse_args()
    print(json.dumps(verify_reviews(args.data, args.timeout, args.retry), indent=2))


def verify_reviews(
    data: Path, timeout: int = 30, retry: bool = False
) -> dict[str, int]:
    """Run new accepted-review checks; save sidecars, never alter reviews or human audits.

    Checker execution is limited to stdlib/SymPy arithmetic, 30 seconds by default, and
    a macOS sandbox denying network and writes. Unsupported code is withheld.
    Passing corroborates an answer; it does not prove route validity/findability.
    """
    try:
        from .review import source_fingerprint
    except ImportError:
        from review import source_fingerprint

    problems = {
        p["id"]: p for p in json.loads((data / "catalog.json").read_text())["problems"]
    }
    destination = data / "machine-review" / "verification"
    destination.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    verifier_digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for path in sorted((data / "machine-review" / "reviews").glob("*.json")):
        raw = path.read_bytes()
        review = json.loads(raw)
        if review.get("status") != "accepted":
            continue
        digest = hashlib.sha256(raw).hexdigest()
        output_path = destination / path.name
        if output_path.exists() and not retry:
            prior = json.loads(output_path.read_text())
            if (
                prior.get("review_sha256") == digest
                and prior.get("verifier_sha256") == verifier_digest
            ):
                counts["cached"] = counts.get("cached", 0) + 1
                continue
        problem = problems.get(review.get("problem_id"))
        check = review.get("verification", {})
        code = check.get("checker_code", "")
        result = dict(
            problem_id=review.get("problem_id"),
            review_sha256=digest,
            verifier_sha256=verifier_digest,
            checker_sha256=hashlib.sha256(code.encode()).hexdigest(),
            source_fingerprint=review.get("source_fingerprint"),
            status="unsupported",
            computed_answer="",
            expected_answer=problem["answer"] if problem else "",
            answer_matches_source=False,
            output="",
            error="",
        )
        if problem is None or review.get("source_fingerprint") != source_fingerprint(
            problem
        ):
            result["error"] = "Missing problem or stale source fingerprint."
        else:
            result.update(execute_checker(code, problem["answer"], timeout))
        atomic_json(output_path, result)
        counts[result["status"]] = counts.get(result["status"], 0) + 1
    return counts


def execute_checker(code: str, source_answer: str, timeout: int = 30) -> dict:
    """Execute an arithmetic checker; compare computed answer to source externally.

    Output must end with one JSON object containing computed_answer and passed.
    No model-provided success flag can override a mismatching source answer.
    """
    result = dict(
        status="unsupported",
        computed_answer="",
        expected_answer=source_answer,
        answer_matches_source=False,
        output="",
        error="",
    )
    try:
        validate_checker(code)
        if not Path("/usr/bin/sandbox-exec").exists():
            raise ValueError("macOS sandbox-exec unavailable; checker was not run.")
        with tempfile.TemporaryDirectory(prefix="method-check-") as directory:
            completed = subprocess.run(
                [
                    "/usr/bin/sandbox-exec",
                    "-p",
                    checker_sandbox_profile(),
                    sys.executable,
                    "-I",
                    "-B",
                    "-c",
                    code,
                ],
                cwd=directory,
                capture_output=True,
                text=True,
                timeout=timeout,
                env={"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "0"},
                preexec_fn=lambda: set_limits(timeout),
            )
        result["output"] = completed.stdout[-16000:]
        if completed.returncode:
            result.update(
                status="error",
                error=completed.stderr[-4000:] or f"Exit {completed.returncode}",
            )
            return result
        value = json.loads(completed.stdout.strip().splitlines()[-1])
        computed = str(value["computed_answer"])
        match = answers_equal(computed, source_answer)
        result.update(
            computed_answer=computed,
            answer_matches_source=match,
            status="passed" if match and value.get("passed") is True else "failed",
        )
        if not match:
            result["error"] = (
                "Computed answer does not match the source under conservative exact normalization."
            )
    except subprocess.TimeoutExpired:
        result.update(status="error", error=f"Exceeded {timeout} seconds.")
    except (ValueError, SyntaxError, KeyError, IndexError, TypeError, OSError) as exc:
        result["error"] = str(exc)
    return result


def answers_equal(computed: str, reference: str) -> bool:
    """Conservative exact scalar/text equality; no floating-point tolerance."""
    left, right = normalize_answer(computed), normalize_answer(reference)
    try:
        return Fraction(left) == Fraction(right)
    except (ValueError, ZeroDivisionError):
        if left == right and bool(left):
            return True
        # A maintained LaTeX parser handles exact algebraic numeric answers.
        # Never use eval-based parse_expr/sympify on source text or accept tolerance.
        try:
            from sympy import Symbol, pi, simplify
            from sympy.parsing.latex import parse_latex

            if max(len(left), len(right)) > 250:
                return False
            lhs = parse_latex(left, strict=True)
            rhs = parse_latex(right, strict=True)
            # The LaTeX parser treats the conventional numeric constant as a
            # symbol. Resolve only explicit \pi, never arbitrary source names.
            if r"\pi" in left:
                lhs = lhs.subs(Symbol("pi"), pi)
            if r"\pi" in right:
                rhs = rhs.subs(Symbol("pi"), pi)
            if lhs.free_symbols or rhs.free_symbols:
                return False
            return simplify(lhs - rhs) == 0
        except Exception:
            return False


def normalize_answer(value: str) -> str:
    value = value.strip().replace("$", "").replace("\\left", "").replace("\\right", "")
    value = re.sub(
        r"\\(?:[,;:! ]|qquad\b|quad\b|enspace\b|thinspace\b)", "", value
    ).strip()
    value = re.sub(r"\\(?:boxed|text)\{([^{}]*)\}", r"\1", value)
    # Only flatten a standalone fraction: removing braces inside an expression
    # can change which factors belong to the denominator.
    fraction = re.fullmatch(
        r"\\(?:d?frac)\s*\{\s*(-?\d+)\s*\}\s*\{\s*(\d+)\s*\}", value
    )
    if fraction:
        value = f"{fraction[1]}/{fraction[2]}"
    value = re.sub(r"\\sqrt\s*([A-Za-z0-9])", r"\\sqrt{\1}", value)
    value = value.replace("{-}", "-").replace("{+}", "+")
    value = re.sub(r"\s+", "", value)
    if re.fullmatch(r"-?\d{1,3}(?:,\d{3})+", value):
        value = value.replace(",", "")
    return value


def validate_checker(code: str) -> None:
    """Reject non-arithmetic capabilities before sandbox execution; fail closed."""
    if not code.strip():
        raise ValueError("No independent checker supplied.")
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name not in ALLOWED_MODULES for alias in node.names):
                raise ValueError("Checker imports unsupported modules.")
        if isinstance(node, ast.ImportFrom):
            if (
                node.level
                or node.module not in ALLOWED_MODULES
                or any(a.name.startswith("_") or a.name == "*" for a in node.names)
            ):
                raise ValueError("Checker imports unsupported names.")
        if isinstance(node, ast.Name) and (
            node.id in FORBIDDEN_CALLS or node.id.startswith("__")
        ):
            raise ValueError("Checker uses unsupported runtime access.")
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise ValueError("Checker uses private runtime access.")
        if isinstance(node, (ast.ClassDef, ast.AsyncFunctionDef, ast.Await)):
            raise ValueError("Checker uses unsupported runtime constructs.")


def checker_sandbox_profile() -> str:
    """Deny network/writes and home-file reads outside this Python installation."""
    home = json.dumps(str(Path.home()))
    prefixes = {str(Path(sys.prefix).resolve()), str(Path(sys.base_prefix).resolve())}
    prefixes.update(
        str(Path(p).resolve()) for p in sys.path if p.endswith("site-packages")
    )
    exceptions = "".join(f"(require-not (subpath {json.dumps(p)}))" for p in prefixes)
    return (
        "(version 1)(allow default)(deny network*)(deny file-write*)"
        f"(deny file-read-data (require-all (subpath {home}){exceptions}))"
    )


def set_limits(timeout: int) -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (timeout, timeout + 1))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


if __name__ == "__main__":
    main()
