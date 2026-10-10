"""Binary APPS reward; all candidate execution happens in a fresh Linux sandbox."""
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import shlex
import subprocess
import tempfile

DIAGNOSTICS = ("format_error", "syntax_error", "runtime_error", "wrong_answer", "timeout")


def compute_score(data_source: str, solution_str: str, ground_truth: str,
                  extra_info: dict | None = None) -> dict:
    """Return binary score/acc and failure indicators; raise on grader failures.

    APPS_GRADER_RUNTIME must contain python/, env/, testing_util.py and worker.py.
    APPS_BWRAP optionally selects the Bubblewrap executable. Never run in-process.
    """
    if data_source != "apps":
        raise ValueError(f"Unsupported coding source: {data_source}")
    tests = validate_tests(json.loads(ground_truth))
    code = extract_code(solution_str)
    if code is None:
        return result("format_error")
    return run_sandbox({"code": code, "tests": tests}, timeout=5 * (len(tests["inputs"]) + 1) + 10)


def extract_code(text: str) -> str | None:
    """Return the last complete explicitly Python-labeled fenced block, unchanged."""
    blocks = re.findall(r"(?m)^```python[ \t]*\r?\n(.*?)^```[ \t]*$", text, re.DOTALL)
    return blocks[-1] if blocks else None


def normalize_tests(raw: str | dict) -> dict:
    """Convert original APPS JSON into pinned verl checker's string-based format."""
    obj = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(obj, dict) or not isinstance(obj.get("inputs"), list) or not isinstance(obj.get("outputs"), list):
        raise ValueError("Expected input/output lists")
    if not obj["inputs"] or len(obj["inputs"]) != len(obj["outputs"]):
        raise ValueError("Empty or unequal test lists")
    if obj.get("fn_name"):
        if not isinstance(obj["fn_name"], str) or not obj["fn_name"].isidentifier():
            raise ValueError("Invalid function name")
        if any(not isinstance(args, list) for args in obj["inputs"]):
            raise ValueError("Call-based inputs must be argument lists")
        inputs = ["\n".join(json.dumps(arg) for arg in args) for args in obj["inputs"]]
        outputs = [json.dumps(v) for v in obj["outputs"]]
    else:
        def lines(value):
            if isinstance(value, str):
                return value
            if isinstance(value, list) and all(isinstance(v, str) for v in value):
                return "\n".join(value)
            raise ValueError("Standard input/output must be text")
        inputs, outputs = [lines(v) for v in obj["inputs"]], [lines(v) for v in obj["outputs"]]
    return validate_tests({"inputs": inputs, "outputs": outputs, "fn_name": obj.get("fn_name"), "format": "verl-apps-v1"})


def validate_tests(tests: dict) -> dict:
    """Reject malformed normalized test specifications, including vacuous success."""
    if not isinstance(tests, dict) or tests.get("format") != "verl-apps-v1":
        raise ValueError("Missing normalized APPS format")
    inputs, outputs = tests.get("inputs"), tests.get("outputs")
    if not isinstance(inputs, list) or not inputs or not isinstance(outputs, list) or len(inputs) != len(outputs):
        raise ValueError("Empty or unequal test lists")
    if any(not isinstance(x, str) for x in inputs + outputs):
        raise ValueError("Checker inputs and outputs must be strings")
    if tests.get("fn_name") and not str(tests["fn_name"]).isidentifier():
        raise ValueError("Invalid function name")
    return tests


def result(status: str) -> dict:
    if status != "pass" and status not in DIAGNOSTICS:
        raise RuntimeError(f"Invalid grader status: {status}")
    return {"score": float(status == "pass"), "acc": float(status == "pass"),
            **{name: float(status == name) for name in DIAGNOSTICS}}


def grader_identity() -> dict:
    """Fingerprint code, checker and locked runtime before accepting resumed grades."""
    runtime = Path(os.environ["APPS_GRADER_RUNTIME"]).resolve()
    files = [Path(__file__), runtime / "worker.py", runtime / "testing_util.py", runtime / "runtime.json"]
    return {"sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
            "test_seconds": 5, "memory_bytes": 2 * 1024**3, "output_bytes": 1024**2}


def run_sandbox(request: dict, *, timeout: int) -> dict:
    """Execute a bounded checker process with only its runtime and input mounted."""
    import resource
    runtime = Path(os.environ["APPS_GRADER_RUNTIME"]).resolve()
    grader_identity()
    def limits():
        resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
        resource.setrlimit(resource.RLIMIT_FSIZE, (1024**2, 1024**2))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    with tempfile.TemporaryDirectory(prefix="apps-input-") as tmp:
        payload = Path(tmp) / "request.json"
        payload.write_text(json.dumps(request))
        command = shlex.split(os.environ.get("APPS_BWRAP", "bwrap")) + ["--unshare-all", "--die-with-parent",
            "--new-session", "--cap-drop", "ALL", "--clearenv", "--ro-bind", str(runtime), str(runtime),
            "--ro-bind", "/usr", "/usr", "--symlink", "usr/lib", "/lib",
            "--symlink", "usr/lib64", "/lib64", "--proc", "/proc", "--dev", "/dev",
            "--tmpfs", "/tmp", "--ro-bind", str(payload), "/request.json",
            "--setenv", "PYTHONHOME", str(runtime / "python"),
            "--setenv", "PYTHONPATH", str(runtime / "env/lib/python3.11/site-packages"),
            "--setenv", "OPENBLAS_NUM_THREADS", "1", "--setenv", "OMP_NUM_THREADS", "1",
            "--chdir", "/tmp", str(runtime / "env/bin/python"), str(runtime / "worker.py"), "/request.json"]
        with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
            proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                    start_new_session=True, preexec_fn=limits)
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                stdout.seek(0)
                if stdout.readline().strip() != b'{"ready": true}':
                    raise RuntimeError("Grader startup timed out")
                return result("timeout")
            stdout.seek(0)
            stderr.seek(0)
            raw, error = stdout.read(1024**2), stderr.read(4096).decode(errors="replace")
            if proc.returncode:
                raise RuntimeError(f"Sandbox/checker exit {proc.returncode}: {error}")
            try:
                lines = raw.splitlines()
                if json.loads(lines[0]) != {"ready": True}:
                    raise ValueError("Missing sandbox readiness marker")
                response = json.loads(lines[-1])
            except (ValueError, UnicodeDecodeError) as exc:
                raise RuntimeError(f"Malformed sandbox response: {error}") from exc
            if "infrastructure_error" in response:
                raise RuntimeError(response["infrastructure_error"])
            return response if request.get("preflight") else result(response["status"])
