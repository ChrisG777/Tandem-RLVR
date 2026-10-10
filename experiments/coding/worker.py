"""Sandbox-only worker for the pinned verl APPS checker; standard JSON protocol."""
import contextlib
import io
import json
from pathlib import Path
import sys


def main() -> None:
    request = json.loads(Path(sys.argv[1]).read_text())
    if request.get("preflight"):
        import socket
        assert not Path("/dev/nvidia0").exists()
        assert not Path("/home").exists()
        try:
            Path("/request.json").write_text("must fail")
        except OSError:
            pass
        else:
            raise RuntimeError("Task input is writable")
        try:
            socket.create_connection(("1.1.1.1", 443), timeout=1)
        except OSError:
            pass
        else:
            raise RuntimeError("Sandbox network is reachable")
        print(json.dumps({"isolated": True}))
        return
    from testing_util import run_test
    import resource
    resource.setrlimit(resource.RLIMIT_NPROC, (1, 1))
    code = request["code"]
    try:
        compile(code, "candidate.py", "exec")
    except SyntaxError:
        print(json.dumps({"status": "syntax_error"}))
        return
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            results, metadata = run_test(request["tests"], test=code, timeout=5)
        if len(results) == len(request["tests"]["inputs"]) and all(v == True for v in results):
            status = "pass"
        elif "timeout" in str(metadata).lower() or "timed out" in str(metadata).lower():
            status = "timeout"
        elif any(v in (-1, -2) for v in results):
            status = "runtime_error"
        else:
            status = "wrong_answer"
        print(json.dumps({"status": status}))
    except BaseException as exc:
        print(json.dumps({"infrastructure_error": f"{type(exc).__name__}: {exc}"}))


if __name__ == "__main__":
    main()
