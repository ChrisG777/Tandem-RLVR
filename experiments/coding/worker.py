"""Sandbox-only worker for the pinned verl APPS checker; standard JSON protocol."""
import contextlib
import io
import json
from pathlib import Path
import sys
import signal


def main() -> None:
    sys.set_int_max_str_digits(0)
    request = json.loads(Path(sys.argv[1]).read_text())
    print(json.dumps({"ready": True}), flush=True)
    if request.get("preflight"):
        import socket
        assert not sys.stdin.isatty()
        assert sys.stdin.read() == ""
        assert not Path("/dev/nvidia0").exists()
        assert not Path("/home").exists()
        for target in ("/request.json", "/tmp/probe", "/dev/probe", "/probe"):
            try:
                Path(target).write_text("must fail")
            except OSError:
                pass
            else:
                raise RuntimeError(f"Sandbox path is writable: {target}")
        try:
            socket.create_connection(("1.1.1.1", 443), timeout=1)
        except OSError:
            pass
        else:
            raise RuntimeError("Sandbox network is reachable")
        print(json.dumps({"isolated": True}))
        return
    import testing_util
    testing_util.StringIO = BoundedIO
    signal.signal(signal.SIGALRM, test_timeout)
    import resource
    resource.setrlimit(resource.RLIMIT_NPROC, (1, 1))
    code = request["code"]
    try:
        compile(code, "candidate.py", "exec")
    except SyntaxError:
        print(json.dumps({"status": "syntax_error"}))
        return
    try:
        # faulthandler inside the upstream checker needs stderr's real descriptor;
        # the parent already bounds that file with RLIMIT_FSIZE.
        with contextlib.redirect_stdout(BoundedIO()):
            results, metadata = testing_util.run_test(request["tests"], test=code, timeout=5)
        if len(results) == len(request["tests"]["inputs"]) and all(v == True for v in results):
            status = "pass"
        elif "timeout" in str(metadata).lower() or "timed out" in str(metadata).lower():
            status = "timeout"
        elif any(v in (-1, -2) for v in results):
            status = "runtime_error"
        else:
            status = "wrong_answer"
        print(json.dumps({"status": status}))
    except (SystemExit, KeyboardInterrupt, MemoryError):
        print(json.dumps({"status": "runtime_error"}))
    except BaseException as exc:
        print(json.dumps({"infrastructure_error": f"{type(exc).__name__}: {exc}"}))


class BoundedIO(io.StringIO):
    """Bound candidate/log output before the checker buffers it in host memory."""
    def write(self, text):
        if self.tell() + len(text) > 1024**2:
            raise RuntimeError("Output limit exceeded")
        return super().write(text)


def test_timeout(signum, frame):
    """Match the outer verl wrapper's alarm setup when calling run_test directly."""
    raise TimeoutError("APPS test timeout")


if __name__ == "__main__":
    main()
