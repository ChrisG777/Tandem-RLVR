"""Build a self-contained, pinned CPU checker runtime; never alter training packages."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import re
import subprocess
import sys
import tomllib
import urllib.request


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    args = parser.parse_args()
    root = args.runtime.resolve()
    root.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[2]
    revision = (repo / "third_party/VERL_BASE_COMMIT.txt").read_text().strip()
    url = f"https://raw.githubusercontent.com/verl-project/verl/{revision}/verl/utils/reward_score/prime_code/testing_util.py"
    original = urllib.request.urlopen(url).read()
    code = original.decode()
    old = 'from pyext import RuntimeModule'
    replacement = '''from types import ModuleType

def _load_module(source):
    module = ModuleType("tmp_sol")
    exec(compile(source, "<string>", "exec"), module.__dict__)
    return module
'''
    if code.count(old) != 1 or code.count('RuntimeModule.from_string("tmp_sol", "", sol)') != 2:
        raise ValueError("Pinned checker loader changed")
    code = code.replace(old, replacement).replace('RuntimeModule.from_string("tmp_sol", "", sol)', '_load_module(sol)')
    (root / "testing_util.py").write_text(code)
    shutil.copyfile(Path(__file__).with_name("worker.py"), root / "worker.py")
    bwrap = shutil.which("bwrap")
    if bwrap is None:
        raise RuntimeError("Bubblewrap must be installed on the setup host")
    shutil.copyfile(bwrap, root / "bwrap")
    (root / "bwrap").chmod(0o755)
    libs = root / "bwrap-libs"
    libs.mkdir(exist_ok=True)
    linked = subprocess.check_output(["ldd", bwrap], text=True)
    for library in re.findall(r"(/[^\s()]+)", linked):
        shutil.copyfile(library, libs / Path(library).name)
        (libs / Path(library).name).chmod(0o755)
    # Copy uv-managed CPython so no user's home/cache must be mounted in a sandbox.
    base = Path(sys.base_prefix)
    if not (root / "python").exists():
        shutil.copytree(base, root / "python", symlinks=True)
    if not (root / "env").exists():
        subprocess.run(["uv", "venv", "--python", str(root / "python/bin/python3.11"), str(root / "env")], check=True)
    lock_path = Path(__file__).parent / "env/uv.lock"
    numpy = next(p for p in tomllib.loads(lock_path.read_text())["package"] if p["name"] == "numpy")
    if numpy["version"] != "2.2.6":
        raise ValueError("Grading numpy pin differs from the CPU tools lock")
    artifacts = [numpy["sdist"], *numpy["wheels"]]
    requirement = "numpy==2.2.6 " + " ".join("--hash=" + a["hash"] for a in artifacts) + "\n"
    (root / "requirements.lock").write_text(requirement)
    subprocess.run(["uv", "pip", "sync", "--require-hashes", "--python", str(root / "env/bin/python"),
                    str(root / "requirements.lock")], check=True)
    record = {"verl_revision": revision, "upstream_sha256": hashlib.sha256(original).hexdigest(),
              "checker_sha256": hashlib.sha256(code.encode()).hexdigest(),
              "patch": "Replace pyext dynamic loader with types.ModuleType and compile/exec",
              "python": sys.version, "numpy": "2.2.6",
              "bwrap_sha256": hashlib.sha256((root / "bwrap").read_bytes()).hexdigest(),
              "tools_lock_sha256": hashlib.sha256(lock_path.read_bytes()).hexdigest()}
    (root / "runtime.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
