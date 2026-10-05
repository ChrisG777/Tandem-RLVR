"""Check verl's Ray runtime-environment shape under the shared uv interpreter."""
import importlib.util
import os
from pathlib import Path
import sys


def main() -> None:
    """Start local CPU Ray, verify the worker interpreter/forks, then stop our instance."""
    import ray

    if os.environ.get("RAY_ENABLE_UV_RUN_RUNTIME_ENV") != "0":
        raise RuntimeError("Source slurm/training-env.sh before this check")
    ray.init(num_cpus=2, include_dashboard=False, object_store_memory=128 * 1024**2,
             runtime_env={"working_dir": None, "env_vars": {"NCCL_DEBUG": "WARN"}})
    try:
        actual = ray.get(ray.remote(worker_paths).remote(), timeout=120)
        expected = worker_paths()
        if actual != expected:
            raise RuntimeError(f"Worker environment differs: {actual} != {expected}")
        repo = Path(__file__).resolve().parents[1]
        for name in ("vllm", "verl"):
            if not Path(actual[name]).is_relative_to(repo / "third_party" / name):
                raise RuntimeError(f"Worker is not using the patched {name}: {actual}")
        print("Ray driver/worker startup passed:", actual, flush=True)
    finally:
        ray.shutdown()


def worker_paths() -> dict[str, str]:
    """Report the interpreter and module origins without loading GPU models."""
    return {"python": sys.executable,
            **{name: importlib.util.find_spec(name).origin for name in ("vllm", "verl")}}


if __name__ == "__main__":
    main()
