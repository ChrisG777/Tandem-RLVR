# Triton compiles cuda_utils.c on the first vLLM import. That file includes Python.h,
# which ships in python3.11-dev, not in the python3.11 runtime package.
ensure_python_dev() {
  local py=$1
  if "$py" -c 'import os, sysconfig; raise SystemExit(0 if os.path.isfile(os.path.join(sysconfig.get_paths()["include"], "Python.h")) else 1)'; then
    return 0
  fi
  echo "installing python3.11-dev (Triton needs Python.h)"
  if [[ "$(id -u)" -eq 0 ]]; then
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y python3.11-dev
  else
    sudo apt-get update
    sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y python3.11-dev
  fi
  if ! "$py" -c 'import os, sysconfig; raise SystemExit(0 if os.path.isfile(os.path.join(sysconfig.get_paths()["include"], "Python.h")) else 1)'; then
    echo "python3.11-dev installed, but Python.h is still missing for $py" >&2
    return 1
  fi
}
