#!/usr/bin/env bash
# Install NVIDIA's pinned CUDA compiler/headers into an isolated shared directory.
# Usage: bash env/install-qwen35-cuda.sh DESTINATION; no driver or Python changes.
set -euo pipefail
destination=${1:?Supply an isolated CUDA toolkit destination}
if [ -f "$destination/.cuda-12.8.1" ]; then
    "$destination/bin/nvcc" --version
    exit 0
fi
test ! -e "$destination" || { echo "Refusing to overwrite $destination" >&2; exit 1; }
mkdir -p "$(dirname "$destination")"
staging=$(mktemp -d "${destination}.partial.XXXXXX")
trap 'rm -rf -- "$staging"' EXIT
while read -r component version digest; do
    archive="$staging/$component.tar.xz"
    curl -fsSL --retry 3 "https://developer.download.nvidia.com/compute/cuda/redist/$component/linux-x86_64/$component-linux-x86_64-$version-archive.tar.xz" -o "$archive"
    echo "$digest  $archive" | sha256sum --check --status
    tar -xJf "$archive" --strip-components=1 -C "$staging"
    rm "$archive"
done <<'PACKAGES'
cuda_nvcc 12.8.93 9961b3484b6b71314063709a4f9529654f96782ad39e72bf1e00f070db8210d3
cuda_cudart 12.8.90 8d566b5fe745c46842dc16945cf36686227536decd2302c372be86da37faca68
cuda_cccl 12.8.90 0740e9e01e4f15e17c5ab8d68bba4f8ec0eb6b84edccba4ac45112d2d2174e4b
PACKAGES
if [ ! -e "$staging/lib64" ]; then ln -s lib "$staging/lib64"; fi
"$staging/bin/nvcc" --version
touch "$staging/.cuda-12.8.1"
mv "$staging" "$destination"
