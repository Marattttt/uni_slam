#!/usr/bin/env bash

# Deletes the parts of the vendored ONNX Runtime subtree that a Linux WebGPU build
# never compiles -- roughly 500 MB of the 572 MB import. Every path below is
# globbed by ORT's CMake only behind an option build_onnxruntime.sh leaves off
# (onnxruntime_BUILD_UNIT_TESTS, onnxruntime_USE_CUDA, onnxruntime_USE_WINML,
# onnxruntime_ENABLE_TRAINING) or belongs to a non-C++ language binding.
#
# The deletions are kept in a commit of their own, separate from this script, so
# upgrading ORT stays a mechanical cycle:
#
#   git revert <the prune commit>
#   git subtree pull --prefix vendor/onnxruntime <url> <new tag> --squash
#   ./prune_onnxruntime.sh && git commit
#
# Re-running this script is harmless; rm -rf on an already-pruned tree is a no-op.

set -euo pipefail

ORT_DIR="$PWD/vendor/onnxruntime"

if [ ! -d "$ORT_DIR" ]; then
	echo "No vendored ONNX Runtime at $ORT_DIR" >&2
	exit 1
fi

PRUNE_PATHS=(
	# Test suites and their model data: ~250 MB, the single largest share.
	onnxruntime/test
	# Python bindings and their tooling, needs onnxruntime_ENABLE_PYTHON.
	onnxruntime/python
	# CUDA contributed operator kernels: ~115 MB, needs onnxruntime_USE_CUDA.
	onnxruntime/contrib_ops/cuda
	# Windows ML, including its test collateral: ~100 MB, needs Windows.
	winml
	# Training runtime, needs onnxruntime_ENABLE_TRAINING.
	orttraining
	# Language bindings other than C/C++.
	js
	csharp
	java
	objectivec
	rust
	# Documentation.
	docs
)

for path in "${PRUNE_PATHS[@]}"; do
	target="$ORT_DIR/$path"

	if [ -e "$target" ]; then
		echo "removing $path"
		rm -rf "$target"
	else
		echo "already gone $path"
	fi
done

echo "======================================================"
echo "ONNX Runtime subtree pruned"
echo "======================================================"
du -sh "$ORT_DIR"
