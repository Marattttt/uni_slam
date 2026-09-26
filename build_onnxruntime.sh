#!/usr/bin/env bash

BUILD_TYPE=Release

ORT_DIR="$PWD/vendor/onnxruntime"
BUILD_DIR="$ORT_DIR/build/$BUILD_TYPE"
INSTALL_DIR="$ORT_DIR/install/$BUILD_TYPE"

rm -r "$BUILD_DIR"
rm -r "$INSTALL_DIR"

set -euo pipefail

# ONNX Runtime keeps its CMakeLists.txt in cmake/, not at the repository root.
#
# onnxruntime_BUILD_DAWN_SHARED_LIBRARY is the load-bearing flag: it makes ORT
# force DAWN_BUILD_MONOLITHIC_LIBRARY=SHARED and DAWN_ENABLE_INSTALL=ON, and that
# is the only ORT configuration which yields an installable Dawn package. The app
# links the very libwebgpu_dawn.so produced here, so exactly one Dawn ends up in
# the process. As a consequence the WebGPU execution provider itself is linked
# statically into libonnxruntime.so -- onnxruntime_USE_EP_API_ADAPTERS (the
# provider as its own plugin library) is a hard error alongside a shared Dawn, and
# would bury a second, private Dawn inside that plugin.
#
# onnxruntime_BUILD_SHARED_LIB is required rather than preferred: a static ORT
# built with WebGPU support emits no CMake package files at all.
#
# --compile-no-warning-as-error is needed because ORT sets COMPILE_WARNING_AS_ERROR
# on every one of its own targets, and GCC 16 is newer than anything ORT's CI
# covers. DAWN_WERROR=OFF does the same for the Dawn subproject.
#
# FETCHCONTENT_TRY_FIND_PACKAGE_MODE=NEVER stops ORT's dependency declarations from
# short-circuiting to system packages. Without it, find_package(re2) picks up Arch's
# re2, whose config pulls in the system abseil on top of the abseil ORT has already
# built from cmake/deps.txt, and the duplicated absl:: targets abort the configure.
cmake \
	-G Ninja \
	-B "$BUILD_DIR" \
	-S "$ORT_DIR/cmake" \
	--compile-no-warning-as-error \
	-D CMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
	-D CMAKE_BUILD_TYPE="$BUILD_TYPE" \
	-D CMAKE_POLICY_VERSION_MINIMUM=3.5 \
	-D FETCHCONTENT_TRY_FIND_PACKAGE_MODE=NEVER \
	-D onnxruntime_USE_WEBGPU=ON \
	-D onnxruntime_BUILD_SHARED_LIB=ON \
	-D onnxruntime_BUILD_DAWN_SHARED_LIBRARY=ON \
	-D onnxruntime_BUILD_UNIT_TESTS=OFF \
	-D DAWN_WERROR=OFF

cmake --build "$BUILD_DIR" --parallel $(nproc)

cmake --install "$BUILD_DIR"

# ORT declares Dawn with EXCLUDE_FROM_ALL, and CMake ignores the install rules of
# an EXCLUDE_FROM_ALL subdirectory when installing the parent. Dawn's headers, its
# libwebgpu_dawn.so and its CMake package therefore need Dawn's own install rules
# invoked directly, into the same prefix, so that one prefix carries both packages.
DAWN_BUILD_DIR="$BUILD_DIR/_deps/dawn-build"

if [ ! -f "$DAWN_BUILD_DIR/cmake_install.cmake" ]; then
	echo "No Dawn build found at $DAWN_BUILD_DIR" >&2
	echo "Locate it under $BUILD_DIR and update DAWN_BUILD_DIR." >&2
	exit 1
fi

cmake --install "$DAWN_BUILD_DIR" --prefix "$INSTALL_DIR"

echo "======================================================"
echo "ONNX Runtime and Dawn build and install successful"
echo "======================================================"
