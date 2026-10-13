#!/usr/bin/env bash

# Build type as the first argument (Release, RelWithDebInfo, Debug). Each one
# gets its own build and install directory; the project picks one with
# -DWSLAM_ORT_BUILD_TYPE.
BUILD_TYPE="${1:-RelWithDebInfo}"

ORT_DIR="$PWD/vendor/onnxruntime"
BUILD_DIR="$ORT_DIR/build/$BUILD_TYPE"
INSTALL_DIR="$ORT_DIR/install/$BUILD_TYPE"

# SwiftShader, Dawn's CPU (fallback) adapter, which the --device=cpu run uses.
# Dawn's fetch_dawn_dependencies.py does not fetch it, so it is checked out here
# at the commit Dawn's DEPS pins. It lives next to the per-type build directories
# so a rebuild does not have to fetch it again.
SWIFTSHADER_DIR="$ORT_DIR/build/swiftshader"
SWIFTSHADER_URL="https://swiftshader.googlesource.com/SwiftShader"
SWIFTSHADER_COMMIT="a7c547b55474c3d8bde53711eae24ae0e28bbc0a"

rm -r "$BUILD_DIR"
rm -r "$INSTALL_DIR"

set -euo pipefail

if [ "$(git -C "$SWIFTSHADER_DIR" rev-parse HEAD 2>/dev/null)" != "$SWIFTSHADER_COMMIT" ]; then
	rm -rf "$SWIFTSHADER_DIR"
	git init -q "$SWIFTSHADER_DIR"
	git -C "$SWIFTSHADER_DIR" remote add origin "$SWIFTSHADER_URL"
	# A shallow fetch of the one commit, or the whole history if the server
	# refuses to serve a commit by its hash
	if ! git -C "$SWIFTSHADER_DIR" fetch --depth 1 origin "$SWIFTSHADER_COMMIT"; then
		git -C "$SWIFTSHADER_DIR" fetch origin
	fi
	git -C "$SWIFTSHADER_DIR" checkout -q "$SWIFTSHADER_COMMIT"
fi

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
#
# DAWN_ENABLE_SWIFTSHADER builds SwiftShader's Vulkan driver alongside Dawn and
# lets Dawn's Vulkan backend discover it as its fallback adapter.
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
	-D DAWN_WERROR=OFF \
	-D DAWN_ENABLE_SWIFTSHADER=ON \
	-D DAWN_SWIFTSHADER_DIR="$SWIFTSHADER_DIR"

# The SwiftShader checkout has to be the one this Dawn was tested against
DAWN_DEPS="$BUILD_DIR/_deps/dawn-src/DEPS"
if ! grep -q "SwiftShader@$SWIFTSHADER_COMMIT" "$DAWN_DEPS"; then
	echo "Dawn pins a different SwiftShader commit than $SWIFTSHADER_COMMIT:" >&2
	grep "SwiftShader@" "$DAWN_DEPS" >&2
	echo "Update SWIFTSHADER_COMMIT." >&2
	exit 1
fi

cmake --build "$BUILD_DIR" --parallel $(nproc)
# Dawn loads the driver at run time, so nothing in the default build depends on it
cmake --build "$BUILD_DIR" --parallel $(nproc) --target vk_swiftshader

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

# Dawn looks for the SwiftShader driver in the directory of libwebgpu_dawn.so
# (Dawn's runtime search paths), and Dawn's install rules don't install it
SWIFTSHADER_LIB="$BUILD_DIR/Linux/libvk_swiftshader.so"
if [ ! -f "$SWIFTSHADER_LIB" ]; then
	echo "No SwiftShader driver found at $SWIFTSHADER_LIB" >&2
	exit 1
fi
install -m 755 "$SWIFTSHADER_LIB" "$INSTALL_DIR/lib/"

echo "======================================================"
echo "ONNX Runtime and Dawn build and install successful"
echo "======================================================"
