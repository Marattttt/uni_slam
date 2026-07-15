#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>

#include "anybag.hpp"

namespace wslam {

// Options for ExportMap. `map_path` is the only required field; the JSON
// metadata sidecar path is derived from it by replacing the extension with
// .json (so `map.ply` produces `map.json` beside it).
struct ExportOpts {
    // Where to write the landmark point cloud. Must end in `.ply` — the
    // metadata path is derived from this by replacing the extension.
    std::filesystem::path map_path;

    // Camera whose intrinsics get embedded in the metadata file. The pipeline
    // currently runs on a single camera so this defaults to 0.
    uint32_t camera_index = 0;
};

// Reads the final MapSnapshot and camera parameters from the shared storage
// and writes the map out as two files:
//   - <map_path>          : ASCII PLY point cloud, one vertex per landmark
//                           with x,y,z and an integer `landmark_id` property
//   - <map_path stem>.json: keyframes, intrinsics, run stats, schema metadata
//
// The two files are designed to be loaded together by the `presentor/` Python
// viewer. The PLY format was chosen because it is universally supported by
// point-cloud tooling (Open3D, MeshLab, CloudCompare, Blender, rerun).
//
// Returns std::nullopt on success or an error message on failure (missing
// snapshot, missing camera params, IO error, malformed map_path).
[[nodiscard]] std::optional<std::string> ExportMap(const AnyBag& storage,
                                                   const ExportOpts& opts);

}  // namespace wslam
