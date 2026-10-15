#include "merge.hpp"

#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>

#include <limits>
#include <string>
#include <utility>

using namespace wslam;
using namespace wslam::v2::mapping::loop_closure;
namespace loop_closure = wslam::v2::mapping::loop_closure;

MapSnapshot loop_closure::ApplyCorrections(const MapSnapshot& vio,
                                           const PoseGraph& graph,
                                           const LandmarkAnchor& anchor_of,
                                           MergeStats* stats) {
    MergeStats local;
    MapSnapshot corrected = vio;

    for (KeyframePose& keyframe : corrected.keyframes) {
        const gtsam::Pose3 pose{gtsam::Rot3{keyframe.R_world_cam},
                                gtsam::Point3{keyframe.t_world_cam}};
        const gtsam::Pose3 moved
            = graph.smoothCorrectionFor(keyframe.id.v) * pose;
        keyframe.R_world_cam = moved.rotation().matrix();
        keyframe.t_world_cam = moved.translation();
        ++local.keyframes_corrected;
    }

    // Past every keyframe the graph got: the newest correction
    constexpr uint64_t kNewest = std::numeric_limits<uint64_t>::max();
    for (LandmarkEstimate& landmark : corrected.landmarks) {
        const std::optional<uint64_t> anchor = anchor_of(landmark.id.v);
        if (anchor) {
            ++local.landmarks_corrected;
        } else {
            ++local.landmarks_unanchored;
        }
        const gtsam::Pose3 correction
            = graph.smoothCorrectionFor(anchor.value_or(kNewest));
        landmark.position_world
            = correction.transformFrom(gtsam::Point3{landmark.position_world});
    }

    if (stats != nullptr) {
        *stats = local;
    }
    return corrected;
}

LoopClosureReport loop_closure::MakeReport(const LoopClosureWorker& worker) {
    const LoopClosureStats& stats = worker.stats();

    LoopClosureReport report{
        .model = std::string{ToString(worker.modelKind())},
        .keyframes = stats.keyframes,
        .nodes = stats.nodes,
        .queries = stats.queries,
        .queries_skipped = stats.queries_skipped,
        .candidates = stats.candidates,
        .rejections = {
            {"similarity", stats.rejected_similarity},
            {"time_gap", stats.rejected_time_gap},
            {"matches", stats.rejected_matches},
            {"correspondences", stats.rejected_correspondences},
            {"pnp", stats.rejected_pnp},
            {"consistency", stats.rejected_consistency},
        },
        .loops = {},
        .mean_embed_ms = stats.embedded > 0
                             ? stats.embed_ms_total
                                   / static_cast<double>(stats.embedded)
                             : 0.0,
        .mean_verify_ms = stats.verifications > 0
                              ? stats.verify_ms_total
                                    / static_cast<double>(stats.verifications)
                              : 0.0,
    };

    for (const AcceptedLoop& loop : worker.loops()) {
        report.loops.push_back(LoopClosureEdge{
            .query = wslam::PoseId{loop.edge.query},
            .candidate = wslam::PoseId{loop.edge.candidate},
            .similarity = loop.similarity,
            .matches = loop.matches,
            .correspondences = loop.correspondences,
            .inliers = loop.inliers,
            .correction_translation_m = loop.correction_translation_m,
            .correction_rotation_rad = loop.correction_rotation_rad,
        });
    }

    return report;
}
