#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>

#include "anybag.hpp"
#include "compute/pass.hpp"
#include "data/provider_base.hpp"
#include "feature_detect_v2/raco_lightglue.hpp"
#include "feature_detect_v2/shared_bindings.hpp"
#include "models.hpp"

namespace wslam::v2 {

// Joins the model's matches with its keypoints into correspondences between
// the reference and the current frame in LOD-0 pixels and undistorted rays.
//
// Input  (from AnyBag): OrtKeypointsReadyName (consume-once), OrtMatchesName,
//                       OrtMatchScoresName and OrtKeypointsName (taken),
//                       camera parameters
// Output (to   AnyBag): KeypointMatches under KeypointMatchesName
//
// Stops the stage on frames the model did not run on (the first one, and
// frames the frame prefilter skipped)
class LoadMatchesPass : public compute::Pass {
   public:
    struct Opts {
        // Fixed-point iterations and tolerance of the inverse distortion
        uint32_t undistort_iterations = 20;
        double undistort_tolerance = 1e-9;

        uint32_t camera_index = 0;
    };

    LoadMatchesPass(std::shared_ptr<SharedBindings> bindings, AnyBag& storage,
                    Opts opts);

    // Forwarded to the Opts-taking constructor so the defaults live in Opts
    LoadMatchesPass(std::shared_ptr<SharedBindings> bindings, AnyBag& storage);

    [[nodiscard]] std::optional<std::string> initialize() override;
    [[nodiscard]] std::optional<std::string> execute() override;
    [[nodiscard]] std::string getId() const override;

   private:
    std::shared_ptr<SharedBindings> bindings_;
    AnyBag& storage_;
    Opts opts_;

    // Set by initialize()
    std::optional<data::CamSensorParams> camera_;

    // Correspondences of the matches whose keypoints both lie in the image;
    // counts the others in `dropped`
    [[nodiscard]] KeypointMatches buildCorrespondences(
        std::span<const raco_lightglue::KeypointMatch> matches,
        std::span<const float> scores,
        std::span<const raco_lightglue::KeypointPosition> keypoints,
        const data::CamSensorParams& camera, size_t& dropped) const;
};

}  // namespace wslam::v2
