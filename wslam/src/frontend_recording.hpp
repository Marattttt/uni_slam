#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "anybag.hpp"
#include "data/provider_base.hpp"
#include "models.hpp"

// Recording of the v2 front-end output, for running the mapping stage without
// the GPU and the model (a model run costs seconds per frame).
//
// The model matches the reference frame (the last keyframe) with the current
// one, so what the front end produces depends on the keyframes the mapping
// stage accepted: a recording is a golden run. It stores, per frame, the
// model's matches, the IMU samples that arrived with the frame and whether the
// mapping stage made the frame the next reference. A replay re-runs the CPU
// RANSAC and triangulation passes and the mapping stage on it, and is only
// valid while the mapping stage makes the same keyframe decisions; it stops at
// the first divergence.
namespace wslam::v2 {

struct RecordedFrame {
    // Position among the frames the pipeline processed, from 0
    uint64_t frame_index = 0;
    uint64_t timestamp_ns = 0;
    // Samples the sensor loader published with this frame
    std::vector<data::IMUReading> imu;
    // The model's correspondences, empty when the model didn't run (first
    // frame)
    std::optional<KeypointMatches> matches;
    // Frame the matches were made against
    uint64_t reference_frame_index = 0;
    // The mapping stage made this frame the reference of the next frames
    bool reference_advanced = false;
};

struct FrontendRecording {
    static constexpr int kFormatVersion = 1;

    // Provenance
    std::string source;
    std::string model_path;
    int64_t keypoint_count = 0;

    data::CamSensorParams camera;
    data::IMUSensorParams imu_params;

    std::vector<RecordedFrame> frames;

    // Final map of the recording run, for checking a replay against
    std::vector<KeyframePose> golden_keyframes;
    size_t golden_landmark_count = 0;
};

// MessagePack file
[[nodiscard]] std::optional<std::string> SaveFrontendRecording(
    const FrontendRecording& recording, const std::filesystem::path& path);
[[nodiscard]] std::expected<FrontendRecording, std::string>
LoadFrontendRecording(const std::filesystem::path& path);

// Builds a FrontendRecording from a live v2 pipeline with mapping
class FrontendRecorder {
   public:
    FrontendRecorder(AnyBag& storage, std::filesystem::path output,
                     FrontendRecording header);

    // Between the pose estimate and the mapping stage: the frame's timestamp,
    // new IMU samples (before the mapping stage consumes them) and the model's
    // matches (taken)
    [[nodiscard]] std::optional<std::string> captureFrame();

    // After the mapping stage: whether it made this frame the reference
    [[nodiscard]] std::optional<std::string> captureDecision();

    // After the mapping stage was flushed: stores the final map and writes
    // the file
    [[nodiscard]] std::optional<std::string> save();

   private:
    AnyBag& storage_;
    const std::filesystem::path output_;
    FrontendRecording recording_;

    uint64_t last_imu_timestamp_ = 0;
    uint64_t reference_frame_index_ = 0;
};

// Feeds a FrontendRecording to the passes downstream of the model
class FrontendReplay {
   public:
    struct Divergence {
        uint64_t frame_index = 0;
        bool recorded_advance = false;
        bool replayed_advance = false;
    };

    FrontendReplay(AnyBag& storage, FrontendRecording recording);

    // Publishes the sensor parameters, as the pipeline factory does. Call
    // before the stages are initialised
    void publishSensorParams();

    // First pass of the replayed pose estimate stage: publishes the next
    // frame's timestamp, IMU samples and matches. Stops the stage on a frame
    // without matches and the whole loop after the last frame
    [[nodiscard]] std::optional<std::string> publishFrame();

    // After the mapping stage: consumes the reference-advance marker, as the
    // model's preserve pass would, and compares it with the recording. Stops
    // the loop at a divergence
    [[nodiscard]] std::optional<std::string> checkDecision();

    [[nodiscard]] const FrontendRecording& recording() const {
        return recording_;
    }
    [[nodiscard]] size_t framesReplayed() const { return next_frame_; }
    [[nodiscard]] bool finished() const {
        return next_frame_ >= recording_.frames.size();
    }
    [[nodiscard]] const std::optional<Divergence>& divergence() const {
        return divergence_;
    }

   private:
    AnyBag& storage_;
    const FrontendRecording recording_;
    size_t next_frame_ = 0;
    std::optional<Divergence> divergence_;
};

}  // namespace wslam::v2
