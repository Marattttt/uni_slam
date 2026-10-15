#pragma once

#include <gtsam/geometry/Pose3.h>
#include <webgpu/webgpu_cpp.h>

#include <Eigen/Core>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <expected>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

#include "compute/onnx_interop.hpp"
#include "compute/performance.hpp"
#include "config.hpp"
#include "data/provider_base.hpp"
#include "model_io.hpp"
#include "pair_matcher.hpp"
#include "place_database.hpp"
#include "place_recognition.hpp"
#include "pose_graph.hpp"

namespace wslam::v2::mapping::loop_closure {

// A keyframe tracked a landmark at this keypoint (observed LOD-0 pixel)
struct KeyframeLandmark {
    uint64_t landmark = 0;
    Eigen::Vector2d pixel;
};

// Sent when a keyframe is accepted: its image and the landmarks it tracks
struct KeyframeImageEvent {
    uint64_t keyframe = 0;
    uint64_t timestamp_ns = 0;
    PlaceImage image;
    std::vector<KeyframeLandmark> landmarks;
};

// Sent after a smoother drain: the poses it settled and what left the window
struct MapUpdateEvent {
    // Keyframes drained since the last update, in id order, with their
    // estimate at the drain
    std::vector<std::pair<uint64_t, gtsam::Pose3>> drained_poses;
    // Keyframes marginalised out of the window, with their last estimate
    std::vector<std::pair<uint64_t, gtsam::Pose3>> archived_poses;
    // Landmarks marginalised with a valid triangulation, VIO world frame
    std::vector<std::pair<uint64_t, Eigen::Vector3d>> archived_landmarks;
    std::optional<Eigen::Vector3d> gravity_world;
};

// What the main loop applies to its newest poses: T_corrected = T_drift *
// T_vio. Yaw about gravity and translation only
struct DriftCorrection {
    // Incremented on every change; 0 means no correction yet
    uint64_t version = 0;
    // Node it was computed at
    uint64_t anchor_keyframe = 0;
    gtsam::Pose3 T_drift;
    size_t loops = 0;
};

// An accepted loop, with what verification measured
struct AcceptedLoop {
    LoopEdge edge;
    float similarity = 0.0F;
    size_t matches = 0;
    size_t correspondences = 0;
    size_t inliers = 0;
    // Disagreement between the PnP relative pose and the graph's estimate
    // before the loop: the drift it corrects
    double correction_translation_m = 0.0;
    double correction_rotation_rad = 0.0;
};

struct LoopClosureStats {
    size_t keyframes = 0;
    size_t embedded = 0;
    size_t nodes = 0;
    size_t queries = 0;
    // Queries never run because a newer node arrived in the same batch
    size_t queries_skipped = 0;
    size_t candidates = 0;
    size_t rejected_similarity = 0;
    size_t rejected_time_gap = 0;
    size_t rejected_matches = 0;
    size_t rejected_correspondences = 0;
    size_t rejected_pnp = 0;
    size_t rejected_consistency = 0;
    size_t accepted = 0;
    double embed_ms_total = 0.0;
    double verify_ms_total = 0.0;
    size_t verifications = 0;
};

// Loop closure on a thread of its own. The main loop pushes keyframe images
// and map updates; the thread embeds every keyframe, gates it into the pose
// graph and the place database, and queries the newest node of each batch:
// FAISS retrieval, RaCo-LightGlue matching against each candidate, PnP on
// the candidate's landmarks. An accepted loop goes into the pose graph and
// updates the drift correction the main loop polls.
//
// push() and latestCorrection() are the only calls the main loop makes while
// the thread runs; finish() stops it, after which the graph and the results
// can be read. The thread records its performance under the thread id
// "loop_closure", into the given recorder (which must outlive the worker) or
// into one of its own
class LoopClosureWorker {
   public:
    struct Models {
        // Device the WebGPU sessions run on. First, so it is destroyed after
        // them
        std::optional<wgpu::Device> device;
        std::unique_ptr<PlaceRecognitionModel> place_recognition;
        std::unique_ptr<PairMatcher> matcher;
    };

    LoopClosureWorker(LoopClosureConfig config, data::CamSensorParams camera,
                      Models models, compute::PerfRecorder& perf);
    LoopClosureWorker(LoopClosureConfig config, data::CamSensorParams camera,
                      Models models);
    // Drops the backlog and joins
    ~LoopClosureWorker();

    LoopClosureWorker(const LoopClosureWorker&) = delete;
    LoopClosureWorker& operator=(const LoopClosureWorker&) = delete;
    LoopClosureWorker(LoopClosureWorker&&) = delete;
    LoopClosureWorker& operator=(LoopClosureWorker&&) = delete;

    void push(KeyframeImageEvent event);
    void push(MapUpdateEvent event);

    // The newest correction, if there is one newer than since_version
    [[nodiscard]] std::optional<DriftCorrection> latestCorrection(
        uint64_t since_version) const;

    // Processes the queued events (or drops them, per the config), then
    // stops the thread. Returns the first error the thread hit. Idempotent
    [[nodiscard]] std::optional<std::string> finish();

    // Valid after finish()
    [[nodiscard]] const PoseGraph& graph() const { return graph_; }
    [[nodiscard]] const LoopClosureStats& stats() const { return stats_; }
    [[nodiscard]] const std::vector<AcceptedLoop>& loops() const {
        return accepted_;
    }
    [[nodiscard]] VprModelKind modelKind() const;

   private:
    using Event = std::variant<KeyframeImageEvent, MapUpdateEvent>;

    struct KeyframeRecord {
        uint64_t timestamp_ns = 0;
        // Dropped once the keyframe is attached to a node: only nodes are
        // matched against
        PlaceImage image;
        std::vector<KeyframeLandmark> landmarks;
        std::optional<PlaceDescriptor> descriptor;
        // The smoother's last estimate, once the keyframe left the window
        std::optional<gtsam::Pose3> archived_pose;
    };

    const LoopClosureConfig config_;
    const data::CamSensorParams camera_;
    Models models_;
    // Recorded into when the worker is not given a recorder. Before perf_,
    // which may refer to it
    compute::PerfRecorder own_perf_;
    compute::PerfRecorder& perf_;

    // Thread-owned state
    PlaceDatabase database_;
    PoseGraph graph_;
    std::map<uint64_t, KeyframeRecord> keyframes_;
    std::map<uint64_t, Eigen::Vector3d> landmarks_;
    LoopClosureStats stats_;
    std::vector<AcceptedLoop> accepted_;
    bool failed_ = false;

    mutable std::mutex mutex_;
    std::condition_variable condition_;
    std::deque<Event> queue_;
    bool stop_ = false;
    bool drop_backlog_ = false;
    std::optional<std::string> error_;
    DriftCorrection correction_;

    std::thread thread_;

    void run();
    void process(std::deque<Event>& batch);
    void handle(KeyframeImageEvent& event);
    // Returns the newest keyframe of the update that became a node
    std::optional<uint64_t> handle(MapUpdateEvent& event);
    void query(uint64_t keyframe);
    [[nodiscard]] std::expected<AcceptedLoop, std::string> verify(
        uint64_t candidate, uint64_t query, float similarity);
    void publishCorrection();
    void fail(std::string error);
};

// Loads the models the config and the compute device select and starts the
// worker. Sessions on the WebGPU EP get an auxiliary device of their own
[[nodiscard]] std::expected<std::shared_ptr<LoopClosureWorker>, std::string>
CreateLoopClosureWorker(compute::OrtProvider& ort,
                        const LoopClosureConfig& config,
                        const data::CamSensorParams& camera,
                        compute::PerfRecorder& perf);

}  // namespace wslam::v2::mapping::loop_closure
