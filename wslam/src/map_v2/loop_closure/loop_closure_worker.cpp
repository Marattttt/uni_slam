#include "loop_closure_worker.hpp"

#include <spdlog/spdlog.h>

#include <chrono>
#include <exception>
#include <format>
#include <set>
#include <utility>

#include "assert.hpp"
#include "map/map_helpers_lods.hpp"
#include "pnp_verify.hpp"

using namespace wslam;
using namespace wslam::v2::mapping::loop_closure;
namespace loop_closure = wslam::v2::mapping::loop_closure;

#define LOG_ID "[Loop Closure]"

namespace {
// WebGPU EP context of the loop closure's auxiliary device; the pipeline's
// own device is OrtProvider::kWebGpuContextId
constexpr int kLoopClosureContextId
    = compute::OrtProvider::kWebGpuContextId + 1;

constexpr uint32_t kUndistortIterations = 20;
constexpr double kUndistortTolerance = 1e-9;

// Landmarks closer than this to the candidate camera are not used for PnP
constexpr double kMinLandmarkDepth = 0.05;

using Clock = std::chrono::steady_clock;

[[nodiscard]] double MillisecondsSince(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start)
        .count();
}

enum class Rejection : uint8_t { Matches, Correspondences, Pnp, Consistency };
}  // namespace

LoopClosureWorker::LoopClosureWorker(LoopClosureConfig config,
                                     data::CamSensorParams camera,
                                     Models models, compute::PerfRecorder& perf)
    : config_(std::move(config)),
      camera_(std::move(camera)),
      models_(std::move(models)),
      perf_(perf),
      database_(models_.place_recognition->spec().descriptor_dim),
      graph_(config_.noise, config_.node_gate) {
    WSLAM_ASSERT(models_.place_recognition && models_.matcher,
                 "loop closure needs both models");
    thread_ = std::thread([this] { run(); });
}

LoopClosureWorker::LoopClosureWorker(LoopClosureConfig config,
                                     data::CamSensorParams camera,
                                     Models models)
    : LoopClosureWorker(std::move(config), std::move(camera), std::move(models),
                        own_perf_) {}

LoopClosureWorker::~LoopClosureWorker() {
    {
        std::scoped_lock lock(mutex_);
        stop_ = true;
        drop_backlog_ = true;
    }
    condition_.notify_all();
    if (thread_.joinable()) {
        thread_.join();
    }
}

VprModelKind LoopClosureWorker::modelKind() const {
    return models_.place_recognition->spec().kind;
}

void LoopClosureWorker::push(KeyframeImageEvent event) {
    {
        std::scoped_lock lock(mutex_);
        WSLAM_ASSERT(!stop_, "pushing to a finished loop closure worker");
        queue_.emplace_back(std::move(event));
    }
    condition_.notify_one();
}

void LoopClosureWorker::push(MapUpdateEvent event) {
    {
        std::scoped_lock lock(mutex_);
        WSLAM_ASSERT(!stop_, "pushing to a finished loop closure worker");
        queue_.emplace_back(std::move(event));
    }
    condition_.notify_one();
}

std::optional<DriftCorrection> LoopClosureWorker::latestCorrection(
    uint64_t since_version) const {
    std::scoped_lock lock(mutex_);
    if (correction_.version <= since_version) {
        return std::nullopt;
    }
    return correction_;
}

std::optional<std::string> LoopClosureWorker::finish() {
    {
        std::scoped_lock lock(mutex_);
        if (!stop_) {
            stop_ = true;
            drop_backlog_ = !config_.flush_processes_backlog;
            spdlog::info(LOG_ID " finishing: {} events queued, {}",
                         queue_.size(),
                         drop_backlog_ ? "dropping them" : "processing them");
        }
    }
    condition_.notify_all();
    if (thread_.joinable()) {
        thread_.join();
    }

    std::scoped_lock lock(mutex_);
    return error_;
}

void LoopClosureWorker::run() {
    compute::PerfRecorder::setThreadId("loop_closure");

    while (true) {
        std::deque<Event> batch;
        {
            std::unique_lock lock(mutex_);
            condition_.wait(lock, [&] { return stop_ || !queue_.empty(); });
            if (stop_ && (drop_backlog_ || queue_.empty())) {
                return;
            }
            batch.swap(queue_);
        }

        if (failed_) {
            continue;
        }

        try {
            const auto perfscope = perf_.beginRecord("batch");
            process(batch);
        } catch (const std::exception& e) {
            fail(std::format("processing: {}", e.what()));
        }
    }
}

void LoopClosureWorker::fail(std::string error) {
    spdlog::error(LOG_ID " {}; loop closure stops", error);
    failed_ = true;
    std::scoped_lock lock(mutex_);
    if (!error_) {
        error_ = std::move(error);
    }
}

void LoopClosureWorker::process(std::deque<Event>& batch) {
    std::optional<uint64_t> newest_node;
    std::vector<uint64_t> settled;

    for (Event& event : batch) {
        if (failed_) {
            return;
        }

        if (auto* image = std::get_if<KeyframeImageEvent>(&event)) {
            const auto perfscope = perf_.beginRecord("embed");
            handle(*image);
            continue;
        }

        auto& update = std::get<MapUpdateEvent>(event);
        const auto perfscope = perf_.beginRecord("map update");
        if (const auto node = handle(update, settled)) {
            if (newest_node) {
                ++stats_.queries_skipped;
            }
            newest_node = node;
        }
    }

    if (failed_ || !newest_node) {
        dropImages(settled);
        return;
    }

    // Only the newest node is queried: the thread is behind otherwise, and
    // the older ones still entered the database
    {
        const auto perfscope = perf_.beginRecord("query");
        query(*newest_node);
    }
    // Candidates are matched by their features; a settled keyframe is never
    // a query again
    dropImages(settled);

    // A new node moves the newest correction even without a new loop
    if (graph_.loopCount() > 0) {
        const auto perfscope = perf_.beginRecord("publish correction");
        publishCorrection();
    }
}

void LoopClosureWorker::handle(KeyframeImageEvent& event) {
    ++stats_.keyframes;

    KeyframeRecord& record = keyframes_[event.keyframe];
    record.timestamp_ns = event.timestamp_ns;
    record.image = std::move(event.image);
    record.features = std::move(event.features);
    for (const KeyframeLandmark& tracked : event.landmarks) {
        record.landmarks.insert_or_assign(tracked.keypoint_index,
                                          tracked.landmark);
    }

    const auto start = Clock::now();
    auto descriptor = models_.place_recognition->describe(record.image);
    const double elapsed = MillisecondsSince(start);
    if (!descriptor) {
        fail(std::format("describing keyframe {}: {}", event.keyframe,
                         descriptor.error()));
        return;
    }

    record.descriptor = std::move(descriptor).value();
    ++stats_.embedded;
    stats_.embed_ms_total += elapsed;

    spdlog::debug(LOG_ID " embedded keyframe {} in {:.1f} ms ({} landmarks)",
                  event.keyframe, elapsed, record.landmarks.size());
}

void LoopClosureWorker::dropImages(std::span<const uint64_t> keyframes) {
    for (const uint64_t id : keyframes) {
        if (const auto it = keyframes_.find(id); it != keyframes_.end()) {
            it->second.image = {};
        }
    }
}

std::optional<uint64_t> LoopClosureWorker::handle(
    MapUpdateEvent& event, std::vector<uint64_t>& settled) {
    if (event.gravity_world) {
        graph_.setGravity(event.gravity_world.value());
    }

    for (auto& [id, position] : event.archived_landmarks) {
        landmarks_.insert_or_assign(id, position);
    }
    for (auto& [id, pose] : event.archived_poses) {
        if (const auto it = keyframes_.find(id); it != keyframes_.end()) {
            it->second.archived_pose = pose;
        }
    }

    std::optional<uint64_t> newest_node;
    for (auto& [id, pose] : event.drained_poses) {
        if (graph_.hasKeyframe(id)) {
            continue;
        }

        const bool is_node = graph_.addKeyframe(id, pose);
        auto record = keyframes_.find(id);
        settled.push_back(id);

        if (!is_node) {
            spdlog::debug(LOG_ID " keyframe {} attached to node {}", id,
                          graph_.lastNode().value_or(0));
            if (record != keyframes_.end()) {
                // Only nodes are ever matched against
                record->second.image = {};
                record->second.features.reset();
                record->second.landmarks.clear();
                record->second.descriptor.reset();
            }
            continue;
        }

        ++stats_.nodes;
        newest_node = id;
        if (record != keyframes_.end() && record->second.descriptor) {
            database_.add(id, record->second.descriptor.value());
        } else {
            spdlog::warn(LOG_ID " node {} has no descriptor; not a place", id);
        }
    }

    return newest_node;
}

void LoopClosureWorker::query(uint64_t keyframe) {
    const MatchAcceptance& acceptance = config_.acceptance;

    const auto record = keyframes_.find(keyframe);
    if (record == keyframes_.end() || !record->second.descriptor) {
        return;
    }
    if (keyframe <= acceptance.min_keyframe_gap) {
        return;
    }

    ++stats_.queries;
    const auto candidates = [&] {
        const auto perfscope = perf_.beginRecord("search");
        return database_.search(record->second.descriptor.value(),
                                acceptance.top_n,
                                keyframe - acceptance.min_keyframe_gap);
    }();
    stats_.candidates += candidates.size();

    for (const PlaceCandidate& candidate : candidates) {
        if (candidate.similarity < acceptance.min_similarity) {
            // Sorted, so the rest are below too
            ++stats_.rejected_similarity;
            spdlog::debug(LOG_ID " q{} -> c{}: similarity {:.3f} < {:.3f}",
                          keyframe, candidate.keyframe, candidate.similarity,
                          acceptance.min_similarity);
            break;
        }

        const uint64_t candidate_ts
            = keyframes_.at(candidate.keyframe).timestamp_ns;
        const double gap_s
            = static_cast<double>(record->second.timestamp_ns - candidate_ts)
              * 1e-9;
        if (gap_s < acceptance.min_time_gap_s) {
            ++stats_.rejected_time_gap;
            continue;
        }

        const auto start = Clock::now();
        auto verified = [&] {
            const auto perfscope = perf_.beginRecord("verify");
            return verify(candidate.keyframe, keyframe, candidate.similarity);
        }();
        stats_.verify_ms_total += MillisecondsSince(start);
        ++stats_.verifications;

        if (!verified) {
            spdlog::info(LOG_ID " q{} -> c{} (similarity {:.3f}) rejected: {}",
                         keyframe, candidate.keyframe, candidate.similarity,
                         verified.error());
            continue;
        }

        AcceptedLoop& loop
            = accepted_.emplace_back(std::move(verified).value());
        {
            const auto perfscope = perf_.beginRecord("add loop");
            graph_.addLoop(loop.edge);
        }
        ++stats_.accepted;

        spdlog::info(LOG_ID
                     " ACCEPTED loop q{} -> c{}: similarity {:.3f}, {} "
                     "matches, {} inliers of {}, corrects {:.3f} m / "
                     "{:.2f} deg",
                     keyframe, candidate.keyframe, loop.similarity,
                     loop.matches, loop.inliers, loop.correspondences,
                     loop.correction_translation_m,
                     loop.correction_rotation_rad * 180.0 / M_PI);

        {
            const auto perfscope = perf_.beginRecord("publish correction");
            publishCorrection();
        }
        // One loop per query
        return;
    }
}

std::expected<AcceptedLoop, std::string> LoopClosureWorker::verify(
    uint64_t candidate, uint64_t query, float similarity) {
    const MatchAcceptance& acceptance = config_.acceptance;
    const KeyframeRecord& candidate_record = keyframes_.at(candidate);
    const KeyframeRecord& query_record = keyframes_.at(query);

    const auto reject
        = [&](Rejection stage,
              std::string message) -> std::unexpected<std::string> {
        switch (stage) {
            case Rejection::Matches:
                ++stats_.rejected_matches;
                break;
            case Rejection::Correspondences:
                ++stats_.rejected_correspondences;
                break;
            case Rejection::Pnp:
                ++stats_.rejected_pnp;
                break;
            case Rejection::Consistency:
                ++stats_.rejected_consistency;
                break;
        }
        return std::unexpected(std::move(message));
    };

    auto pair = [&] {
        const auto perfscope = perf_.beginRecord("match");
        return models_.matcher->match(*candidate_record.features,
                                      query_record.image);
    }();
    if (!pair) {
        // A model failure, not a rejection
        throw std::runtime_error("matching: " + pair.error());
    }

    std::vector<const PairMatch*> confident;
    for (const PairMatch& match : pair->matches) {
        if (match.score >= acceptance.matcher_min_score) {
            confident.push_back(&match);
        }
    }
    if (confident.size() < acceptance.min_matches) {
        return reject(
            Rejection::Matches,
            std::format("{} matches above score {:.2f} < {}", confident.size(),
                        acceptance.matcher_min_score, acceptance.min_matches));
    }

    // The candidate's landmarks in its own camera frame. Its archived pose is
    // the estimate the landmarks were triangulated against; the drained one
    // stands in while it is still in the window
    const gtsam::Pose3 T_world_candidate
        = candidate_record.archived_pose.value_or(graph_.vioPose(candidate));

    std::vector<PnpCorrespondence> correspondences;
    // Landmark of each correspondence
    std::vector<uint64_t> correspondence_landmarks;
    std::set<uint64_t> used_landmarks;

    for (const PairMatch* match : confident) {
        // The candidate's keypoints are the ones the front end tracked its
        // landmarks with, under the same indices
        const auto tracked = candidate_record.landmarks.find(match->left_index);
        if (tracked == candidate_record.landmarks.end()
            || used_landmarks.contains(tracked->second)) {
            continue;
        }
        const uint64_t landmark = tracked->second;

        const auto position = landmarks_.find(landmark);
        if (position == landmarks_.end()) {
            continue;
        }

        const Eigen::Vector3d point_candidate
            = T_world_candidate.transformTo(gtsam::Point3{position->second});
        if (point_candidate.z() < kMinLandmarkDepth) {
            continue;
        }

        used_landmarks.insert(landmark);
        correspondence_landmarks.push_back(landmark);
        correspondences.push_back(PnpCorrespondence{
            .point_candidate = point_candidate,
            .query_ray = map::util::PixelToNormalised(
                match->right_px, camera_, kUndistortIterations,
                kUndistortTolerance),
        });
    }

    if (correspondences.size() < acceptance.min_correspondences) {
        return reject(
            Rejection::Correspondences,
            std::format("{} of {} matches reach an archived landmark < {}",
                        correspondences.size(), confident.size(),
                        acceptance.min_correspondences));
    }

    auto pnp = [&] {
        const auto perfscope = perf_.beginRecord("pnp");
        return SolvePnp(correspondences, camera_.intrinsics[0], acceptance);
    }();
    if (!pnp) {
        return reject(Rejection::Pnp, std::move(pnp).error());
    }

    // The graph's current estimate of the same relative pose: the loop may
    // correct it by the drift since the candidate, not by more
    const gtsam::Pose3 estimated
        = graph_.correctedPose(candidate).between(graph_.correctedPose(query));
    const gtsam::Pose3 discrepancy = estimated.between(pnp->T_candidate_query);
    const double translation = discrepancy.translation().norm();
    const double rotation = discrepancy.rotation().axisAngle().second;
    if (translation > acceptance.max_correction_translation_m
        || rotation > acceptance.max_correction_rotation_rad) {
        return reject(
            Rejection::Consistency,
            std::format("PnP ({} inliers) disagrees with the estimate by "
                        "{:.3f} m / {:.2f} deg",
                        pnp->inliers, translation, rotation * 180.0 / M_PI));
    }

    std::vector<LoopObservation> observations;
    observations.reserve(pnp->inlier_indices.size());
    for (const size_t index : pnp->inlier_indices) {
        observations.push_back(LoopObservation{
            .landmark = correspondence_landmarks[index],
            .query_ray = correspondences[index].query_ray,
        });
    }

    return AcceptedLoop{
        .edge = {.candidate = candidate,
                 .query = query,
                 .T_candidate_query = pnp->T_candidate_query},
        .observations = std::move(observations),
        .similarity = similarity,
        .matches = confident.size(),
        .correspondences = pnp->correspondences,
        .inliers = pnp->inliers,
        .correction_translation_m = translation,
        .correction_rotation_rad = rotation,
    };
}

void LoopClosureWorker::publishCorrection() {
    const gtsam::Pose3 drift = graph_.driftCorrection();
    std::scoped_lock lock(mutex_);
    correction_.version++;
    correction_.anchor_keyframe = graph_.lastNode().value_or(0);
    correction_.T_drift = drift;
    correction_.loops = graph_.loopCount();
}

std::expected<std::shared_ptr<LoopClosureWorker>, std::string>
loop_closure::CreateLoopClosureWorker(compute::OrtProvider& ort,
                                      const LoopClosureConfig& config,
                                      const data::CamSensorParams& camera,
                                      compute::PerfRecorder& perf) {
    const compute::ComputeDevice device = ort.getGPU()->getComputeDevice();
    const VprModelKind kind = SelectModelKind(device, config);
    const std::filesystem::path& vpr_path = kind == VprModelKind::MegaLoc
                                                ? config.megaloc_model_path
                                                : config.mixvpr_model_path;
    if (vpr_path.empty()) {
        return std::unexpected(std::format(
            "{} model path is not set ({})", ToString(kind),
            kind == VprModelKind::MegaLoc ? "WSLAM_VPR_MEGALOC_MODEL"
                                          : "WSLAM_VPR_MIXVPR_MODEL"));
    }
    if (config.matcher_model_path.empty()
        || config.matcher_keypoint_count <= 0) {
        return std::unexpected("matcher model is not set");
    }

    const bool on_gpu = device == compute::ComputeDevice::Gpu;
    const bool matcher_on_gpu = on_gpu && config.matcher_on_gpu;

    LoopClosureWorker::Models models;
    if (on_gpu) {
        auto auxiliary
            = ort.getGPU()->createAuxiliaryDevice("Loop closure device");
        if (!auxiliary) {
            return std::unexpected("auxiliary device: " + auxiliary.error());
        }
        models.device = std::move(auxiliary).value();
    }

    const auto placement = [&](bool use_webgpu) {
        return compute::OrtProvider::SessionPlacement{
            .use_webgpu_ep = use_webgpu,
            .device = use_webgpu ? models.device : std::nullopt,
            .context_id = kLoopClosureContextId,
            .intra_op_threads = config.intra_op_threads,
            // Idle between keyframes: spinning would take the front end's
            // cores for nothing
            .allow_spinning = false,
        };
    };

    spdlog::info(LOG_ID
                 " place recognition: {} on the {} ({}), matcher on "
                 "the {}",
                 ToString(kind), on_gpu ? "GPU" : "CPU", vpr_path.string(),
                 matcher_on_gpu ? "GPU" : "CPU");

    auto vpr_session = ort.LoadModel(vpr_path, placement(on_gpu));
    if (!vpr_session) {
        return std::unexpected(std::move(vpr_session).error());
    }
    auto place_recognition
        = PlaceRecognitionModel::Create(kind, std::move(vpr_session).value());
    if (!place_recognition) {
        return std::unexpected(std::move(place_recognition).error());
    }
    models.place_recognition = std::move(place_recognition).value();

    auto matcher_session
        = ort.LoadModel(config.matcher_model_path, placement(matcher_on_gpu));
    if (!matcher_session) {
        return std::unexpected(std::move(matcher_session).error());
    }
    auto matcher = CachedMatcher::Create(std::move(matcher_session).value(),
                                         config.matcher_keypoint_count);
    if (!matcher) {
        return std::unexpected(std::move(matcher).error());
    }
    models.matcher = std::move(matcher).value();

    return std::make_shared<LoopClosureWorker>(config, camera,
                                               std::move(models), perf);
}
