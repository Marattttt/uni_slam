#include "frontend_recording.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstdint>
#include <format>
#include <fstream>
#include <iterator>
#include <nlohmann/json.hpp>
#include <utility>

#include "common.hpp"
#include "compute/compute.hpp"
#include "compute/stage.hpp"

using namespace wslam;
using nlohmann::json;

#define LOG_ID "[Frontend recording]"

namespace {

// ------------------------------------------------------------ Eigen <-> json

template <typename Matrix>
json MatrixToJson(const Matrix& matrix) {
    json values = json::array();
    for (Eigen::Index row = 0; row < matrix.rows(); ++row) {
        for (Eigen::Index col = 0; col < matrix.cols(); ++col) {
            values.push_back(matrix(row, col));
        }
    }
    return values;
}

template <typename Matrix>
Matrix MatrixFromJson(const json& values) {
    Matrix matrix;
    if (values.size() != static_cast<size_t>(matrix.size())) {
        throw std::runtime_error(std::format("expected {} values, got {}",
                                             matrix.size(), values.size()));
    }
    size_t index = 0;
    for (Eigen::Index row = 0; row < matrix.rows(); ++row) {
        for (Eigen::Index col = 0; col < matrix.cols(); ++col) {
            matrix(row, col)
                = values.at(index++).get<typename Matrix::Scalar>();
        }
    }
    return matrix;
}

// ------------------------------------------------------------ sensors

json CameraToJson(const data::CamSensorParams& camera) {
    return {
        {"T_BS", MatrixToJson(camera.T_BS)},
        {"rate_hz", camera.rate_hz},
        {"resolution", camera.resolution},
        {"camera_model", camera.camera_model},
        {"intrinsics", MatrixToJson(camera.intrinsics)},
        {"distortion_model", camera.distortion_model},
        {"distortion_coefficients",
         MatrixToJson(camera.distortion_coefficients)},
    };
}

data::CamSensorParams CameraFromJson(const json& value) {
    return {
        .T_BS = MatrixFromJson<Eigen::Matrix4d>(value.at("T_BS")),
        .rate_hz = value.at("rate_hz").get<uint32_t>(),
        .resolution = value.at("resolution").get<std::array<int, 2>>(),
        .camera_model = value.at("camera_model").get<std::string>(),
        .intrinsics = MatrixFromJson<Eigen::Vector4d>(value.at("intrinsics")),
        .distortion_model = value.at("distortion_model").get<std::string>(),
        .distortion_coefficients
        = MatrixFromJson<Eigen::Vector4d>(value.at("distortion_coefficients")),
    };
}

json ImuParamsToJson(const data::IMUSensorParams& imu) {
    return {
        {"T_BS", MatrixToJson(imu.T_BS)},
        {"rate_hz", imu.rate_hz},
        {"gyroscope_noise_density", imu.gyroscope_noise_density},
        {"gyroscope_random_walk", imu.gyroscope_random_walk},
        {"accelerometer_noise_density", imu.accelerometer_noise_density},
        {"accelerometer_random_walk", imu.accelerometer_random_walk},
    };
}

data::IMUSensorParams ImuParamsFromJson(const json& value) {
    return {
        .T_BS = MatrixFromJson<Eigen::Matrix4d>(value.at("T_BS")),
        .rate_hz = value.at("rate_hz").get<uint32_t>(),
        .gyroscope_noise_density
        = value.at("gyroscope_noise_density").get<double>(),
        .gyroscope_random_walk
        = value.at("gyroscope_random_walk").get<double>(),
        .accelerometer_noise_density
        = value.at("accelerometer_noise_density").get<double>(),
        .accelerometer_random_walk
        = value.at("accelerometer_random_walk").get<double>(),
    };
}

// ------------------------------------------------------------ frames
// Rows are flat arrays: the file holds ~1000 of them per frame

// [timestamp, wx, wy, wz, ax, ay, az]
json ImuReadingToJson(const data::IMUReading& reading) {
    json row = json::array({reading.timestamp});
    for (const float value : reading.vals) {
        row.push_back(value);
    }
    return row;
}

data::IMUReading ImuReadingFromJson(const json& row) {
    data::IMUReading reading{.timestamp = row.at(0).get<uint64_t>(),
                             .vals = {}};
    for (Eigen::Index i = 0; i < reading.vals.size(); ++i) {
        reading.vals[i] = row.at(static_cast<size_t>(i) + 1).get<float>();
    }
    return reading;
}

// [prev_index, curr_index, score, prev_px(2), curr_px(2), prev_ray(2),
//  curr_ray(2)]
json CorrespondenceToJson(const v2::KeypointCorrespondence& match) {
    return json::array({
        match.prev_index,
        match.curr_index,
        match.score,
        match.prev_px.x(),
        match.prev_px.y(),
        match.curr_px.x(),
        match.curr_px.y(),
        match.prev_ray.x(),
        match.prev_ray.y(),
        match.curr_ray.x(),
        match.curr_ray.y(),
    });
}

v2::KeypointCorrespondence CorrespondenceFromJson(const json& row) {
    const auto at
        = [&row](size_t index) { return row.at(index).get<double>(); };
    return {
        .prev_index = row.at(0).get<uint32_t>(),
        .curr_index = row.at(1).get<uint32_t>(),
        .score = row.at(2).get<float>(),
        .prev_px = {at(3), at(4)},
        .curr_px = {at(5), at(6)},
        .prev_ray = {at(7), at(8)},
        .curr_ray = {at(9), at(10)},
    };
}

json FrameToJson(const v2::RecordedFrame& frame) {
    json imu = json::array();
    for (const auto& reading : frame.imu) {
        imu.push_back(ImuReadingToJson(reading));
    }

    json value = {
        {"frame_index", frame.frame_index},
        {"timestamp_ns", frame.timestamp_ns},
        {"imu", std::move(imu)},
        {"reference_frame_index", frame.reference_frame_index},
        {"reference_advanced", frame.reference_advanced},
    };

    if (frame.matches) {
        json matches = json::array();
        for (const auto& match : *frame.matches) {
            matches.push_back(CorrespondenceToJson(match));
        }
        value["matches"] = std::move(matches);
    }
    return value;
}

v2::RecordedFrame FrameFromJson(const json& value) {
    v2::RecordedFrame frame{
        .frame_index = value.at("frame_index").get<uint64_t>(),
        .timestamp_ns = value.at("timestamp_ns").get<uint64_t>(),
        .imu = {},
        .matches = std::nullopt,
        .reference_frame_index
        = value.at("reference_frame_index").get<uint64_t>(),
        .reference_advanced = value.at("reference_advanced").get<bool>(),
    };
    for (const auto& row : value.at("imu")) {
        frame.imu.push_back(ImuReadingFromJson(row));
    }
    if (value.contains("matches")) {
        auto& matches = frame.matches.emplace();
        matches.reserve(value.at("matches").size());
        for (const auto& row : value.at("matches")) {
            matches.push_back(CorrespondenceFromJson(row));
        }
    }
    return frame;
}

// [id, timestamp_ns, R (9, row-major), t (3)]
json KeyframeToJson(const KeyframePose& keyframe) {
    json row = json::array({keyframe.id.v, keyframe.timestamp_ns});
    for (const auto& value : MatrixToJson(keyframe.R_world_cam)) {
        row.push_back(value);
    }
    for (const auto& value : MatrixToJson(keyframe.t_world_cam)) {
        row.push_back(value);
    }
    return row;
}

KeyframePose KeyframeFromJson(const json& row) {
    constexpr size_t kRotationBegin = 2;
    constexpr size_t kTranslationBegin = 11;
    return {
        .id = PoseId{row.at(0).get<uint64_t>()},
        .timestamp_ns = row.at(1).get<uint64_t>(),
        .R_world_cam = MatrixFromJson<Eigen::Matrix3d>(json(
            row.begin() + kRotationBegin, row.begin() + kTranslationBegin)),
        .t_world_cam = MatrixFromJson<Eigen::Vector3d>(
            json(row.begin() + kTranslationBegin, row.end())),
    };
}

}  // namespace

// ------------------------------------------------------------ file

std::optional<std::string> v2::SaveFrontendRecording(
    const FrontendRecording& recording, const std::filesystem::path& path) {
    json frames = json::array();
    for (const auto& frame : recording.frames) {
        frames.push_back(FrameToJson(frame));
    }
    json golden_keyframes = json::array();
    for (const auto& keyframe : recording.golden_keyframes) {
        golden_keyframes.push_back(KeyframeToJson(keyframe));
    }

    const json document = {
        {"format_version", FrontendRecording::kFormatVersion},
        {"source", recording.source},
        {"model_path", recording.model_path},
        {"keypoint_count", recording.keypoint_count},
        {"camera", CameraToJson(recording.camera)},
        {"imu_params", ImuParamsToJson(recording.imu_params)},
        {"frames", std::move(frames)},
        {"golden_keyframes", std::move(golden_keyframes)},
        {"golden_landmark_count", recording.golden_landmark_count},
    };

    const std::vector<std::uint8_t> bytes = json::to_msgpack(document);

    if (path.has_parent_path()) {
        std::error_code error;
        std::filesystem::create_directories(path.parent_path(), error);
        if (error) {
            return std::format("creating '{}': {}", path.parent_path().string(),
                               error.message());
        }
    }

    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file) {
        return std::format("could not open '{}' for writing", path.string());
    }
    file.write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
    if (!file) {
        return std::format("writing '{}' failed", path.string());
    }
    return std::nullopt;
}

std::expected<v2::FrontendRecording, std::string> v2::LoadFrontendRecording(
    const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return std::unexpected(
            std::format("could not open '{}'", path.string()));
    }
    const std::vector<std::uint8_t> bytes{std::istreambuf_iterator<char>(file),
                                          std::istreambuf_iterator<char>()};

    try {
        const json document = json::from_msgpack(bytes);

        const int version = document.at("format_version").get<int>();
        if (version != FrontendRecording::kFormatVersion) {
            return std::unexpected(
                std::format("format version {}, this build reads {}", version,
                            FrontendRecording::kFormatVersion));
        }

        FrontendRecording recording{
            .source = document.at("source").get<std::string>(),
            .model_path = document.at("model_path").get<std::string>(),
            .keypoint_count = document.at("keypoint_count").get<int64_t>(),
            .camera = CameraFromJson(document.at("camera")),
            .imu_params = ImuParamsFromJson(document.at("imu_params")),
            .frames = {},
            .golden_keyframes = {},
            .golden_landmark_count
            = document.at("golden_landmark_count").get<size_t>(),
        };
        for (const auto& frame : document.at("frames")) {
            recording.frames.push_back(FrameFromJson(frame));
        }
        for (const auto& keyframe : document.at("golden_keyframes")) {
            recording.golden_keyframes.push_back(KeyframeFromJson(keyframe));
        }
        return recording;
    } catch (const std::exception& e) {
        return std::unexpected(
            std::format("parsing '{}': {}", path.string(), e.what()));
    }
}

// ------------------------------------------------------------ recorder

v2::FrontendRecorder::FrontendRecorder(AnyBag& storage,
                                       std::filesystem::path output,
                                       FrontendRecording header)
    : storage_(storage),
      output_(std::move(output)),
      recording_(std::move(header)) {}

std::optional<std::string> v2::FrontendRecorder::captureFrame() {
    const auto timestamp
        = storage_.get<uint64_t>(ResourceIdentifier::FrameTimestampNsName);
    if (!timestamp) {
        return "no frame timestamp";
    }

    RecordedFrame frame{
        .frame_index = recording_.frames.size(),
        .timestamp_ns = timestamp.value(),
        .imu = {},
        // Taken so a frame whose pose estimate stopped early can't record the
        // previous frame's matches
        .matches = storage_.take<KeypointMatches>(
            ResourceIdentifier::KeypointMatchesName),
        .reference_frame_index = reference_frame_index_,
        .reference_advanced = false,
    };

    // The buffer still holds samples left over from earlier frames; the new
    // ones are those after the last recorded sample
    if (const auto buffer = storage_.getPtr<std::vector<data::IMUReading>>(
            ResourceIdentifier::GetImuVecName())) {
        for (const auto& reading : *buffer.value()) {
            if (reading.timestamp > last_imu_timestamp_) {
                frame.imu.push_back(reading);
            }
        }
    }
    if (!frame.imu.empty()) {
        last_imu_timestamp_ = frame.imu.back().timestamp;
    }

    recording_.frames.push_back(std::move(frame));
    return std::nullopt;
}

std::optional<std::string> v2::FrontendRecorder::captureDecision() {
    if (recording_.frames.empty()) {
        return "no frame captured before the decision";
    }

    RecordedFrame& frame = recording_.frames.back();
    // Left in place: the model's preserve pass consumes it next frame
    frame.reference_advanced
        = storage_.get<bool>(ResourceIdentifier::FeatureReferenceAdvanceName)
              .value_or(false);
    if (frame.reference_advanced) {
        reference_frame_index_ = frame.frame_index;
    }
    return std::nullopt;
}

std::optional<std::string> v2::FrontendRecorder::save() {
    // The smoother's map: with loop closure, MapSnapshotName holds the
    // loop-corrected copy, which the replay (no loop closure) can't reproduce
    auto snapshot
        = storage_.getPtr<MapSnapshot>(ResourceIdentifier::MapSnapshotVioName);
    if (!snapshot) {
        snapshot
            = storage_.getPtr<MapSnapshot>(ResourceIdentifier::MapSnapshotName);
    }
    if (snapshot) {
        recording_.golden_keyframes = snapshot.value()->keyframes;
        recording_.golden_landmark_count = snapshot.value()->landmarks.size();
    }

    const size_t with_matches = std::ranges::count_if(
        recording_.frames,
        [](const auto& frame) { return frame.matches.has_value(); });
    spdlog::info(LOG_ID
                 " Saving {} frames ({} with matches), {} golden keyframes to "
                 "'{}'",
                 recording_.frames.size(), with_matches,
                 recording_.golden_keyframes.size(), output_.string());

    return SaveFrontendRecording(recording_, output_);
}

// ------------------------------------------------------------ replay

v2::FrontendReplay::FrontendReplay(AnyBag& storage, FrontendRecording recording)
    : storage_(storage), recording_(std::move(recording)) {}

void v2::FrontendReplay::publishSensorParams() {
    storage_.set(ResourceIdentifier::GetCameraIntrinsicsName(0),
                 recording_.camera);
    storage_.set(ResourceIdentifier::ImuParamsName, recording_.imu_params);
    storage_.set(ResourceIdentifier::GetImuVecName(),
                 std::vector<data::IMUReading>{});
}

std::optional<std::string> v2::FrontendReplay::publishFrame() {
    if (finished()) {
        return compute::kComputeStopExecution;
    }
    const RecordedFrame& frame = recording_.frames.at(next_frame_++);

    storage_.set(ResourceIdentifier::FrameTimestampNsName, frame.timestamp_ns);

    auto buffer = storage_
                      .take<std::vector<data::IMUReading>>(
                          ResourceIdentifier::GetImuVecName())
                      .value_or(std::vector<data::IMUReading>{});
    std::ranges::copy(frame.imu, std::back_inserter(buffer));
    storage_.set(ResourceIdentifier::GetImuVecName(), std::move(buffer));

    // Never leave an earlier frame's matches behind
    (void)storage_.take<KeypointMatches>(
        ResourceIdentifier::KeypointMatchesName);
    if (!frame.matches) {
        return compute::kStageStopExecution;
    }
    storage_.set(ResourceIdentifier::KeypointMatchesName, *frame.matches);
    return std::nullopt;
}

std::optional<std::string> v2::FrontendReplay::checkDecision() {
    if (next_frame_ == 0) {
        return "no frame replayed before the decision";
    }
    const RecordedFrame& frame = recording_.frames.at(next_frame_ - 1);

    const bool advanced
        = storage_.take<bool>(ResourceIdentifier::FeatureReferenceAdvanceName)
              .value_or(false);
    if (advanced != frame.reference_advanced) {
        divergence_ = Divergence{
            .frame_index = frame.frame_index,
            .recorded_advance = frame.reference_advanced,
            .replayed_advance = advanced,
        };
        spdlog::warn(LOG_ID
                     " Replay diverged at frame {}: recorded reference "
                     "advance {}, replayed {}",
                     frame.frame_index, frame.reference_advanced, advanced);
        return compute::kComputeStopExecution;
    }
    return std::nullopt;
}
