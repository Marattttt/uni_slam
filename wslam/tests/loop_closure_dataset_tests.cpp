// Place recognition on real V1_01_easy frames: a revisit of the same place,
// picked with the ground truth, must look more alike to the model than a
// view of another part of the room. Runs every model whose path is set
// (WSLAM_VPR_MIXVPR_MODEL, WSLAM_VPR_MEGALOC_MODEL) on ONNX Runtime's CPU EP;
// skips without EUROC_DIR

#include <gtest/gtest.h>
#include <spdlog/spdlog.h>

#include <Eigen/Geometry>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "compute/onnx_interop.hpp"
#include "data/euroc_provider.hpp"
#include "data/provider_base.hpp"
#include "map_v2/loop_closure/model_io.hpp"
#include "map_v2/loop_closure/place_recognition.hpp"

namespace lc = wslam::v2::mapping::loop_closure;
namespace data = wslam::data;

namespace {
const char* Env(const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr && *value != '\0' ? value : nullptr;
}

struct GroundTruthPose {
    uint64_t timestamp;
    Eigen::Vector3d position;
    Eigen::Quaterniond orientation;  // body in world
};

std::vector<GroundTruthPose> ReadGroundTruth(const std::filesystem::path& csv) {
    std::vector<GroundTruthPose> poses;
    std::ifstream file(csv);
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line.front() == '#') {
            continue;
        }
        std::stringstream stream(line);
        std::string field;
        std::vector<double> values;
        uint64_t timestamp = 0;
        for (int column = 0; std::getline(stream, field, ','); ++column) {
            if (column == 0) {
                timestamp = std::stoull(field);
            } else if (column <= 7) {
                values.push_back(std::stod(field));
            }
        }
        poses.push_back(GroundTruthPose{
            .timestamp = timestamp,
            .position = {values[0], values[1], values[2]},
            .orientation = Eigen::Quaterniond{values[3], values[4], values[5],
                                              values[6]},
        });
    }
    return poses;
}

struct Frame {
    uint64_t timestamp;
    std::filesystem::path path;
    Eigen::Vector3d position;
    // Camera optical axis in the world
    Eigen::Vector3d view;
};

// Every step-th cam0 frame inside the ground truth, with its camera pose
std::vector<Frame> ReadFrames(const std::filesystem::path& mav0,
                              const Eigen::Matrix3d& R_body_cam, size_t step) {
    const auto truth
        = ReadGroundTruth(mav0 / "state_groundtruth_estimate0" / "data.csv");

    std::vector<Frame> frames;
    std::ifstream file(mav0 / "cam0" / "data.csv");
    std::string line;
    size_t index = 0;
    size_t cursor = 0;
    while (std::getline(file, line)) {
        if (line.empty() || line.front() == '#') {
            continue;
        }
        if (index++ % step != 0) {
            continue;
        }
        const uint64_t timestamp = std::stoull(line.substr(0, line.find(',')));
        while (cursor + 1 < truth.size()
               && truth[cursor + 1].timestamp <= timestamp) {
            ++cursor;
        }
        if (truth.empty() || truth[cursor].timestamp > timestamp
            || timestamp - truth[cursor].timestamp > 10'000'000) {
            continue;
        }
        const GroundTruthPose& pose = truth[cursor];
        frames.push_back(Frame{
            .timestamp = timestamp,
            .path = mav0 / "cam0" / "data" / (std::to_string(timestamp) + ".png"),
            .position = pose.position,
            .view = pose.orientation * (R_body_cam * Eigen::Vector3d::UnitZ()),
        });
    }
    return frames;
}

double AngleBetween(const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
    return std::acos(std::clamp(a.normalized().dot(b.normalized()), -1.0, 1.0));
}

struct FramePair {
    const Frame* a = nullptr;
    const Frame* b = nullptr;
};

// The closest same-view pair at least min_gap_s apart
std::optional<FramePair> FindRevisit(const std::vector<Frame>& frames,
                                     double min_gap_s) {
    std::optional<FramePair> best;
    double best_score = 1e9;
    for (size_t i = 0; i < frames.size(); ++i) {
        for (size_t j = i + 1; j < frames.size(); ++j) {
            const double gap
                = static_cast<double>(frames[j].timestamp - frames[i].timestamp)
                  * 1e-9;
            if (gap < min_gap_s) {
                continue;
            }
            const double distance
                = (frames[i].position - frames[j].position).norm();
            const double angle = AngleBetween(frames[i].view, frames[j].view);
            const double score = distance + angle;  // metres + radians
            if (distance < 0.3 && angle < 0.2 && score < best_score) {
                best_score = score;
                best = FramePair{.a = &frames[i], .b = &frames[j]};
            }
        }
    }
    return best;
}

// A frame of the first one's revisit pair looking the opposite way
const Frame* FindOtherPlace(const std::vector<Frame>& frames,
                            const Frame& reference) {
    const Frame* best = nullptr;
    double best_angle = 0.0;
    for (const Frame& frame : frames) {
        const double angle = AngleBetween(frame.view, reference.view);
        if (angle > best_angle) {
            best_angle = angle;
            best = &frame;
        }
    }
    return best_angle > 2.0 ? best : nullptr;
}

lc::PlaceImage LoadImage(const Frame& frame) {
    auto loaded = data::getLocalFrame(frame.path, frame.timestamp);
    EXPECT_TRUE(loaded.has_value()) << frame.path;
    return lc::PlaceImage::FromFrame(loaded.value());
}
}  // namespace

class LoopClosureDatasetTest : public ::testing::TestWithParam<lc::VprModelKind> {
};

TEST_P(LoopClosureDatasetTest, RevisitIsMoreSimilarThanAnotherPlace) {
    const lc::VprModelKind kind = GetParam();
    const char* euroc = Env("EUROC_DIR");
    const char* model_path = Env(kind == lc::VprModelKind::MegaLoc
                                     ? "WSLAM_VPR_MEGALOC_MODEL"
                                     : "WSLAM_VPR_MIXVPR_MODEL");
    if (euroc == nullptr || model_path == nullptr) {
        GTEST_SKIP() << "needs EUROC_DIR and the model path";
    }

    data::EurocProvider provider(data::CreateEurocProviderOpts());
    const auto params = provider.getSensorParams();
    ASSERT_TRUE(params.has_value()) << params.error();
    const Eigen::Matrix3d R_body_cam
        = params->cams.at(0).T_BS.topLeftCorner<3, 3>();

    const auto frames = ReadFrames(euroc, R_body_cam, 10);
    ASSERT_GT(frames.size(), 100U);

    const auto revisit = FindRevisit(frames, 30.0);
    ASSERT_TRUE(revisit.has_value()) << "no revisit in the sequence";
    const Frame* other = FindOtherPlace(frames, *revisit->a);
    ASSERT_NE(other, nullptr);

    auto env = std::make_shared<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "test");
    wslam::compute::OrtProvider ort(nullptr, env);
    wslam::compute::OrtProvider::SessionPlacement placement;
    placement.use_webgpu_ep = false;
    placement.intra_op_threads = 8;
    auto session = ort.LoadModel(model_path, placement);
    ASSERT_TRUE(session.has_value()) << session.error();
    auto model = lc::PlaceRecognitionModel::Create(kind, std::move(*session));
    ASSERT_TRUE(model.has_value()) << model.error();

    const auto describe = [&](const Frame& frame) {
        auto descriptor = (*model)->describe(LoadImage(frame));
        EXPECT_TRUE(descriptor.has_value()) << descriptor.error();
        return std::move(descriptor).value();
    };
    const lc::PlaceDescriptor a = describe(*revisit->a);
    const lc::PlaceDescriptor b = describe(*revisit->b);
    const lc::PlaceDescriptor c = describe(*other);

    const float revisit_similarity = a.similarity(b);
    const float other_similarity = a.similarity(c);
    spdlog::info(
        "{}: revisit {:.1f} s apart, {:.2f} m: similarity {:.3f}; "
        "opposite view: {:.3f}",
        lc::ToString(kind),
        static_cast<double>(revisit->b->timestamp - revisit->a->timestamp)
            * 1e-9,
        (revisit->a->position - revisit->b->position).norm(),
        revisit_similarity, other_similarity);

    // For a parity check against a Python reference: the first frame and
    // its descriptor, one value per line
    if (const char* parity_out = Env("WSLAM_LC_PARITY_OUT")) {
        std::ofstream out(std::string{parity_out} + "_"
                          + std::string{lc::ToString(kind)} + ".txt");
        out << revisit->a->path.string() << '\n';
        for (const float value : a.values) {
            out << value << '\n';
        }
    }

    EXPECT_GT(revisit_similarity, other_similarity + 0.1F);
    EXPECT_GT(revisit_similarity, lc::MatchAcceptance{}.min_similarity);
}

INSTANTIATE_TEST_SUITE_P(Models, LoopClosureDatasetTest,
                         ::testing::Values(lc::VprModelKind::MixVpr,
                                           lc::VprModelKind::MegaLoc),
                         [](const auto& info) {
                             return std::string{lc::ToString(info.param)};
                         });
