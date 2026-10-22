#include <onnxruntime/onnxruntime_cxx_api.h>
#include <spdlog/cfg/env.h>
#include <spdlog/spdlog-inl.h>
#include <spdlog/spdlog.h>
#include <unistd.h>

#include <array>
#include <charconv>
#include <csignal>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include "compute/compute.hpp"
#include "data/euroc_provider.hpp"
#include "data/provider_base.hpp"
#include "data/uzh_fpv_provider.hpp"
#include "wslam/common.hpp"
#include "wslam/wslam.hpp"

using namespace wslam;
using namespace std::chrono_literals;

#define LOG_ID "[MAIN]"

namespace {
// Dataset format the sensor readings come from
enum class Dataset : uint8_t {
    Euroc,   // EUROC_DIR: a EuRoC MAV mav0/ directory
    UzhFpv,  // UZH_FPV_DIR + UZH_FPV_CALIB_DIR: a UZH-FPV text sequence
};

// --dataset=euroc|uzh-fpv, EuRoC by default
std::optional<Dataset> parseDataset(std::span<char*> args) {
    constexpr std::string_view kDatasetFlag = "--dataset=";
    Dataset dataset = Dataset::Euroc;
    for (std::string_view arg : args) {
        if (!arg.starts_with(kDatasetFlag)) {
            continue;
        }
        const auto value = arg.substr(kDatasetFlag.size());
        if (value == "euroc") {
            dataset = Dataset::Euroc;
        } else if (value == "uzh-fpv") {
            dataset = Dataset::UzhFpv;
        } else {
            spdlog::critical(LOG_ID " unknown --dataset value '{}'", value);
            return std::nullopt;
        }
    }
    return dataset;
}

std::unique_ptr<data::ProviderBase<2U>> createProvider(Dataset dataset) {
    switch (dataset) {
        case Dataset::UzhFpv:
            return std::make_unique<data::UzhFpvProvider>(
                data::CreateUzhFpvProviderOpts());
        case Dataset::Euroc:
            break;
    }
    return std::make_unique<data::EurocProvider>(
        data::CreateEurocProviderOpts());
}
}  // namespace

WslamConfig parseArgs(std::span<char*> args) {
    constexpr std::string_view kIterFlag = "--max-iters=";
    constexpr std::string_view kMapOutFlag = "--map-out=";
    constexpr std::string_view kKeyframesFlag = "--keyframes=";
    constexpr std::string_view kRecordFrontendFlag = "--record-frontend=";
    constexpr std::string_view kDeviceFlag = "--device=";
    WslamConfig config;
    for (std::string_view arg : args) {
        if (arg == "-gui") {
            config.enable_gui = true;
        } else if (arg == "-v2") {
            config.use_pipeline_v2 = true;
        } else if (arg.starts_with(kIterFlag)) {
            const auto value = arg.substr(kIterFlag.size());
            uint64_t parsed = 0;
            const auto [ptr, ec] = std::from_chars(
                value.data(), value.data() + value.size(), parsed);
            if (ec != std::errc{} || ptr != value.data() + value.size()) {
                spdlog::warn(
                    LOG_ID " Ignoring malformed --max-iters value '{}'", value);
                continue;
            }
            config.max_iterations = parsed;
        } else if (arg.starts_with(kMapOutFlag)) {
            config.map_out_path = std::filesystem::path(
                std::string(arg.substr(kMapOutFlag.size())));
        } else if (arg.starts_with(kKeyframesFlag)) {
            config.keyframes_path = std::filesystem::path(
                std::string(arg.substr(kKeyframesFlag.size())));
        } else if (arg == "-no-mapping") {
            config.enable_mapping = false;
        } else if (arg == "-no-loop-closure") {
            config.enable_loop_closure = false;
        } else if (arg == "-no-frame-prefilter") {
            config.enable_frame_prefilter = false;
        } else if (arg.starts_with(kDeviceFlag)) {
            const auto value = arg.substr(kDeviceFlag.size());
            if (value == "gpu") {
                config.compute_device = compute::ComputeDevice::Gpu;
            } else if (value == "cpu") {
                config.compute_device = compute::ComputeDevice::Cpu;
            } else {
                spdlog::warn(LOG_ID " Ignoring unknown --device value '{}'",
                             value);
            }
        } else if (arg.starts_with(kRecordFrontendFlag)) {
            config.frontend_recording_out = std::filesystem::path(
                std::string(arg.substr(kRecordFrontendFlag.size())));
        }
    }
    return config;
}

namespace {
volatile std::sig_atomic_t g_shutdown_requested = 0;

// Handle a shutdown signal
extern "C" void handleShutdownSignal(int signal) {
    (void)signal;

    g_shutdown_requested = 1;
    // A second interruption signal like a Ctrl+C kills the program
    // Helps if program hangs after the first shutdown request
    std::signal(SIGINT, SIG_DFL);
    std::signal(SIGTERM, SIG_DFL);
}
}  // namespace

int main_test(WslamConfig config, Dataset dataset) {
    compute::PerfRecorder::setThreadId("main");

    compute::Compute comp;
    if (auto err
        = comp.prepare(compute::createDefaultConfig(config.compute_device))) {
        spdlog::critical(LOG_ID " initializing: {}", err.value());
        std::terminate();
    }

    wslam::GpuSharedBindings shared(comp.getGPUPtr(), comp.getStorage());
    if (auto err = shared.initialize()) {
        spdlog::critical(LOG_ID " shared: {}", err.value());
        std::terminate();
    }

    std::unique_ptr<data::ProviderBase<2U>> provider = createProvider(dataset);

    if (auto err = config.use_pipeline_v2
                       ? CreateWslamPipelineV2(comp, *provider, config)
                       : CreateWslamPipeline(comp, shared, *provider, config)) {
        spdlog::critical(LOG_ID " creating wslam pipeline: {}",
                         std::move(err).value());
        std::terminate();
    }

    if (auto err = comp.initizalize()) {
        spdlog::critical(LOG_ID " initializing stages: {}", err.value());
        std::terminate();
    }

    // Stop the loop cleanly on Ctrl+C (SIGINT) or kill (SIGTERM) so finalize()
    // still flushes the iSAM2 worker and exports the map. The handler re-arms
    // the OS defaults, so a second signal force-quits.
    std::signal(SIGINT, handleShutdownSignal);
    std::signal(SIGTERM, handleShutdownSignal);

    for (uint64_t i = 0;
         (config.max_iterations == 0 || i < config.max_iterations)
         && g_shutdown_requested == 0;
         ++i) {
        auto err = comp.execute();
        if (i % 100 == 0) {
            // Warn level so progress stays visible when info is disabled.
            spdlog::warn(LOG_ID " Handled frame {}", i);
        }

        if (err) {
            // If the interrupt landed mid-execute and surfaced as an error,
            // fall through to finalize rather than aborting the export.
            if (g_shutdown_requested != 0) {
                break;
            }
            spdlog::critical(LOG_ID " executing: {}", err.value());
            return 1;
        }
    }

    if (g_shutdown_requested != 0) {
        spdlog::info(LOG_ID
                     " Shutdown signal received; stopping loop and "
                     "finalizing (flush + export). Send the signal again to "
                     "force-quit.");
    } else if (config.max_iterations != 0) {
        spdlog::info(LOG_ID " Reached max-iters limit ({}); exiting",
                     config.max_iterations);
    }

    if (auto err = comp.finalize()) {
        spdlog::error(LOG_ID " Could not finalize. Errors: {}",
                      std::move(err).value());
    }

    return 0;
}

namespace {
std::string formatShape(const std::vector<int64_t>& shape) {
    std::string text;
    for (size_t i = 0; i < shape.size(); ++i) {
        text += (i == 0 ? "" : ", ") + std::to_string(shape[i]);
    }
    return text;
}

// Sums every element of a numeric output tensor as double, whatever its type.
double sumTensor(const Ort::Value& tensor) {
    const auto info = tensor.GetTensorTypeAndShapeInfo();
    const size_t count = info.GetElementCount();
    const auto sum = [&]<typename T>() {
        const T* data = tensor.GetTensorData<T>();
        double total = 0.0;
        for (size_t i = 0; i < count; ++i) {
            total += static_cast<double>(data[i]);
        }
        return total;
    };
    switch (info.GetElementType()) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            return sum.operator()<float>();
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
            return sum.operator()<double>();
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            return sum.operator()<int64_t>();
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
            return sum.operator()<int32_t>();
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
            return sum.operator()<uint8_t>();
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
            return sum.operator()<bool>();
        default:
            spdlog::warn(LOG_ID " Unsupported output element type {}",
                         static_cast<int>(info.GetElementType()));
            return 0.0;
    }
}
}  // namespace

int ort_main(int argc, char** argv) {
    (void)argc;
    (void)argv;
    constexpr const char* kModelPath
        = "./resources/raco_aliked_lightglue_pipeline_k1024.onnx";
    // Substitutes for the model's dynamic dims (batch = image pair, H and W
    // must be multiples of 32).
    constexpr int64_t kBatch = 2;
    constexpr int64_t kHeight = 480;
    constexpr int64_t kWidth = 640;

    try {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "uni_slam");

        Ort::SessionOptions options;
        options.AppendExecutionProvider("WebGPU", {});
        Ort::Session session{env, kModelPath, options};

        Ort::AllocatorWithDefaultOptions allocator;
        const size_t input_count = session.GetInputCount();
        const size_t output_count = session.GetOutputCount();

        std::vector<Ort::AllocatedStringPtr> name_holders;
        std::vector<const char*> input_names;
        std::vector<const char*> output_names;
        for (size_t i = 0; i < input_count; ++i) {
            name_holders.push_back(session.GetInputNameAllocated(i, allocator));
            input_names.push_back(name_holders.back().get());
        }
        for (size_t i = 0; i < output_count; ++i) {
            name_holders.push_back(
                session.GetOutputNameAllocated(i, allocator));
            output_names.push_back(name_holders.back().get());
        }

        // Fill every input with a deterministic pattern in [0, 1].
        const Ort::MemoryInfo memory_info
            = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::vector<std::vector<float>> input_buffers;
        std::vector<Ort::Value> inputs;
        for (size_t i = 0; i < input_count; ++i) {
            const auto type_info = session.GetInputTypeInfo(i);
            const auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            if (tensor_info.GetElementType()
                != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
                spdlog::error(LOG_ID " Input '{}' is not float32",
                              input_names[i]);
                return 1;
            }

            std::vector<int64_t> shape = tensor_info.GetShape();
            const std::array<int64_t, 4> dynamic_fill{kBatch, 3, kHeight,
                                                      kWidth};
            for (size_t d = 0; d < shape.size(); ++d) {
                if (shape[d] < 0) {
                    shape[d] = d < dynamic_fill.size() ? dynamic_fill[d] : 1;
                }
            }
            spdlog::info(LOG_ID " Input '{}' shape [{}]", input_names[i],
                         formatShape(shape));

            size_t element_count = 1;
            for (const int64_t dim : shape) {
                element_count *= static_cast<size_t>(dim);
            }
            auto& buffer = input_buffers.emplace_back(element_count);
            for (size_t e = 0; e < element_count; ++e) {
                buffer[e] = static_cast<float>((e * 7919U) % 256U) / 255.0F;
            }
            inputs.push_back(Ort::Value::CreateTensor<float>(
                memory_info, buffer.data(), buffer.size(), shape.data(),
                shape.size()));
        }

        auto outputs = session.Run(Ort::RunOptions{nullptr}, input_names.data(),
                                   inputs.data(), inputs.size(),
                                   output_names.data(), output_names.size());

        for (size_t i = 0; i < outputs.size(); ++i) {
            const auto info = outputs[i].GetTensorTypeAndShapeInfo();
            spdlog::info(LOG_ID " Output '{}' shape [{}] sum = {}",
                         output_names[i], formatShape(info.GetShape()),
                         sumTensor(outputs[i]));
        }
    } catch (const Ort::Exception& e) {
        spdlog::error(LOG_ID " ONNX Runtime error: {}", e.what());
        return 1;
    }

    return 0;
}

int main(int argc, char* argv[]) {
#ifndef NDEBUG
    spdlog::set_level(spdlog::level::debug);
#else
    spdlog::set_level(spdlog::level::info);
#endif
    // SPDLOG_LEVEL=info (etc.) overrides the compiled-in default at
    // runtime — e.g. to see the per-pass timing lines logged by Stage
    // without rebuilding.
    spdlog::cfg::load_env_levels();

    if (argc > 1 && std::string_view(argv[1]) == "--ort") {
        return ort_main(argc, argv);
    }

    const std::span args{argv + 1, static_cast<size_t>(argc - 1)};
    const WslamConfig config = parseArgs(args);
    const std::optional<Dataset> dataset = parseDataset(args);
    if (!dataset) {
        return 1;
    }

    // The v1 mapping stage selects keyframes itself from every frame
    if (!config.keyframes_path.empty() && !config.use_pipeline_v2) {
        spdlog::critical(LOG_ID " --keyframes is only supported with -v2");
        return 1;
    }
    if (!config.frontend_recording_out.empty()
        && (!config.use_pipeline_v2 || !config.enable_mapping)) {
        spdlog::critical(LOG_ID
                         " --record-frontend needs -v2 with the mapping stage: "
                         "the recording replays its keyframe decisions");
        return 1;
    }

    spdlog::info(LOG_ID " Device: {}, GUI: {}, max iterations: {}, map_out: {}",
                 config.compute_device, config.enable_gui,
                 config.max_iterations == 0
                     ? std::string("unlimited")
                     : std::to_string(config.max_iterations),
                 config.map_out_path.empty() ? std::string("(none)")
                                             : config.map_out_path.string());
    return main_test(config, *dataset);
}
