#include "gpu.hpp"

#include <spdlog/spdlog.h>
#include <webgpu/webgpu_cpp.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <format>
#include <fstream>
#include <memory>
#include <optional>
#include <print>
#include <ranges>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "awaiter.hpp"

using namespace wslam::compute;
using namespace std::chrono_literals;

#define LOG_ID "[GPU]"
constexpr std::string getLogId() { return LOG_ID; }

void BufferBinding::swap(BufferBinding& buf2) noexcept {
    gpu_ = {buf2.gpu_};
    buf_type_ = {buf2.buf_type_};
    size_ = {buf2.size_};
    offset_ = {buf2.offset_};

    buf2.gpu_ = nullptr;
    buf2.buf_type_ = {};
    buf2.size_ = {};
    buf2.offset_ = {};
}

BufferBinding::BufferBinding(BufferBinding&& other) noexcept { swap(other); }
BufferBinding& BufferBinding::operator=(BufferBinding&& other) noexcept {
    swap(other);
    return *this;
}

BufferBinding::~BufferBinding() noexcept {
    if (gpu_ != nullptr) {
        (*gpu_).unAssignBuffer(*this);
    }
    gpu_ = nullptr;
}

const wgpu::Device& GPU::getDevice() const { return device_; }
const wgpu::Instance& GPU::getInstance() const { return instance_; }
const wgpu::Limits& GPU::getLimits() const { return limits_; }
Awaiter GPU::getAwaiter() const { return {instance_, device_}; }

size_t GPU::getMinBufferAlignment(BufferType buftype) const {
    using BT = BufferType;

    constexpr auto kMinMappedAlignment = 4;

    switch (buftype) {
        case BT::StorageA:
        case BT::StorageB:
        case BT::SharedStorage:
            return limits_.minStorageBufferOffsetAlignment;
        case BT::Uniform:
            return limits_.minUniformBufferOffsetAlignment;
        case BT::Input:
        case BT::Output:
            return kMinMappedAlignment;
        case BT::COUNT:
            std::unreachable();
    }

    std::unreachable();
}

std::expected<GPU::BufferBindingMap, std::string> GPU::assignBuffersAndOffsets(
    BindGroupBindings bindings) {
    std::expected<GPU::BufferBindingMap, std::string> result{};

    auto handle = [&](const GPU::BindingKey& key, wgpu::BindGroupEntry& entry,
                      BufferType buf_type) -> bool {
        entry.buffer = getBuffer(buf_type);
        auto res = assignBufferRegion(entry, buf_type);

        if (!res) {
            result = std::unexpected(std::format("{}", buf_type) + res.error());
            return false;
        }

        result.value().emplace(key, std::move(res.value()));

        return true;
    };

    for (auto& [key, binding] : bindings) {
        bool ok = handle(key, binding.bg_entry, binding.buf_type);
        if (!ok) {
            return result;
        }
    }

    return result;
}

const wgpu::Buffer& GPU::getBuffer(BufferType buftype) const {
    const wgpu::Buffer* buf = nullptr;

    switch (buftype) {
        case BufferType::StorageA:
            buf = &storage_buf_a_;
            break;
        case BufferType::StorageB:
            buf = &storage_buf_b_;
            break;
        case BufferType::Input:
            buf = &input_buf_;
            break;
        case BufferType::Output:
            buf = &output_buf_;
            break;
        case BufferType::Uniform:
            buf = &uniform_buf_;
            break;
        case BufferType::SharedStorage:
            buf = &shared_buf_;
            break;
        case BufferType::COUNT:
            assert(false && "Invalid buffer type COUNT");
    }

    assert(buf != nullptr);

    return *buf;
}

std::expected<BufferBinding, std::string> GPU::assignBufferRegion(
    wgpu::BindGroupEntry& entry, BufferType buftype) {
    std::vector<BufferRegion>& regions
        = buf_regions_.at(static_cast<size_t>(buftype));

    std::expected<BufferBinding, std::string> result
        = std::unexpected("uninitialized");

    const size_t offset_unset = std::numeric_limits<size_t>::max();
    entry.offset = offset_unset;

    auto region = std::ranges::find_if(regions, [&](BufferRegion& reg) {
        return reg.is_free && reg.size >= entry.size;
    });

    if (region == regions.end()) {
        result = std::unexpected("could not find a suitable buffer");
        return result;
    }

    // Always a multiple of the minimum alignment
    const size_t region_size = std::invoke([&]() {
        const auto alignment = getMinBufferAlignment(buftype);
        const auto remainder = entry.size % alignment;
        if (remainder == 0) {
            return entry.size;
        }
        return entry.size + alignment - remainder;
    });

    BufferRegion taken{
        .is_free = false,
        .offset = region->offset,
        .size = region_size,
    };

    BufferRegion free{
        .is_free = true,
        .offset = region->offset + region_size,
        .size = region->size - region_size,
    };

    region = regions.erase(region);
    regions.insert(region, {taken, free});

    entry.offset = taken.offset;

    result = BufferBinding{
        BufferBinding::Token{},
        BufferBinding::Params{
            .gpu = *this,
            .buf_type = buftype,
            .size = region_size,
            .offset = entry.offset,
        },
    };

    // Keep the region sorted
    std::ranges::sort(buf_regions_.at(static_cast<size_t>(buftype)),
                      std::less{}, &BufferRegion::offset);

    return result;
}

std::optional<std::string> GPU::initialize() {
    spdlog::info("[GPU] Initializing");

    if (auto err = initInstance()) {
        return "instance: " + err.value();
    }

    if (auto err = initAdapter()) {
        return "adapter: " + err.value();
    }

    if (auto err = initDevice()) {
        return "device: " + err.value();
    }

    if (auto err = initLimits()) {
        return "limits: " + err.value();
    }

    if (auto err = initBuffers()) {
        return "buffers: " + err.value();
    }

    if (auto err = initBufferRegions()) {
        return "buffer regions: " + err.value();
    }

    return std::nullopt;
}

std::optional<std::string> GPU::initInstance() {
    spdlog::info("[GPU] Creating instance");

    static constexpr auto timed_wait_any
        = wgpu::InstanceFeatureName::TimedWaitAny;

    wgpu::InstanceDescriptor descriptor{
        .requiredFeatureCount = 1,
        .requiredFeatures = &timed_wait_any,
    };

    wgpu::Instance instance = wgpu::CreateInstance(&descriptor);
    if (instance == nullptr) {
        return "Instance creation failed!";
    }

    instance_ = std::move(instance);

    return {};
}

std::optional<std::string> GPU::initAdapter() {
    spdlog::info("[GPU] Creating adapter");

    struct UserData {
        wgpu::Adapter adapter_;
        std::string errormsg;
    };

    auto callback = [](wgpu::RequestAdapterStatus status, wgpu::Adapter adapter,
                       const char* message, UserData* userdata) {
        if (status != wgpu::RequestAdapterStatus::Success) {
            userdata->errormsg = message ? message : "Unknown error";
            std::println(stderr, "Failed to get wgpu adapter. Reason: {}",
                         userdata->errormsg);
            return;
        }

        userdata->adapter_ = std::move(adapter);
    };

    wgpu::RequestAdapterOptions options{
        .nextInChain = nullptr,
        .compatibleSurface = nullptr,
    };

    UserData user_data;

    auto callback_mode = wgpu::CallbackMode::WaitAnyOnly;
    const std::uint64_t timeout_ns = 10 * 1'000'000'000LLU;

    instance_.WaitAny(
        instance_.RequestAdapter(&options, callback_mode, callback, &user_data),
        timeout_ns);

    if (user_data.adapter_ == nullptr) {
        std::string err = user_data.errormsg.empty()
                              ? "timeout or unknown error"
                              : user_data.errormsg;
        return "wgpu RequestAdapter failed: " + err;
    }

    adapter_ = std::move(user_data.adapter_);
    return std::nullopt;
}

std::optional<std::string> GPU::initDevice() {
    static const uint64_t gigabyte = 1 << 30;

    spdlog::info("[GPU] Creating device");

    const wgpu::Limits limits{.maxBufferSize = gigabyte};

    const std::array<wgpu::FeatureName, 1> features{
        wgpu::FeatureName::Float32Filterable};

    wgpu::DeviceDescriptor descriptor{};
    descriptor.nextInChain = nullptr;
    descriptor.label = "First device";
    descriptor.requiredFeatureCount = features.size();
    descriptor.requiredFeatures = features.data();
    descriptor.requiredLimits = &limits;
    descriptor.defaultQueue.nextInChain = nullptr;
    descriptor.defaultQueue.label = "Default queue";
    descriptor.SetDeviceLostCallback(
        wgpu::CallbackMode::AllowSpontaneous,
        [callback = device_lost_callback_](
            const wgpu::Device& device, wgpu::DeviceLostReason reason,
            wgpu::StringView msg) { (*callback)(device, reason, msg); });
    descriptor.SetUncapturedErrorCallback(
        [](const wgpu::Device& device, wgpu::ErrorType errtype,
           wgpu::StringView msg, GPU* userdata) {
            auto* gpu = userdata;
            if (gpu && gpu->uncaptured_error_callback_) {
                (*gpu->uncaptured_error_callback_)(device, errtype, msg);
            }
        },
        this);

    struct UserData {
        wgpu::Device device;
        std::string errormsg;
        bool requestEnded = false;
    };
    UserData userData{.device = device_, .errormsg = ""};

    const auto device_callback = [](wgpu::RequestDeviceStatus status,
                                    wgpu::Device device, const char* msg,
                                    UserData* data) {
        if (status == wgpu::RequestDeviceStatus::Success) {
            data->device = std::move(device);
        } else {
            data->errormsg
                = std::format("could not get WebGPU device. reason: {}", msg);
        }
        data->requestEnded = true;
    };

    const wgpu::Future request
        = adapter_.RequestDevice(&descriptor, wgpu::CallbackMode::WaitAnyOnly,
                                 device_callback, &userData);

    const uint64_t timeout_ns = 10 * 1'000'000'000LLU;

    instance_.WaitAny(request, timeout_ns);

    if (userData.errormsg.length() > 0) {
        return userData.errormsg;
    }

    device_ = std::move(userData.device);

    return std::nullopt;
}

std::optional<std::string> GPU::initLimits() {
    spdlog::info(LOG_ID " Getting device limits");

    device_.GetLimits(&limits_);

    return std::nullopt;
}

std::optional<std::string> GPU::initBuffers() {
    spdlog::info("[GPU] Creating buffers");

    using BU = wgpu::BufferUsage;

    struct BufferSpec {
        wgpu::Buffer* target;
        const char* label;
        uint64_t size;
        wgpu::BufferUsage usage;
    };

    const std::array specs = {
        BufferSpec{.target = &input_buf_,
                   .label = "Input buffer",
                   .size = WCOMPUTE_MAP_WRITE_BUF_SIZE,
                   .usage = BU::CopySrc | BU::MapWrite},
        BufferSpec{.target = &uniform_buf_,
                   .label = "Uniform buffer",
                   .size = WCOMPUTE_UNIFORM_BUF_SIZE,
                   .usage = BU::CopySrc | BU::CopyDst | BU::Uniform},
        BufferSpec{.target = &storage_buf_a_,
                   .label = "Storage buffer A",
                   .size = WCOMPUTE_STORAGE_BUF_SIZE,
                   .usage = BU::Storage | BU::CopyDst | BU::CopySrc},
        BufferSpec{.target = &storage_buf_b_,
                   .label = "Storage buffer B",
                   .size = WCOMPUTE_STORAGE_BUF_SIZE,
                   .usage = BU::Storage | BU::CopyDst | BU::CopySrc},
        BufferSpec{.target = &shared_buf_,
                   .label = "Shared storage buffer",
                   .size = WCOMPUTE_STORAGE_BUF_SIZE,
                   .usage = BU::Storage | BU::CopyDst | BU::CopySrc},
        BufferSpec{.target = &output_buf_,
                   .label = "Map read buffer",
                   .size = WCOMPUTE_MAP_READ_BUF_SIZE,
                   .usage = BU::CopyDst | BU::MapRead},
    };

    std::vector<wgpu::FutureWaitInfo> wait_infos;
    wait_infos.reserve(specs.size());

    std::string errors;

    for (const auto& spec : specs) {
        wgpu::BufferDescriptor desc;
        desc.label = spec.label;
        desc.size = spec.size;
        desc.usage = spec.usage;

        device_.PushErrorScope(wgpu::ErrorFilter::OutOfMemory);
        *spec.target = device_.CreateBuffer(&desc);
        wait_infos.push_back(
            {.future = device_.PopErrorScope(
                 wgpu::CallbackMode::WaitAnyOnly,
                 [&errors, &spec](wgpu::PopErrorScopeStatus status,
                                  wgpu::ErrorType errtype, const char* msg) {
                     if (status == wgpu::PopErrorScopeStatus::Error) {
                         errors += std::format(
                             "creating '{}' (error type: {}): {}", spec.label,
                             static_cast<int>(errtype), msg);
                         *spec.target = nullptr;
                     } else {
                         spdlog::debug("[GPU] Created {}", spec.label);
                     }
                 }),
             .completed = false});
    }

    const auto start = std::chrono::steady_clock::now();
    const auto end = start + 20s;

    bool all_futures_complete = true;
    while (std::chrono::steady_clock::now() < end) {
        instance_.WaitAny(wait_infos.size(), wait_infos.data(),
                          std::chrono::nanoseconds(5s).count());

        all_futures_complete = true;
        for (auto& info : wait_infos) {
            all_futures_complete = all_futures_complete && info.completed;
        }

        if (all_futures_complete) {
            break;
        }
    }

    if (errors.length() > 0) {
        return errors;
    }

    if (!all_futures_complete) {
        return "timeout exceeded";
    }

    return std::nullopt;
}

std::expected<wgpu::ShaderModule, std::string> GPU::loadShaderModule(
    const std::filesystem::path& path, std::string_view label) {
    const auto full_path = path_prefix_ / path;

    spdlog::info("[GPU] loading shader source {} at {}", label,
                 full_path.string());

    std::ifstream file(full_path);
    if (!file.is_open()) {
        return std::unexpected("could not open file");
    }

    std::stringstream sst;
    sst << file.rdbuf();
    std::string shader_source = sst.str();

    if (auto err = checkShaderModule(shader_source)) {
        return std::unexpected(
            std::format("cheking shader source: {}", err.value()));
    }

    wgpu::ShaderSourceWGSL source;
    source.code = {shader_source};

    wgpu::ShaderModuleDescriptor descriptor;
    descriptor.label = label;
    descriptor.nextInChain = &source;

    wgpu::ShaderModule module;
    auto awaiter = getAwaiter();
    awaiter.addCall([&]() { module = device_.CreateShaderModule(&descriptor); },
                    "Creating shader module");
    awaiter.addCall(
        [&]() -> wgpu::Future {
            return module.GetCompilationInfo(
                wgpu::CallbackMode::WaitAnyOnly,
                [&](wgpu::CompilationInfoRequestStatus status,
                    const wgpu::CompilationInfo* info) {
                    (void)status;

                    if (!info) {
                        return;
                    }

                    const auto messages
                        = std::span(info->messages, info->messageCount);

                    const auto error_messages
                        = messages
                          | std::views::filter(
                              [](const wgpu::CompilationMessage& msg) {
                                  return msg.type
                                         != wgpu::CompilationMessageType::Info;
                              })
                          | std::views::transform(
                              [](const wgpu::CompilationMessage& msg) {
                                  return std::string(msg.message);
                              })
                          | std::ranges::to<std::vector<std::string>>();

#ifndef NDEBUG
                    const auto all_messages
                        = messages
                          | std::views::transform(
                              [](const wgpu::CompilationMessage& msg) {
                                  return std::string(msg.message);
                              })
                          | std::views::join_with(std::string(", "))
                          | std::ranges::to<std::string>();

                    spdlog::debug(
                        "[GPU] Compiled shader module. Messages: {}",
                        all_messages.length() == 0 ? "<empty>" : all_messages);
#endif

                    auto joined = error_messages
                                  | std::views::join_with(std::string(", "))
                                  | std::ranges::to<std::string>();

                    auto* joined_and_never_released = new std::string{joined};

                    if (joined.length() > 0) {
                        device_.InjectError(
                            wgpu::ErrorType::Validation,
                            wgpu::StringView(*joined_and_never_released));
                    }
                });
        },
        "Shader compilation info");

    if (auto err = awaiter.executeAll()) {
        return std::unexpected(
            std::format("creating shader module: {}", err.value()));
    }

    return module;
}

std::optional<std::string> GPU::checkShaderModule(std::string_view module) {
    if (module.length() == 0) {
        return "module is empty";
    }
    return std::nullopt;
}

namespace {
[[nodiscard]] std::vector<std::byte> FloatBwTextureToRGB(
    std::span<std::byte> data) {
    assert(data.size() % sizeof(float) == 0
           && "Data size must be a multiple of sizeof(float)");

    auto floats
        = std::span<const float>(reinterpret_cast<const float*>(data.data()),
                                 data.size() / sizeof(float));

    std::vector<std::byte> result;
    result.reserve(floats.size() * 3);

    for (const float val : floats) {
        const auto channel = static_cast<std::byte>(
            static_cast<uint8_t>(std::clamp(val, 0.0F, 1.0F) * 255.0F));
        result.emplace_back(channel);
        result.emplace_back(channel);
        result.emplace_back(channel);
    }

    return result;
}
};  // namespace

std::expected<TextureData, std::string> GPU::readTexture(
    const wgpu::Texture& texture, uint32_t bytes_per_pixel, bool is_bw) {
    assert(is_bw && "colored is not supported");
    assert(texture.GetFormat() == wgpu::TextureFormat::R32Float
           && "r32float is the only supported format");

    const auto width = texture.GetWidth();
    const auto height = texture.GetHeight();

    constexpr uint32_t kBytesPerRowAlignment = 256;
    const uint32_t unpadded_bytes_per_row = width * bytes_per_pixel;
    const uint32_t padded_bytes_per_row
        = (unpadded_bytes_per_row + kBytesPerRowAlignment - 1)
          & ~(kBytesPerRowAlignment - 1);

    const auto queue = device_.GetQueue();
    const auto encoder = device_.CreateCommandEncoder();
    const wgpu::TexelCopyTextureInfo texture_src{
        .texture = texture,
        .aspect = wgpu::TextureAspect::All,
    };
    const wgpu::TexelCopyBufferInfo buffer_dst{
        .layout = wgpu::TexelCopyBufferLayout{
            .offset = 0,
            .bytesPerRow = padded_bytes_per_row,   // <-- padded
            .rowsPerImage = height,
        },
        .buffer = output_buf_,
    };
    const wgpu::Extent3D extent{
        .width = width,
        .height = height,
        .depthOrArrayLayers = 1,
    };
    encoder.CopyTextureToBuffer(&texture_src, &buffer_dst, &extent);

    const auto commands = encoder.Finish();
    queue.Submit(1, &commands);

    auto wait_copy = [&]() -> wgpu::Future {
        return queue.OnSubmittedWorkDone(
            wgpu::CallbackMode::WaitAnyOnly,
            [](wgpu::QueueWorkDoneStatus status, wgpu::StringView msg) {
                spdlog::debug(LOG_ID
                              " Finished copying texture to output buffer. "
                              "status:{} msg:'{}'",
                              static_cast<int>(status), std::string(msg));
            });
    };

    const auto err = getAwaiter()
                         .addCall(std::move(wait_copy),
                                  "Copy texture to output buffer", false)
                         .executeAll();

    if (err) {
        return std::unexpected("copying to output buffer: " + err.value());
    }

    const auto padded_size = padded_bytes_per_row * height;
    auto output = readOutputbuffer(padded_size, 0);
    if (!output) {
        return std::unexpected("reading output buffer: " + output.error());
    }

    const std::vector<uint8_t> ehe
        = output.value() | std::views::take(10)
          | std::views::transform(
              [](const std::byte& val) { return static_cast<uint8_t>(val); })
          | std::ranges::to<std::vector<uint8_t>>();

    // Strip the row padding
    std::vector<std::byte> tight;

    tight.reserve(static_cast<size_t>(unpadded_bytes_per_row) * height);
    const auto& padded = output.value();

    for (uint32_t row = 0; row < height; ++row) {
        const auto* src
            = padded.data() + static_cast<size_t>(row) * padded_bytes_per_row;
        tight.insert(tight.end(), src, src + unpadded_bytes_per_row);
    }

    std::vector<std::byte> rgb = FloatBwTextureToRGB(tight);

    if (is_bw) {
        tight.reserve(tight.size() * 3);
    }

    return TextureData{
        .width = width,
        .height = height,
        .pixel_data = std::move(rgb),
    };
}

std::expected<std::vector<std::byte>, std::string> GPU::readOutputbuffer(
    size_t size, size_t offset) {
    struct UserData {
        size_t offset;
        size_t size;
        wgpu::Buffer& buf;
        wgpu::Device& device;
        std::expected<std::vector<std::byte>, std::string> result;
    };
    UserData user_data{.offset = offset,
                       .size = size,
                       .buf = output_buf_,
                       .device = device_,
                       .result = {}};

    auto wgpu_cb = [](wgpu::MapAsyncStatus status, wgpu::StringView msg,
                      UserData* user_data) {
        if (status != wgpu::MapAsyncStatus::Success) {
            spdlog::error(
                "[GPU] Map async unsucessful when reading output "
                "buffer. status: {}; msg: {}",
                static_cast<int>(status), std::string{msg});
            user_data->result = std::unexpected(std::string{msg});
            return;
        }

        const auto* output
            = static_cast<const std::byte*>(user_data->buf.GetConstMappedRange(
                user_data->offset, user_data->size));

        user_data->result->resize(user_data->size);
        std::copy_n(output, user_data->size, user_data->result->begin());
    };

    auto future_cb = [&, this]() -> wgpu::Future {
        return output_buf_.MapAsync(wgpu::MapMode::Read, offset, size,
                                    wgpu::CallbackMode::WaitAnyOnly, wgpu_cb,
                                    &user_data);
    };

    auto awaiter = getAwaiter();
    awaiter.addCall(std::move(future_cb), "Reading output buffer");
    const auto err = awaiter.executeAll();

    output_buf_.Unmap();

    if (err) {
        return std::unexpected("gpu callback: " + err.value());
    }

    return std::move(user_data.result);
}

std::optional<std::string> GPU::initBufferRegions() {
    buf_regions_[static_cast<size_t>(BufferType::StorageA)] = {{
        .is_free = true,
        .offset = 0,
        .size = storage_buf_a_.GetSize(),
    }};

    buf_regions_[static_cast<size_t>(BufferType::StorageB)] = {{
        .is_free = true,
        .offset = 0,
        .size = storage_buf_b_.GetSize(),
    }};

    buf_regions_[static_cast<size_t>(BufferType::Input)] = {{
        .is_free = true,
        .offset = 0,
        .size = input_buf_.GetSize(),
    }};

    buf_regions_[static_cast<size_t>(BufferType::Output)] = {{
        .is_free = true,
        .offset = 0,
        .size = output_buf_.GetSize(),
    }};

    buf_regions_[static_cast<size_t>(BufferType::Uniform)] = {{
        .is_free = true,
        .offset = 0,
        .size = uniform_buf_.GetSize(),
    }};

    buf_regions_[static_cast<size_t>(BufferType::SharedStorage)] = {{
        .is_free = true,
        .offset = 0,
        .size = shared_buf_.GetSize(),
    }};

    for (auto& region : buf_regions_) {
        std::ranges::sort(region, std::less{}, &BufferRegion::offset);
        assert(region.size() > 0);
    }

    return std::nullopt;
}

void GPU::unAssignBuffer(BufferBinding& binding) {
    spdlog::debug(
        LOG_ID " unassigning buffer: [Binding offset:{}, size:{}, type:{}",
        binding.getOffset(), binding.getSize(), binding.getBuffertype());

    const wgpu::Buffer& buffer = getBuffer(binding.getBuffertype());
    if (buffer == nullptr) {
        return;
    }

    std::vector<BufferRegion>& regions
        = buf_regions_[static_cast<size_t>(binding.getBuffertype())];

    if (std::ranges::is_sorted(regions, std::less{}, &BufferRegion::size)) {
        std::ranges::sort(regions, std::less{}, &BufferRegion::offset);
    }

    auto reg = std::ranges::find(regions, binding.getOffset(),
                                 &BufferRegion::offset);

    if (reg == regions.end()) {
        spdlog::error(
            std::format(" could not find region (offset out of range) for "
                        "[BufferBinding offset:{} size:{}]",
                        binding.getOffset(), binding.getSize()));
        assert(false);
    }

    if (reg->is_free) {
        spdlog::error(std::format(
            LOG_ID " double free of memory at offset {} with size {}",
            binding.getOffset(), binding.getSize()));
        assert(false);
    }

    reg->is_free = true;

    // Merge with consecutive free regions forward
    auto next = reg + 1;
    while (next != regions.end() && next->is_free) {
        reg->size += next->size;
        // erase returns iterator to the element after erased
        next = regions.erase(next);
    }

    // Merge with preceding free region
    if (reg != regions.begin()) {
        auto prev = reg - 1;
        if (prev->is_free) {
            prev->size += reg->size;
            regions.erase(reg);
        }
    }
}

std::optional<std::string> GPU::fillInputBuffer(std::span<const std::byte> data,
                                                size_t offset) {
    if (data.size_bytes() >= input_buf_.GetSize()) {
        return std::format("requested size:{} bigger than available:{}",
                           data.size_bytes(), input_buf_.GetSize());
    }

    struct UserData {
        std::span<const std::byte> input;
        std::string error;
        wgpu::Buffer& buf;
    };

    UserData user_data{.input = data, .error = "", .buf = input_buf_};
    const size_t data_size = data.size_bytes() + offset;

    auto future = [&]() -> wgpu::Future {
        return input_buf_.MapAsync(
            wgpu::MapMode::Write, 0, data_size, wgpu::CallbackMode::WaitAnyOnly,
            [](wgpu::MapAsyncStatus status, wgpu::StringView msg,
               UserData* user_data) {
                if (status == wgpu::MapAsyncStatus::Success) {
                    auto* start = static_cast<std::byte*>(
                        user_data->buf.GetMappedRange());
                    std::ranges::copy(user_data->input, start);
                    user_data->buf.Unmap();
                    return;
                }

                std::string errmsg
                    = (msg.length == 0) ? "<empty msg>" : msg.data;

                user_data->error
                    = std::format("status: {}; message: {}",
                                  static_cast<int>(status), errmsg);
            },
            &user_data);
    };

    Awaiter awaiter{instance_, device_};

    awaiter.addCall(std::move(future), "Map and copy data to input buffer");

    if (auto err = awaiter.executeAll()) {
        return err;
    }

    return std::nullopt;
}

std::optional<std::string> GPU::fillNonInputBuffer(
    const wgpu::CommandEncoder& encoder, std::span<const std::byte> data,
    const BufferBinding& binding) const {
    assert(binding.getSize() >= data.size_bytes());

    encoder.WriteBuffer(getBuffer(binding.getBuffertype()), binding.getOffset(),
                        reinterpret_cast<const uint8_t*>(data.data()),
                        data.size_bytes());

    return std::nullopt;
}

std::expected<std::vector<std::byte>, std::string> GPU::readBuffer(
    const BufferBinding& binding) {
    if (binding.getBuffertype() == BufferType::Output) {
        return readOutputbuffer(binding.getSize(), binding.getOffset());
    }

    if (auto err = copyDataToOutput(binding)) {
        return std::unexpected("copying to output: " + err.value());
    }

    return readOutputbuffer(binding.getSize(), binding.getOffset());
}

std::optional<std::string> GPU::copyDataToOutput(const BufferBinding& binding) {
    if (binding.getBuffertype() == BufferType::Output) {
        return std::nullopt;
    }

    const auto& buf = getBuffer(binding.getBuffertype());

    if (binding.getSize() >= WCOMPUTE_MAP_READ_BUF_SIZE) {
        throw std::out_of_range("reading large buffers is not implemented");
    }

    const auto queue = device_.GetQueue();
    const auto encoder = device_.CreateCommandEncoder();

    encoder.CopyBufferToBuffer(buf, binding.getOffset(), output_buf_, 0,
                               binding.getSize());

    const auto commands = encoder.Finish();
    queue.Submit(1, &commands);

    auto wait = [&]() -> wgpu::Future {
        return queue.OnSubmittedWorkDone(
            wgpu::CallbackMode::WaitAnyOnly,
            [](wgpu::QueueWorkDoneStatus status, wgpu::StringView msg) {
                spdlog::debug(
                    LOG_ID
                    " finished copying to output buffer. status: {}, msg: '{}'",
                    static_cast<int>(status), std::string(msg));
            });
    };

    auto err = getAwaiter()
                   .addCall(std::move(wait),
                            std::format("copy {} to output buf", binding))
                   .executeAll();
    if (err) {
        return "wgpu: " + err.value();
    }

    return std::nullopt;
}
