#pragma once

#include <spdlog/spdlog.h>
#include <webgpu/webgpu_cpp.h>

#include <expected>
#include <filesystem>
#include <memory>
#include <optional>
#include <string_view>

#include "compute/awaiter.hpp"

namespace wslam {
namespace compute {

#ifndef WCOMPUTE_STORAGE_BUF_SIZE
#define WCOMPUTE_STORAGE_BUF_SIZE (500 * (1ULL << 20))  // 0.5 gigabytes
#endif

#ifndef WCOMPUTE_SHARED_BUF_SIZE
#define WCOMPUTE_SHARED_BUF_SIZE (64 * (1ULL << 20))  // 64 megabytes
#endif

#ifndef WCOMPUTE_UNIFORM_BUF_SIZE
#define WCOMPUTE_UNIFORM_BUF_SIZE (32 * (1ULL << 10))  // 32 kilobytes
#endif

#ifndef WCOMPUTE_MAP_READ_BUF_SIZE
#define WCOMPUTE_MAP_READ_BUF_SIZE (64 * (1ULL << 20))  // 64 megabytes
#endif

#ifndef WCOMPUTE_MAP_WRITE_BUF_SIZE
#define WCOMPUTE_MAP_WRITE_BUF_SIZE (64 * (1ULL << 20))  // 64 megabytes
#endif

enum class BufferType : uint8_t {
    Input,
    Uniform,
    Output,
    StorageA,
    StorageB,
    SharedStorage,
    COUNT
};

wgpu::BufferBindingType BufferTypeToWgpuBindingType(BufferType bt);

// Stores RGB data of an image
struct TextureData {
    uint32_t width;
    uint32_t height;
    std::vector<std::byte> pixel_data_rgb;
};

class GPU;

// Created by the GPU class as a binding to a buffer region
class BufferBinding {
   public:
    // Can only be created in a method of GPU class
    struct Token {
       private:
        Token() = default;
        friend class GPU;
    };

    struct Params {
        GPU& gpu;
        BufferType buf_type;
        size_t size;
        size_t offset;
    };

    BufferBinding(Token token, Params&& params)
        : gpu_(&params.gpu),
          buf_type_(params.buf_type),
          size_(params.size),
          offset_(params.offset) {
        (void)token;
    }

    BufferBinding(BufferBinding&&) noexcept;
    BufferBinding& operator=(BufferBinding&&) noexcept;
    ~BufferBinding() noexcept;

    BufferBinding() = delete;
    BufferBinding(const BufferBinding&) = delete;
    BufferBinding& operator=(const BufferBinding&) = delete;

    void swap(BufferBinding& buf2) noexcept;

    [[nodiscard]] constexpr size_t getSize() const { return size_; }
    [[nodiscard]] constexpr size_t getOffset() const { return offset_; }
    [[nodiscard]] constexpr BufferType getBuffertype() const {
        return buf_type_;
    }

   private:
    GPU* gpu_;
    BufferType buf_type_;
    size_t size_, offset_;
};

// Runtime-configurable sizes for the fixed GPU slabs allocated in
// GPU::initialize(). Defaults reproduce the sizes historically hard-coded via
// the WCOMPUTE_* macros. NOTE: the SharedStorage slab defaults to the storage
// size (500 MB) to match the pre-Opts behaviour — that buffer was always
// allocated with WCOMPUTE_STORAGE_BUF_SIZE, never the (unused)
// WCOMPUTE_SHARED_BUF_SIZE (64 MB) macro. Shrink these in tests to allocate a
// tiny device. Referred to as GPU::Opts via the alias inside the class.
struct GpuBufferSizes {
    size_t input_size = WCOMPUTE_MAP_WRITE_BUF_SIZE;   // 64 MB
    size_t uniform_size = WCOMPUTE_UNIFORM_BUF_SIZE;   // 32 KB
    size_t storage_size = WCOMPUTE_STORAGE_BUF_SIZE;   // 500 MB
    size_t shared_size = WCOMPUTE_STORAGE_BUF_SIZE;    // 500 MB (see note)
    size_t output_size = WCOMPUTE_MAP_READ_BUF_SIZE;   // 64 MB
};

class GPU {
   public:
    using DeviceCb = std::function<wgpu::DeviceLostCallback<void>>;
    using ErrorCb = std::function<wgpu::UncapturedErrorCallback<void>>;

    using DeviceCbPtr = std::shared_ptr<DeviceCb>;
    using ErrorCbPtr = std::shared_ptr<ErrorCb>;

    using Opts = GpuBufferSizes;

    GPU(DeviceCbPtr deviceLostCallback, ErrorCbPtr errorCallback,
        std::filesystem::path prefix, Opts opts = {})
        : path_prefix_(std::move(prefix)),
          opts_(opts),
          device_lost_callback_(std::move(deviceLostCallback)),
          uncaptured_error_callback_(std::move(errorCallback)) {}

    using ShaderOverride = std::pair<std::string, std::string>;
    [[nodiscard]] virtual std::expected<wgpu::ShaderModule, std::string>
    loadShaderModule(const std::filesystem::path& path, std::string_view label,
                     std::optional<std::span<const ShaderOverride>> overrides
                     = std::nullopt);

    [[nodiscard]] std::optional<std::string> initialize();
    [[nodiscard]] const wgpu::Device& getDevice() const;
    [[nodiscard]] const wgpu::Instance& getInstance() const;
    [[nodiscard]] const wgpu::Limits& getLimits() const;

    struct BgBinding {
        BufferType buf_type;
        wgpu::BindGroupEntry& bg_entry;
        bool is_retained = false;
    };
    using BindingKey = std::string;
    using BindGroupBindings = std::unordered_map<BindingKey, BgBinding>;
    using BufferBindingMap = std::unordered_map<BindingKey, BufferBinding>;

    // Bind group is valid only during the lifetime of its binding as space will
    // be available for reuse after its lifetime ends
    [[nodiscard]] std::expected<BufferBindingMap, std::string>
    assignBuffersAndOffsets(BindGroupBindings&& bindings);

    std::optional<std::string> clearBuffersAndOffsets();

    [[nodiscard]] const wgpu::Buffer& getBuffer(BufferType buftype) const;
    [[nodiscard]] Awaiter getAwaiter() const;

    // Submits the command buffer, registers OnSubmittedWorkDone, blocks
    // until done or timeout exceeded. Prefer this over hand-rolling the
    // submit-then-wait dance (avoids the queue-serial vs WaitListEvent
    // mixing trap — see awaiter.hpp).
    [[nodiscard]] std::optional<std::string> submitAndWait(
        const wgpu::CommandBuffer& commands, std::string label,
        std::chrono::nanoseconds timeout
        = std::chrono::seconds(kDefaultTimeoutSeconds));

    [[nodiscard]] std::expected<std::vector<std::byte>, std::string>
    readOutputbuffer(size_t size, size_t offset);
    [[nodiscard]] std::expected<std::vector<std::byte>, std::string> readBuffer(
        const BufferBinding& binding);
    [[nodiscard]] std::expected<TextureData, std::string> readTexture(
        const wgpu::Texture& texture, uint32_t bytes_per_pixel, bool is_bw);

    [[nodiscard]] std::optional<std::string> fillInputBuffer(
        std::span<const std::byte> data, size_t offset = 0);
    [[nodiscard]] std::optional<std::string> fillNonInputBuffer(
        const wgpu::CommandEncoder& encoder, std::span<const std::byte> data,
        const BufferBinding& binding) const;

   private:
    [[nodiscard]] std::optional<std::string> initInstance();
    [[nodiscard]] std::optional<std::string> initDevice();
    [[nodiscard]] std::optional<std::string> initAdapter();
    std::optional<std::string> initLimits();
    [[nodiscard]] std::optional<std::string> initBuffers();
    [[nodiscard]] std::optional<std::string> initBufferRegions();

    [[nodiscard]] size_t getMinBufferAlignment(BufferType buftype) const;
    [[nodiscard]] static std::optional<std::string> checkShaderModule(
        std::string_view module);
    [[nodiscard]] std::expected<BufferBinding, std::string> assignBufferRegion(
        wgpu::BindGroupEntry& entry, BufferType buftype, bool is_retained);

    [[nodiscard]] std::optional<std::string> copyDataToOutput(
        const BufferBinding& binding);

    // Allow buffer binding to call this in its destructor
    friend BufferBinding::~BufferBinding() noexcept;
    // Release the binding created by assignBuffer()
    //
    // May throw exceptions
    void unAssignBuffer(BufferBinding& binding);

    const std::filesystem::path path_prefix_;
    const Opts opts_;

    wgpu::Instance instance_;
    wgpu::Device device_;
    wgpu::Adapter adapter_;
    wgpu::Limits limits_;
    const DeviceCbPtr device_lost_callback_;
    const ErrorCbPtr uncaptured_error_callback_;

    struct BufferRegion {
        bool is_free;
        bool is_retained;
        size_t offset;
        size_t size;
    };

    std::array<std::vector<BufferRegion>, static_cast<int>(BufferType::COUNT)>
        buf_regions_;
    wgpu::Buffer storage_buf_a_;
    wgpu::Buffer storage_buf_b_;
    wgpu::Buffer shared_buf_;
    wgpu::Buffer uniform_buf_;
    wgpu::Buffer input_buf_;
    wgpu::Buffer output_buf_;
};
};  // namespace compute
};  // namespace wslam

template <>
struct std::formatter<wslam::compute::BufferType> {
    using BufferType = wslam::compute::BufferType;

    static constexpr auto parse(std::format_parse_context& ctx) {
        return ctx.begin();
    }

    static constexpr auto format(const BufferType& buf_type,
                                 std::format_context& ctx) {
        const char* str = nullptr;

        switch (buf_type) {
            case BufferType::Input:
                str = "Input";
                break;
            case BufferType::Uniform:
                str = "Uniform";
                break;
            case BufferType::Output:
                str = "Output";
                break;
            case BufferType::StorageA:
                str = "Storage A";
                break;
            case BufferType::StorageB:
                str = "Storage B";
                break;
            case BufferType::SharedStorage:
                str = "Shared storage";
                break;
            case BufferType::COUNT:
                assert(false && "invalid buffer type COUNT");
        }

        return std::format_to(ctx.out(), "{}", str);
    }
};

template <>
struct std::formatter<wslam::compute::GPU::BgBinding> {
    static constexpr auto parse(std::format_parse_context& ctx) {
        return ctx.begin();
    }

    static constexpr auto format(const wslam::compute::GPU::BgBinding& b,
                                 std::format_context& ctx) {
        return std::format_to(
            ctx.out(),
            "{{BgBinding  buf_type:{}, binding:{}, offset:{}, size:{} }}",
            b.buf_type, b.bg_entry.binding, b.bg_entry.offset, b.bg_entry.size);
    }
};

template <>
struct std::formatter<wslam::compute::BufferBinding> {
    static constexpr auto parse(std::format_parse_context& ctx) {
        return ctx.begin();
    }
    static constexpr auto format(const wslam::compute::BufferBinding& b,
                                 std::format_context& ctx) {
        return std::format_to(
            ctx.out(), "{{BufferBinding  buf_type: {}, offset: {}, size: {} }}",
            b.getBuffertype(), b.getOffset(), b.getSize());
    }
};
