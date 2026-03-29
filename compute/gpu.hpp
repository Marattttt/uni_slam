#pragma once

#include <spdlog/spdlog.h>
#include <webgpu/webgpu_cpp.h>

#include <expected>
#include <filesystem>
#include <memory>
#include <optional>
#include <string_view>

#include "awaiter.hpp"

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
constexpr std::string_view to_string(BufferType t);

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

class GPU {
   public:
    using DeviceCb = std::function<wgpu::DeviceLostCallback<void>>;
    using ErrorCb = std::function<wgpu::UncapturedErrorCallback<void>>;

    using DeviceCbPtr = std::shared_ptr<DeviceCb>;
    using ErrorCbPtr = std::shared_ptr<ErrorCb>;

    GPU(DeviceCbPtr deviceLostCallback, ErrorCbPtr errorCallback,
        std::filesystem::path prefix)
        : path_prefix_(std::move(prefix)),
          device_lost_callback_(std::move(deviceLostCallback)),
          uncaptured_error_callback_(std::move(errorCallback)) {}

    [[nodiscard]] virtual std::expected<wgpu::ShaderModule, std::string>
    loadShaderModule(const std::filesystem::path& path, std::string_view label);

    std::optional<std::string> initialize();
    [[nodiscard]] const wgpu::Device& getDevice() const;
    [[nodiscard]] const wgpu::Instance& getInstance() const;

    struct BgBinding {
        BufferType buf_type;
        wgpu::BindGroupEntry& bg_entry;
    };
    using BindingKey = std::string_view;
    using BindGroupBindings = std::unordered_map<BindingKey, BgBinding>;
    using BufferBindingMap = std::unordered_map<BindingKey, BufferBinding>;

    // Bind group is valid only during the lifetime of its binding as space will
    // be available for reuse after its lifetime ends
    [[nodiscard]] std::expected<BufferBindingMap, std::string>
    assignBuffersAndOffsets(BindGroupBindings bindings);

    [[nodiscard]] const wgpu::Buffer& getBuffer(BufferType buftype) const;
    [[nodiscard]] Awaiter getAwaiter() const;
    [[nodiscard]] std::expected<std::vector<uint8_t>, std::string>
    readOutputbuffer(size_t size, size_t offset);

    [[nodiscard]] std::optional<std::string> fillInputBuffer(
        std::span<const std::byte> data, size_t offset = 0);

    [[nodiscard]] std::optional<std::string> fillNonInputBuffer(
        const wgpu::CommandEncoder& encoder, std::span<const std::byte> data,
        const BufferBinding& binding) const;

   private:
    [[nodiscard]] std::optional<std::string> initInstance();
    [[nodiscard]] std::optional<std::string> initDevice();
    [[nodiscard]] std::optional<std::string> initAdapter();
    [[nodiscard]] std::optional<std::string> initBuffers();
    [[nodiscard]] std::optional<std::string> initBufferRegions();

    [[nodiscard]] static std::optional<std::string> checkShaderModule(
        std::string_view module);
    [[nodiscard]] std::expected<BufferBinding, std::string> assignBufferRegion(
        wgpu::BindGroupEntry& entry, BufferType buftype);

    // Allow buffer binding to call this in its destructor
    friend BufferBinding::~BufferBinding() noexcept;
    // Release the binding created by assignBuffer()
    //
    // May throw exceptions
    void unAssignBuffer(BufferBinding& binding);

    const std::filesystem::path path_prefix_;

    wgpu::Instance instance_;
    wgpu::Device device_;
    wgpu::Adapter adapter_;
    const DeviceCbPtr device_lost_callback_;
    const ErrorCbPtr uncaptured_error_callback_;

    struct BufferRegion {
        bool is_free;
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
