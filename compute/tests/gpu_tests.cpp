#include <gtest/gtest.h>
#include <webgpu/webgpu_cpp.h>

#include <memory>
#include <optional>
#include <string>

#include "compute/compute.hpp"
#include "compute/gpu.hpp"

using namespace wslam::compute;

namespace {

// Slabs small enough for a quick device, CPU or not
constexpr GPU::Opts kSmallSlabs{
    .input_size = 1 << 20,
    .uniform_size = 1 << 10,
    .storage_size = 1 << 20,
    .shared_size = 1 << 20,
    .output_size = 1 << 20,
};

std::shared_ptr<GPU> makeGpu(ComputeDevice compute_device) {
    return std::make_shared<GPU>(
        std::make_shared<GPU::DeviceCb>(impl::print_device_unresponsive),
        std::make_shared<GPU::ErrorCb>(impl::print_device_captured_error), ".",
        compute_device, kSmallSlabs);
}

}  // namespace

// Needs Dawn built with SwiftShader, which build_onnxruntime.sh installs next
// to libwebgpu_dawn.so
TEST(GpuTest, CpuDeviceRunsOnTheCpuAdapter) {
    const auto gpu = makeGpu(ComputeDevice::Cpu);
    ASSERT_EQ(gpu->initialize(), std::nullopt);
    EXPECT_EQ(gpu->getComputeDevice(), ComputeDevice::Cpu);

    wgpu::AdapterInfo info;
    ASSERT_EQ(gpu->getDevice().GetAdapterInfo(&info), wgpu::Status::Success);
    EXPECT_EQ(info.adapterType, wgpu::AdapterType::CPU);
}

// Dawn searches SwiftShader before the system's drivers, so GPU compute must
// never end up on it
TEST(GpuTest, GpuDeviceRunsOnAHardwareAdapter) {
    const auto gpu = makeGpu(ComputeDevice::Gpu);
    ASSERT_EQ(gpu->initialize(), std::nullopt);

    wgpu::AdapterInfo info;
    ASSERT_EQ(gpu->getDevice().GetAdapterInfo(&info), wgpu::Status::Success);
    EXPECT_NE(info.adapterType, wgpu::AdapterType::CPU);
}
