#include <gtest/gtest.h>
#include <onnxruntime_cxx_api.h>

#include <cctype>
#include <string>
#include <string_view>
#include <vector>

namespace {

// Pinned to the tag vendored under vendor/onnxruntime. A mismatch here means
// the install prefix build_onnxruntime.sh produced is stale relative to the
// subtree.
constexpr std::string_view kExpectedVersion = "1.29.1";

bool containsIgnoringCase(std::string_view haystack, std::string_view needle) {
    const auto lower = [](std::string_view text) {
        std::string lowered;
        lowered.reserve(text.size());
        for (char chr : text) {
            lowered.push_back(static_cast<char>(
                std::tolower(static_cast<unsigned char>(chr))));
        }
        return lowered;
    };

    return lower(haystack).contains(lower(needle));
}

std::string joinProviders(const std::vector<std::string>& providers) {
    if (providers.empty()) {
        return "<none>";
    }

    std::string joined = providers.front();
    for (size_t i = 1; i < providers.size(); i++) {
        joined += ", ";
        joined += providers[i];
    }
    return joined;
}

}  // namespace

// The install these tests exercise is produced by build_onnxruntime.sh, which
// is also where the Dawn the compute library links comes from. Together they
// assert that ONNX Runtime and its WebGPU execution provider survived that
// build, that the headers and the imported target resolve, and that
// libonnxruntime.so is found at run time.
TEST(OnnxRuntimeInstallTest, ReportsTheVendoredVersion) {
    EXPECT_EQ(std::string_view(OrtGetApiBase()->GetVersionString()),
              kExpectedVersion);
}

TEST(OnnxRuntimeInstallTest, ProvidesTheWebGpuExecutionProvider) {
    const std::vector<std::string> providers = Ort::GetAvailableProviders();

    bool has_webgpu = false;
    for (const std::string& provider : providers) {
        if (containsIgnoringCase(provider, "webgpu")) {
            has_webgpu = true;
            break;
        }
    }

    EXPECT_TRUE(has_webgpu)
        << "the WebGPU execution provider is missing; ONNX Runtime was built "
           "without onnxruntime_USE_WEBGPU=ON. Available providers: "
        << joinProviders(providers);
}
