#include "pass.hpp"

#include <optional>

using namespace wslam::compute;

BindingMap GPUPass::getOutputBindings() { return {}; }

std::optional<std::string> CustomPass::initialize() { return std::nullopt; }
std::optional<std::string> CustomPass::execute() { return callback_(this); }
std::string CustomPass::getId() const { return id_; }
