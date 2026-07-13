#include "vc/core/types/TextureStack.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

using namespace volcart;

namespace
{
auto Clamp(int v, int lo, int hi) -> int { return std::min(std::max(v, lo), hi); }
}  // namespace

void TextureStack::addChannel(const std::string& name, const cv::Mat& image)
{
    if (image.empty()) {
        throw std::invalid_argument("Channel image is empty: " + name);
    }
    if (std::find(order_.begin(), order_.end(), name) != order_.end()) {
        throw std::invalid_argument("Channel name already in use: " + name);
    }

    cv::Mat converted;
    image.convertTo(converted, CV_MAKETYPE(CV_64F, image.channels()));

    order_.push_back(name);
    channels_.push_back(converted);
}

auto TextureStack::channelNames() const -> std::vector<std::string>
{
    return order_;
}

auto TextureStack::channel(const std::string& name) const -> cv::Mat
{
    auto it = std::find(order_.begin(), order_.end(), name);
    if (it == order_.end()) {
        throw std::invalid_argument("No channel named: " + name);
    }
    return channels_[static_cast<std::size_t>(it - order_.begin())];
}

auto TextureStack::numChannels() const -> std::size_t { return order_.size(); }

auto TextureStack::empty() const -> bool { return order_.empty(); }

auto TextureStack::totalSamples() const -> std::size_t
{
    std::size_t total{0};
    for (const auto& img : channels_) {
        total += static_cast<std::size_t>(img.channels());
    }
    return total;
}

auto TextureStack::sample(const cv::Vec2d& uv) const -> std::vector<double>
{
    if (channels_.empty()) {
        throw std::domain_error("TextureStack has no channels");
    }

    std::vector<double> out;
    out.reserve(totalSamples());

    for (const auto& img : channels_) {
        const auto cols = img.cols;
        const auto rows = img.rows;
        const auto nc = img.channels();

        // Map normalized UV to this image's pixel space
        const auto px = uv[0] * static_cast<double>(cols - 1);
        const auto py = uv[1] * static_cast<double>(rows - 1);

        const auto x0 = ::Clamp(static_cast<int>(std::floor(px)), 0, cols - 1);
        const auto x1 = ::Clamp(x0 + 1, 0, cols - 1);
        const auto y0 = ::Clamp(static_cast<int>(std::floor(py)), 0, rows - 1);
        const auto y1 = ::Clamp(y0 + 1, 0, rows - 1);

        const auto fx = std::min(std::max(px - x0, 0.0), 1.0);
        const auto fy = std::min(std::max(py - y0, 0.0), 1.0);

        const auto* row0 = img.ptr<double>(y0);
        const auto* row1 = img.ptr<double>(y1);
        for (int c = 0; c < nc; c++) {
            const auto v00 = row0[x0 * nc + c];
            const auto v10 = row0[x1 * nc + c];
            const auto v01 = row1[x0 * nc + c];
            const auto v11 = row1[x1 * nc + c];
            const auto v0 = v00 * (1.0 - fx) + v10 * fx;
            const auto v1 = v01 * (1.0 - fx) + v11 * fx;
            out.push_back(v0 * (1.0 - fy) + v1 * fy);
        }
    }

    return out;
}
