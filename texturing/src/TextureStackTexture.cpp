#include "vc/texturing/TextureStackTexture.hpp"

#include <stdexcept>

#include "vc/core/util/Iteration.hpp"

using namespace volcart;
using namespace volcart::texturing;

auto TextureStackTexture::New() -> TextureStackTexture::Pointer
{
    return std::make_shared<TextureStackTexture>();
}

void TextureStackTexture::setTextureStack(TextureStack::Pointer stack)
{
    stack_ = std::move(stack);
}

auto TextureStackTexture::compute() -> Texture
{
    if (not ppm_) {
        throw std::runtime_error("No PerPixelMap set");
    }
    if (not stack_ or stack_->empty()) {
        throw std::runtime_error("No TextureStack set");
    }

    auto texCoordMap = ppm_->textureCoordMap();
    if (texCoordMap.empty()) {
        throw std::runtime_error(
            "PerPixelMap has no texture coordinate map. It must be "
            "generated with PPMGenerator::setTextureFaceUVs().");
    }

    // Setup
    result_.clear();
    auto height = static_cast<int>(ppm_->height());
    auto width = static_cast<int>(ppm_->width());

    // Allocate one output image per channel, matching that channel's
    // original component count
    auto names = stack_->channelNames();
    Texture outputs;
    outputs.reserve(names.size());
    for (const auto& name : names) {
        auto nc = stack_->channel(name).channels();
        outputs.push_back(
            cv::Mat::zeros(height, width, CV_MAKETYPE(CV_64F, nc)));
    }

    // Get the mappings
    auto mappings = ppm_->getMappingCoords();

    // Iterate through the mappings
    progressStarted();
    for (const auto [idx, coord] : enumerate(mappings)) {
        progressUpdated(idx);

        const auto [y, x] = coord;
        const auto intY = static_cast<int>(y);
        const auto intX = static_cast<int>(x);

        const auto tc = texCoordMap.at<cv::Vec2f>(intY, intX);
        auto samples = stack_->sample({tc[0], tc[1]});

        std::size_t offset{0};
        for (std::size_t i = 0; i < names.size(); i++) {
            auto& img = outputs[i];
            auto nc = img.channels();
            auto* dst = img.ptr<double>(intY) + intX * nc;
            for (int c = 0; c < nc; c++) {
                dst[c] = samples[offset + static_cast<std::size_t>(c)];
            }
            offset += static_cast<std::size_t>(nc);
        }
    }
    progressComplete();

    result_ = outputs;
    return result_;
}
