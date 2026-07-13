#pragma once

/** @file */

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace volcart
{

/**
 * @brief A stack of named, co-registered 2D texture channels
 *
 * A TextureStack holds one or more single- or multi-channel images (e.g. RGB,
 * IR, and UV-fluorescence captures of a surface) that all share the same
 * normalized UV parameterization: pixel `(x, y)` in each image corresponds to
 * normalized coordinate `(x / (width - 1), y / (height - 1))`, regardless of
 * that image's own pixel resolution. This is the same convention used by
 * volcart::texturing::PPMGenerator for its own UV raster, so a UV coordinate
 * produced by PPMGenerator (e.g. PerPixelMap::textureCoordMap()) can be
 * sampled directly against a TextureStack without additional conversion.
 *
 * @note TextureStack does not perform any Y-axis flip. If a texture image's
 * own coordinate convention differs from this (e.g. some OBJ/photogrammetry
 * exports place `v = 0` at the bottom of the image), the caller is
 * responsible for flipping the image or its UV map before use.
 *
 * This class is intended as the texture-based analog of volcart::Volume for
 * texturing algorithms that sample from 2D capture images (e.g. RGB, IR,
 * UV-fluorescence) rather than a 3D CT volume.
 *
 * @ingroup Types
 */
class TextureStack
{
public:
    /** Pointer type */
    using Pointer = std::shared_ptr<TextureStack>;

    /** Static New function */
    static auto New() -> Pointer { return std::make_shared<TextureStack>(); }

    /**
     * @brief Add a named channel image
     *
     * The image may have any pixel resolution and any number of components
     * (e.g. 3 for RGB, 1 for IR); its samples are appended, in the order
     * they're stored in the image, to the output of sample(). Channel names
     * must be unique.
     *
     * @throws std::invalid_argument if the image is empty or the name is
     * already in use
     */
    void addChannel(const std::string& name, const cv::Mat& image);

    /** @brief Get the ordered list of channel names */
    [[nodiscard]] auto channelNames() const -> std::vector<std::string>;

    /**
     * @brief Get a previously added channel image, converted to CV_64F
     *
     * @throws std::invalid_argument if no channel with this name was added
     */
    [[nodiscard]] auto channel(const std::string& name) const -> cv::Mat;

    /** @brief Get the number of channel images that have been added */
    [[nodiscard]] auto numChannels() const -> std::size_t;

    /** @brief Get whether any channel images have been added */
    [[nodiscard]] auto empty() const -> bool;

    /**
     * @brief Get the total number of scalar samples returned by sample()
     *
     * This is the sum of the component counts (e.g. 3 for an RGB image) of
     * all added channel images.
     */
    [[nodiscard]] auto totalSamples() const -> std::size_t;

    /**
     * @brief Bilinearly sample every channel at a normalized UV coordinate
     *
     * For each channel, in the order added, bilinearly interpolates that
     * channel's image at the pixel coordinate corresponding to `uv` and
     * appends its component values to the output. Coordinates outside
     * `[0, 1]` are clamped to the image edge.
     *
     * @throws std::domain_error if no channels have been added
     */
    [[nodiscard]] auto sample(const cv::Vec2d& uv) const -> std::vector<double>;

private:
    /** Ordered channel names */
    std::vector<std::string> order_;
    /** Channel images, stored as CV_64F with their original channel count */
    std::vector<cv::Mat> channels_;
};

}  // namespace volcart
