#pragma once

/** @file */

#include "vc/core/types/TextureStack.hpp"
#include "vc/texturing/TexturingAlgorithm.hpp"

namespace volcart::texturing
{

/**
 * @brief Resample a TextureStack's channels into a PerPixelMap's UV space
 *
 * For each pixel in the PerPixelMap that has a valid mapping, bilinearly
 * samples every channel of the given TextureStack at that pixel's texture
 * coordinate (PerPixelMap::textureCoordMap()), producing one output image per
 * input channel. Each output image has the same pixel dimensions as the
 * PerPixelMap and the same number of components as its corresponding source
 * channel (e.g. an RGB source channel produces a 3-component output image).
 *
 * Unlike CompositeTexture and its siblings, this algorithm does not consult a
 * Volume; it instead requires a PerPixelMap that was generated with an
 * associated source texture UVs (see PPMGenerator::setTextureFaceUVs()). It
 * is intended for texturing meshes from 2D capture images (e.g. RGB, IR,
 * UV-fluorescence photography) rather than a 3D CT volume.
 *
 * @ingroup Texture
 */
class TextureStackTexture : public TexturingAlgorithm
{
public:
    /** Pointer type */
    using Pointer = std::shared_ptr<TextureStackTexture>;

    /** Make shared pointer */
    static auto New() -> Pointer;

    /** Default constructor */
    TextureStackTexture() = default;
    /** Default destructor */
    ~TextureStackTexture() override = default;
    /** Default copy constructor */
    TextureStackTexture(TextureStackTexture&) = default;
    /** Default move constructor */
    TextureStackTexture(TextureStackTexture&&) = default;
    /** Default copy operator */
    auto operator=(const TextureStackTexture&)
        -> TextureStackTexture& = default;
    /** Default move operator */
    auto operator=(TextureStackTexture&&) -> TextureStackTexture& = default;

    /** @brief Set the source TextureStack */
    void setTextureStack(TextureStack::Pointer stack);

    /**
     * @brief Compute the Texture
     *
     * Returns one output image per channel in the TextureStack, in the order
     * the channels were added.
     *
     * @throws std::runtime_error if no PerPixelMap or TextureStack has been
     * set, or if the PerPixelMap has no texture coordinate map
     */
    auto compute() -> Texture override;

private:
    /** Source texture stack */
    TextureStack::Pointer stack_;
};

}  // namespace volcart::texturing
