#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include "vc/core/shapes/Plane.hpp"
#include "vc/core/util/Iteration.hpp"
#include "vc/texturing/PPMGenerator.hpp"
#include "vc/texturing/TextureStackTexture.hpp"

namespace vc = volcart;
namespace vct = volcart::texturing;

namespace
{
// Build a Plane mesh with a UV map that spans [0, 1] x [0, 1] linearly
auto BuildPlaneWithUVMap(std::size_t gridDim) -> std::pair<vc::ITKMesh::Pointer, vc::UVMap::Pointer>
{
    vc::shapes::Plane plane(static_cast<int>(gridDim), static_cast<int>(gridDim));
    auto mesh = plane.itkMesh();
    auto uvMap = vc::UVMap::New();
    std::size_t id{0};
    for (const auto uv : vc::range2D(gridDim, gridDim)) {
        auto u = static_cast<double>(uv.first) / static_cast<double>(gridDim - 1);
        auto v = static_cast<double>(uv.second) / static_cast<double>(gridDim - 1);
        uvMap->set(id++, {u, v});
    }
    return {mesh, uvMap};
}

// Derive per-face-corner UVs from a (seamless) UV map, matching what
// OBJReader::getFaceUVs() would produce for a mesh with no seams.
auto DeriveFaceUVs(const vc::ITKMesh::Pointer& mesh, const vc::UVMap::Pointer& uvMap)
    -> std::vector<std::array<cv::Vec2d, 3>>
{
    std::vector<std::array<cv::Vec2d, 3>> faceUVs;
    faceUVs.reserve(mesh->GetNumberOfCells());
    for (auto cellIt = mesh->GetCells()->Begin();
         cellIt != mesh->GetCells()->End(); ++cellIt) {
        const auto& ids = cellIt->Value()->GetPointIdsContainer();
        faceUVs.push_back(
            {uvMap->get(ids.GetElement(0)), uvMap->get(ids.GetElement(1)),
             uvMap->get(ids.GetElement(2))});
    }
    return faceUVs;
}

// Build a PPM whose texture coordinate map is analytically known: since the
// same UV map is used for both flattening and the source texture UV, the
// capture UV at output pixel (y, x) is (x / (dim - 1), y / (dim - 1)).
auto BuildPPMWithKnownTextureCoords(std::size_t dim) -> vc::PerPixelMap::Pointer
{
    auto [mesh, uvMap] = BuildPlaneWithUVMap(5);

    vct::PPMGenerator gen;
    gen.setDimensions(dim, dim);
    gen.setMesh(mesh);
    gen.setUVMap(uvMap);
    gen.setTextureFaceUVs(DeriveFaceUVs(mesh, uvMap));
    return gen.compute();
}

// A single-channel image that is an exact linear ramp along its columns:
// pixel value at column c is round(255 * c / (cols - 1)). Bilinear sampling
// of an exactly-linear ramp reconstructs the underlying linear function
// exactly, regardless of fractional sample position.
auto MakeGrayRamp(int size) -> cv::Mat
{
    cv::Mat img(size, size, CV_8UC1);
    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            auto v = static_cast<uchar>(
                std::round(255.0 * x / static_cast<double>(size - 1)));
            img.at<uchar>(y, x) = v;
        }
    }
    return img;
}

// A 3-channel image: R ramps with x, G ramps with y, B is constant
auto MakeRGBRamp(int size) -> cv::Mat
{
    cv::Mat img(size, size, CV_8UC3);
    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            auto r = static_cast<uchar>(
                std::round(255.0 * x / static_cast<double>(size - 1)));
            auto g = static_cast<uchar>(
                std::round(255.0 * y / static_cast<double>(size - 1)));
            img.at<cv::Vec3b>(y, x) = {r, g, 128};
        }
    }
    return img;
}
}  // namespace

TEST(TextureStackTexture, ThrowsWithoutPerPixelMap)
{
    vct::TextureStackTexture tex;
    auto stack = vc::TextureStack::New();
    stack->addChannel("gray", ::MakeGrayRamp(10));
    tex.setTextureStack(stack);
    EXPECT_THROW(tex.compute(), std::runtime_error);
}

TEST(TextureStackTexture, ThrowsWithoutTextureStack)
{
    auto ppm = ::BuildPPMWithKnownTextureCoords(20);

    vct::TextureStackTexture tex;
    tex.setPerPixelMap(ppm);
    EXPECT_THROW(tex.compute(), std::runtime_error);
}

TEST(TextureStackTexture, ThrowsWithoutTextureCoordMap)
{
    // Build a PPM without a texture UV map
    auto [mesh, uvMap] = ::BuildPlaneWithUVMap(5);
    vct::PPMGenerator gen;
    gen.setDimensions(20, 20);
    gen.setMesh(mesh);
    gen.setUVMap(uvMap);
    auto ppm = gen.compute();

    vct::TextureStackTexture tex;
    tex.setPerPixelMap(ppm);
    auto stack = vc::TextureStack::New();
    stack->addChannel("gray", ::MakeGrayRamp(10));
    tex.setTextureStack(stack);

    EXPECT_THROW(tex.compute(), std::runtime_error);
}

TEST(TextureStackTexture, ResampleMatchesAnalyticExpectation)
{
    constexpr std::size_t dim{20};
    auto ppm = ::BuildPPMWithKnownTextureCoords(dim);

    auto stack = vc::TextureStack::New();
    stack->addChannel("gray", ::MakeGrayRamp(50));
    stack->addChannel("rgb", ::MakeRGBRamp(37));

    vct::TextureStackTexture tex;
    tex.setPerPixelMap(ppm);
    tex.setTextureStack(stack);
    auto result = tex.compute();

    ASSERT_EQ(result.size(), 2);
    EXPECT_EQ(result[0].channels(), 1);
    EXPECT_EQ(result[1].channels(), 3);
    EXPECT_EQ(result[0].size(), cv::Size(static_cast<int>(dim), static_cast<int>(dim)));
    EXPECT_EQ(result[1].size(), cv::Size(static_cast<int>(dim), static_cast<int>(dim)));

    for (const auto [y, x] : vc::range2D(dim, dim)) {
        if (not ppm->hasMapping(y, x)) {
            continue;
        }
        auto u = static_cast<double>(x) / static_cast<double>(dim - 1);
        auto v = static_cast<double>(y) / static_cast<double>(dim - 1);

        auto gray = result[0].at<double>(static_cast<int>(y), static_cast<int>(x));
        EXPECT_NEAR(gray, 255.0 * u, 0.5);

        auto rgb = result[1].at<cv::Vec3d>(static_cast<int>(y), static_cast<int>(x));
        EXPECT_NEAR(rgb[0], 255.0 * u, 0.5);
        EXPECT_NEAR(rgb[1], 255.0 * v, 0.5);
        EXPECT_NEAR(rgb[2], 128.0, 0.5);
    }
}

