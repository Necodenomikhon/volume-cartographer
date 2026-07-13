#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "vc/core/shapes/Plane.hpp"
#include "vc/core/util/Iteration.hpp"
#include "vc/texturing/PPMGenerator.hpp"

namespace vc = volcart;
namespace vct = volcart::texturing;

class PPMGeneratorTest : public testing::TestWithParam<int>
{
};

TEST(PPMGeneratorTest, RegressionTest)
{
    // Build Plane UVMap
    vc::shapes::Plane plane(5, 5);
    auto mesh = plane.itkMesh();
    auto uvMap = vc::UVMap::New();
    std::size_t id{0};
    for (const auto uv : vc::range2D(5, 5)) {
        auto u = double(uv.first) / 4.0;
        auto v = double(uv.second) / 4.0;
        uvMap->set(id++, {u, v});
    }

    // Setup PPM Generator
    vct::PPMGenerator ppmGenerator;
    ppmGenerator.setDimensions(100, 100);
    ppmGenerator.setMesh(mesh);
    ppmGenerator.setUVMap(uvMap);

    // Generate PPM
    auto ppm = ppmGenerator.compute();

    // Compare against existing PPM
    auto expected = vc::PerPixelMap::ReadPPM("PPMGenerator_100x100.ppm");

    // Compare mappings
    for (const auto [y, x] : vc::range2D(100, 100)) {
        EXPECT_EQ(ppm->hasMapping(y, x), expected.hasMapping(y, x));

        if (not ppm->hasMapping(y, x)) {
            continue;
        }

        EXPECT_EQ(ppm->getMapping(y, x), expected(y, x));
        EXPECT_EQ(
            ppm->cellMap().at<std::int32_t>(y, x),
            expected.cellMap().at<std::int32_t>(y, x));
    }

    // No texture UV map was provided, so no texture coordinate map should
    // have been generated
    EXPECT_TRUE(ppm->textureCoordMap().empty());
}

TEST(PPMGeneratorTest, TextureFaceUVs)
{
    constexpr std::size_t dim{100};

    // Build Plane UVMap
    vc::shapes::Plane plane(5, 5);
    auto mesh = plane.itkMesh();
    auto uvMap = vc::UVMap::New();
    std::size_t id{0};
    for (const auto uv : vc::range2D(5, 5)) {
        auto u = double(uv.first) / 4.0;
        auto v = double(uv.second) / 4.0;
        uvMap->set(id++, {u, v});
    }

    // Derive per-face-corner UVs from the same (seamless) UV map, matching
    // what OBJReader::getFaceUVs() would produce for a mesh with no seams
    std::vector<std::array<cv::Vec2d, 3>> faceUVs;
    faceUVs.reserve(mesh->GetNumberOfCells());
    for (auto cellIt = mesh->GetCells()->Begin();
         cellIt != mesh->GetCells()->End(); ++cellIt) {
        const auto& ids = cellIt->Value()->GetPointIdsContainer();
        faceUVs.push_back(
            {uvMap->get(ids.GetElement(0)), uvMap->get(ids.GetElement(1)),
             uvMap->get(ids.GetElement(2))});
    }

    // Setup PPM Generator, providing the same UV map as both the flattening
    // and the (derived) source texture face UVs
    vct::PPMGenerator ppmGenerator;
    ppmGenerator.setDimensions(dim, dim);
    ppmGenerator.setMesh(mesh);
    ppmGenerator.setUVMap(uvMap);
    ppmGenerator.setTextureFaceUVs(faceUVs);

    // Generate PPM
    auto ppm = ppmGenerator.compute();

    // Since the texture UV map is identical to the flattening UV map,
    // barycentric interpolation of the texture UV map at each output pixel's
    // own raster coordinate must reconstruct that same coordinate
    ASSERT_FALSE(ppm->textureCoordMap().empty());
    for (const auto [y, x] : vc::range2D(dim, dim)) {
        if (not ppm->hasMapping(y, x)) {
            continue;
        }
        auto expectedU = static_cast<float>(x) / static_cast<float>(dim - 1);
        auto expectedV = static_cast<float>(y) / static_cast<float>(dim - 1);
        auto texUV = ppm->textureCoordMap().at<cv::Vec2f>(y, x);
        EXPECT_NEAR(texUV[0], expectedU, 1e-4);
        EXPECT_NEAR(texUV[1], expectedV, 1e-4);
    }
}

TEST_P(PPMGeneratorTest, PerformanceTest)
{
    // Build Plane
    vc::shapes::Plane plane(GetParam(), GetParam());

    // Get ITK Mesh
    auto mesh = plane.itkMesh();

    // Generate UV map
    auto uvMap = vc::UVMap::New();
    std::size_t pID = 0;
    for (int y = 0; y < GetParam(); y++) {
        auto v = double(y) / (GetParam() - 1);
        for (int x = 0; x < GetParam(); x++) {
            auto u = double(x) / (GetParam() - 1);
            uvMap->set(pID++, {u, v});
        }
    }

    // Setup PPM Generator
    vct::PPMGenerator ppmGenerator;
    ppmGenerator.setDimensions(100, 100);
    ppmGenerator.setMesh(mesh);
    ppmGenerator.setUVMap(uvMap);

    // Generate PPM
    auto start = std::chrono::system_clock::now();
    ppmGenerator.compute();
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> secs = end - start;
    std::cout << "size: " << GetParam() << " | elapsed: " << secs.count()
              << "s\n";
}

INSTANTIATE_TEST_SUITE_P(
    PerformanceTest,
    PPMGeneratorTest,
    testing::Values(2, 5, 10, 50, 100, 500, 1000));
