#include <gtest/gtest.h>

#include <cstdint>

#include "vc/core/types/PerPixelMap.hpp"

using namespace volcart;

TEST(PerPixelMap, WriteRead)
{
    // Build a PPM
    PerPixelMap ppm(100, 100);
    cv::Mat mask = cv::Mat::zeros(100, 100, CV_8UC1);
    cv::Mat cellMap = cv::Mat(100, 100, CV_32SC1);
    cellMap = cv::Scalar::all(-1);
    cv::Mat textureCoordMap = cv::Mat::zeros(100, 100, CV_32FC2);
    for (auto y = 0; y < 10; ++y) {
        for (auto x = 0; x < 10; ++x) {
            auto outY = 46 + y;
            auto outX = 46 + x;
            auto dx = static_cast<double>(x);
            auto dy = static_cast<double>(y);
            ppm(outY, outX) = {dx, dy, (dx + dy) / 2.0,
                               dx, dy, (dx + dy) / 2.0};
            mask.at<std::uint8_t>(outY, outX) = 255U;
            cellMap.at<std::int32_t>(outY, outX) = y + 10;
            textureCoordMap.at<cv::Vec2f>(outY, outX) = {
                static_cast<float>(dx / 10.0), static_cast<float>(dy / 10.0)};
        }
    }
    ppm.setMask(mask);
    ppm.setCellMap(cellMap);
    ppm.setTextureCoordMap(textureCoordMap);

    // Write the PPM
    std::string path{"vc_core_PerPixelMap_WriteRead.ppm"};
    EXPECT_NO_THROW(PerPixelMap::WritePPM(path, ppm));

    // Read the PPM
    PerPixelMap result;
    EXPECT_NO_THROW(result = PerPixelMap::ReadPPM(path));

    // Test the values
    for (auto y = 0; y < 100; ++y) {
        for (auto x = 0; x < 100; ++x) {
            EXPECT_EQ(result(y, x), ppm(y, x));
        }
    }

    // Test the mask
    cv::Mat diff = ppm.mask() != result.mask();
    EXPECT_EQ(cv::countNonZero(diff), 0);

    // Test the cell map
    diff = ppm.cellMap() != result.cellMap();
    EXPECT_EQ(cv::countNonZero(diff), 0);

    // Test the texture coordinate map
    EXPECT_EQ(result.textureCoordMap().size(), ppm.textureCoordMap().size());
    for (auto y = 0; y < 100; ++y) {
        for (auto x = 0; x < 100; ++x) {
            EXPECT_EQ(
                result.textureCoordMap().at<cv::Vec2f>(y, x),
                ppm.textureCoordMap().at<cv::Vec2f>(y, x));
        }
    }
}

TEST(PerPixelMap, WriteReadNoTextureCoordMap)
{
    // Build a PPM without a texture coordinate map (e.g. a CT-derived PPM)
    PerPixelMap ppm(10, 10);

    // Write the PPM
    std::string path{"vc_core_PerPixelMap_WriteReadNoTextureCoordMap.ppm"};
    EXPECT_NO_THROW(PerPixelMap::WritePPM(path, ppm));

    // Read the PPM
    PerPixelMap result;
    EXPECT_NO_THROW(result = PerPixelMap::ReadPPM(path));

    // The texture coordinate map should remain empty
    EXPECT_TRUE(result.textureCoordMap().empty());
}
