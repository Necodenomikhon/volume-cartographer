#include <gtest/gtest.h>

#include <array>

#include "vc/core/types/TextureStack.hpp"

using namespace volcart;

TEST(TextureStack, EmptyByDefault)
{
    TextureStack stack;
    EXPECT_TRUE(stack.empty());
    EXPECT_EQ(stack.numChannels(), 0);
    EXPECT_EQ(stack.totalSamples(), 0);
    EXPECT_THROW(stack.sample({0.5, 0.5}), std::domain_error);
}

TEST(TextureStack, RejectsEmptyImage)
{
    TextureStack stack;
    EXPECT_THROW(stack.addChannel("empty", cv::Mat()), std::invalid_argument);
}

TEST(TextureStack, RejectsDuplicateName)
{
    TextureStack stack;
    cv::Mat img(2, 2, CV_8UC1, cv::Scalar(1));
    stack.addChannel("a", img);
    EXPECT_THROW(stack.addChannel("a", img), std::invalid_argument);
}

TEST(TextureStack, UnknownChannelThrows)
{
    TextureStack stack;
    cv::Mat img(2, 2, CV_8UC1, cv::Scalar(1));
    stack.addChannel("a", img);
    EXPECT_THROW(stack.channel("b"), std::invalid_argument);
}

TEST(TextureStack, BookkeepingAndOrder)
{
    TextureStack stack;
    cv::Mat a(2, 2, CV_8UC1, cv::Scalar(1));
    cv::Mat b(2, 2, CV_8UC3, cv::Scalar(1, 2, 3));
    stack.addChannel("a", a);
    stack.addChannel("b", b);

    EXPECT_FALSE(stack.empty());
    EXPECT_EQ(stack.numChannels(), 2);
    EXPECT_EQ(stack.totalSamples(), 1 + 3);
    EXPECT_EQ(stack.channelNames(), std::vector<std::string>({"a", "b"}));
}

TEST(TextureStack, SampleHorizontalGradient)
{
    // 1 row, 5 columns: values 0, 64, 128, 192, 255 along x
    cv::Mat img(1, 5, CV_8UC1);
    std::array<uchar, 5> vals{0, 64, 128, 192, 255};
    for (int x = 0; x < 5; x++) {
        img.at<uchar>(0, x) = vals[static_cast<std::size_t>(x)];
    }

    TextureStack stack;
    stack.addChannel("gradX", img);

    // Exact grid points
    EXPECT_NEAR(stack.sample({0.0, 0.0})[0], 0.0, 1e-9);
    EXPECT_NEAR(stack.sample({0.25, 0.0})[0], 64.0, 1e-9);
    EXPECT_NEAR(stack.sample({0.5, 0.0})[0], 128.0, 1e-9);
    EXPECT_NEAR(stack.sample({1.0, 0.0})[0], 255.0, 1e-9);

    // Halfway between column 0 (0) and column 1 (64)
    EXPECT_NEAR(stack.sample({0.125, 0.0})[0], 32.0, 1e-6);
}

TEST(TextureStack, SampleVerticalGradient)
{
    // 5 rows, 1 column: values 0, 64, 128, 192, 255 along y
    cv::Mat img(5, 1, CV_8UC1);
    std::array<uchar, 5> vals{0, 64, 128, 192, 255};
    for (int y = 0; y < 5; y++) {
        img.at<uchar>(y, 0) = vals[static_cast<std::size_t>(y)];
    }

    TextureStack stack;
    stack.addChannel("gradY", img);

    EXPECT_NEAR(stack.sample({0.0, 0.0})[0], 0.0, 1e-9);
    EXPECT_NEAR(stack.sample({0.0, 0.5})[0], 128.0, 1e-9);
    EXPECT_NEAR(stack.sample({0.0, 1.0})[0], 255.0, 1e-9);
}

TEST(TextureStack, SampleConcatenatesChannelsInOrder)
{
    cv::Mat gray(1, 1, CV_8UC1, cv::Scalar(42));
    cv::Mat rgb(1, 1, CV_8UC3, cv::Scalar(10, 20, 30));

    TextureStack stack;
    stack.addChannel("gray", gray);
    stack.addChannel("rgb", rgb);

    auto result = stack.sample({0.0, 0.0});
    ASSERT_EQ(result.size(), 4);
    EXPECT_NEAR(result[0], 42.0, 1e-9);
    EXPECT_NEAR(result[1], 10.0, 1e-9);
    EXPECT_NEAR(result[2], 20.0, 1e-9);
    EXPECT_NEAR(result[3], 30.0, 1e-9);
}

TEST(TextureStack, SampleClampsOutOfRangeCoordinates)
{
    cv::Mat img(1, 5, CV_8UC1);
    std::array<uchar, 5> vals{0, 64, 128, 192, 255};
    for (int x = 0; x < 5; x++) {
        img.at<uchar>(0, x) = vals[static_cast<std::size_t>(x)];
    }

    TextureStack stack;
    stack.addChannel("gradX", img);

    EXPECT_NEAR(stack.sample({-1.0, -5.0})[0], 0.0, 1e-9);
    EXPECT_NEAR(stack.sample({2.0, 5.0})[0], 255.0, 1e-9);
}
