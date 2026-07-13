#include <algorithm>
#include <array>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <boost/program_options.hpp>

#include "vc/core/filesystem.hpp"
#include "vc/core/io/MeshIO.hpp"
#include "vc/core/util/Logging.hpp"

namespace fs = volcart::filesystem;
namespace po = boost::program_options;
namespace vc = volcart;

namespace
{
// Largest pairwise distance between a face's 3 corner UVs. A face whose
// corners are all part of the same original UV chart should have a small
// spread; a large spread indicates the face's corners were pulled from
// disconnected regions of the source UV atlas - i.e. this single triangle
// incorrectly bridges a UV seam, most commonly introduced by mesh decimation
// or retopology tools that aren't UV-seam-aware.
auto CornerSpread(const std::array<cv::Vec2d, 3>& corners) -> double
{
    auto d01 = cv::norm(corners[0] - corners[1]);
    auto d12 = cv::norm(corners[1] - corners[2]);
    auto d02 = cv::norm(corners[0] - corners[2]);
    return std::max({d01, d12, d02});
}
}  // namespace

auto main(int argc, char* argv[]) -> int
{
    // clang-format off
    po::options_description all("Usage");
    all.add_options()
        ("help,h", "Show this message")
        ("input-mesh,i", po::value<std::string>()->required(),
            "Path to the input mesh")
        ("threshold,t", po::value<double>()->default_value(0.1),
            "UV-space distance above which a face's corner UVs are "
            "considered divergent (i.e. likely bridging a UV seam)")
        ("output,o", po::value<std::string>(),
            "Optional path to a CSV file listing every flagged face's index "
            "and corner spread");
    // clang-format on

    po::variables_map parsed;
    po::store(po::command_line_parser(argc, argv).options(all).run(), parsed);

    if (parsed.count("help") > 0 || argc < 2) {
        std::cout << all << '\n';
        return EXIT_SUCCESS;
    }

    try {
        po::notify(parsed);
    } catch (po::error& e) {
        vc::Logger()->error(e.what());
        return EXIT_FAILURE;
    }

    fs::path meshPath = parsed["input-mesh"].as<std::string>();
    auto threshold = parsed["threshold"].as<double>();

    vc::Logger()->info("Loading mesh...");
    auto meshFile = vc::ReadMesh(meshPath);

    if (meshFile.faceUVs.empty()) {
        vc::Logger()->error(
            "Mesh has no per-face UV data (no UV coordinates in file, or "
            "unsupported format)");
        return EXIT_FAILURE;
    }

    vc::Logger()->info("Checking {} faces...", meshFile.faceUVs.size());

    std::size_t flaggedCount{0};
    double minSpread{std::numeric_limits<double>::max()};
    double maxSpread{0};
    double sumSpread{0};
    std::vector<std::pair<std::size_t, double>> flagged;

    for (std::size_t i = 0; i < meshFile.faceUVs.size(); i++) {
        auto spread = ::CornerSpread(meshFile.faceUVs[i]);
        minSpread = std::min(minSpread, spread);
        maxSpread = std::max(maxSpread, spread);
        sumSpread += spread;
        if (spread > threshold) {
            flaggedCount++;
            flagged.emplace_back(i, spread);
        }
    }

    auto meanSpread = sumSpread / static_cast<double>(meshFile.faceUVs.size());
    auto pct = 100.0 * static_cast<double>(flaggedCount) /
               static_cast<double>(meshFile.faceUVs.size());

    vc::Logger()->info(
        "UV seam report:\n"
        " - Total faces: {}\n"
        " - Corner-spread threshold: {}\n"
        " - Flagged faces (spread > threshold): {} ({:.3g}%)\n"
        " - Corner spread across all faces: min={:.4f}, mean={:.4f}, "
        "max={:.4f}",
        meshFile.faceUVs.size(), threshold, flaggedCount, pct, minSpread,
        meanSpread, maxSpread);

    if (parsed.count("output") > 0) {
        fs::path outPath = parsed["output"].as<std::string>();
        std::ofstream file(outPath.string());
        if (not file.is_open()) {
            vc::Logger()->error(
                "Could not open output file: {}", outPath.string());
            return EXIT_FAILURE;
        }
        file << "face_index,corner_spread\n";
        for (const auto& [idx, spread] : flagged) {
            file << idx << ',' << spread << '\n';
        }
        vc::Logger()->info("Wrote flagged face list: {}", outPath.string());
    }

    return EXIT_SUCCESS;
}
