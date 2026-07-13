#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <boost/program_options.hpp>

#include "vc/app_support/GeneralOptions.hpp"
#include "vc/app_support/ProgressIndicator.hpp"
#include "vc/core/filesystem.hpp"
#include "vc/core/io/ImageIO.hpp"
#include "vc/core/types/PerPixelMap.hpp"
#include "vc/core/types/TextureStack.hpp"
#include "vc/core/util/DateTime.hpp"
#include "vc/core/util/Logging.hpp"
#include "vc/texturing/TextureStackTexture.hpp"

namespace vc = volcart;
namespace vct = volcart::texturing;
namespace fs = volcart::filesystem;
namespace po = boost::program_options;

auto main(int argc, char* argv[]) -> int
{
    // clang-format off
    po::options_description ioOpts("Input/Output Options");
    ioOpts.add_options()
        ("ppm,p", po::value<std::string>()->required(),
            "Input PPM file. Must have been generated with a source texture "
            "UV map (see vc_generate_ppm).")
        ("channel,c", po::value<std::vector<std::string>>()->required(),
            "A named source texture channel, in the form 'name=path'. "
            "Repeatable. Example: --channel rgb=capture_rgb.png "
            "--channel ir=capture_ir.tif")
        ("output-dir,o", po::value<std::string>()->required(),
            "Output directory. One image is written per channel, named "
            "'<name>.tif'.");

    po::options_description all("Usage");
    all.add(GetGeneralOpts()).add(ioOpts);
    // clang-format on

    // Parse the cmd line
    po::variables_map parsed;
    po::store(po::command_line_parser(argc, argv).options(all).run(), parsed);

    // Show the help message
    if (parsed.count("help") > 0 || argc < 2) {
        std::cout << all << '\n';
        return EXIT_SUCCESS;
    }

    // Warn of missing options
    try {
        po::notify(parsed);
    } catch (po::error& e) {
        vc::Logger()->error(e.what());
        return EXIT_FAILURE;
    }

    // Set logging level
    vc::logging::SetLogLevel(parsed["log-level"].as<std::string>());

    // Get the parsed options
    const fs::path ppmPath = parsed["ppm"].as<std::string>();
    const auto channelArgs =
        parsed["channel"].as<std::vector<std::string>>();

    // Check for output directory
    auto outDir = fs::weakly_canonical(parsed["output-dir"].as<std::string>());
    if (outDir.has_extension()) {
        vc::Logger()->error(
            "Provided output path is not a directory: {}", outDir.string());
        return EXIT_FAILURE;
    }
    if (not fs::exists(outDir)) {
        if (not fs::create_directory(outDir)) {
            vc::Logger()->error(
                "Could not create output directory: {}. Check that parent "
                "exists and is writable.",
                outDir.string());
            return EXIT_FAILURE;
        }
    }

    // Load the PPM
    vc::Logger()->info("Loading PPM...");
    auto ppm =
        vc::PerPixelMap::New(std::move(vc::PerPixelMap::ReadPPM(ppmPath)));
    if (ppm->textureCoordMap().empty()) {
        vc::Logger()->error(
            "PPM has no texture coordinate map. It must be generated with "
            "vc_generate_ppm using a mesh that has its own texture UVs.");
        return EXIT_FAILURE;
    }

    // Load the texture stack channels
    vc::Logger()->info("Loading texture channels...");
    auto stack = vc::TextureStack::New();
    for (const auto& arg : channelArgs) {
        auto pos = arg.find('=');
        if (pos == std::string::npos) {
            vc::Logger()->error(
                "Invalid --channel argument (expected 'name=path'): {}", arg);
            return EXIT_FAILURE;
        }
        auto name = arg.substr(0, pos);
        fs::path path = arg.substr(pos + 1);
        auto img = vc::ReadImage(path);
        if (img.empty()) {
            vc::Logger()->error(
                "Failed to load channel image: {}", path.string());
            return EXIT_FAILURE;
        }
        stack->addChannel(name, img);
    }

    // Compute
    vct::TextureStackTexture texGen;
    texGen.setPerPixelMap(ppm);
    texGen.setTextureStack(stack);

    if (parsed["progress"].as<bool>()) {
        vc::ProgressConfig cfg;
        cfg.label = "Sampling texture stack:";
        if (parsed.count("progress-interval") > 0) {
            cfg.interval = vc::DurationFromString(
                parsed["progress-interval"].as<std::string>());
        }
        vc::ReportProgress(texGen, cfg);
    } else {
        vc::Logger()->info("Sampling texture stack...");
    }

    auto textures = texGen.compute();

    // Write outputs, rescaled per-image to the full 16-bit unsigned range.
    // Multi-channel (e.g. RGB) floating-point TIFFs are poorly supported by
    // common readers -- notably Pillow, which ink-id's data loading uses, can
    // open a single-channel float TIFF but not a multi-channel one, even
    // though libtiff itself (and VC's own TIFFIO) read them fine. 16-bit
    // integer TIFFs are universally supported, and since real capture images
    // are practically always <=16-bit to begin with, the precision loss
    // versus floating-point output is negligible.
    vc::Logger()->info("Writing output images...");
    auto names = stack->channelNames();
    constexpr auto maxU16 = std::numeric_limits<std::uint16_t>::max();
    for (std::size_t i = 0; i < names.size(); i++) {
        double minVal{0};
        double maxVal{0};
        cv::minMaxLoc(textures[i].reshape(1), &minVal, &maxVal);

        cv::Mat scaled;
        if (maxVal > minVal) {
            const auto scale = static_cast<double>(maxU16) / (maxVal - minVal);
            textures[i].convertTo(
                scaled, CV_MAKETYPE(CV_16U, textures[i].channels()), scale,
                -minVal * scale);
        } else {
            scaled = cv::Mat::zeros(
                textures[i].size(),
                CV_MAKETYPE(CV_16U, textures[i].channels()));
        }

        auto outPath = outDir / (names[i] + ".tif");
        vc::WriteImage(outPath, scaled);
    }

    vc::Logger()->info("Done.");
    return EXIT_SUCCESS;
}
