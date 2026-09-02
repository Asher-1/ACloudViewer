// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.

#define TEST_NAME "mvs/advancing_front_meshing_test"
#include "util/testing.h"

#include "mvs/advancing_front_meshing.h"

#include "util/ply.h"

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace colmap {
namespace mvs {
namespace {

class TemporaryDirectory {
public:
    TemporaryDirectory() {
        const auto suffix = std::chrono::steady_clock::now()
                                    .time_since_epoch()
                                    .count();
        path = std::filesystem::temp_directory_path() /
               ("colmap_advancing_front_test_" + std::to_string(suffix));
        std::filesystem::create_directories(path);
    }

    ~TemporaryDirectory() { std::filesystem::remove_all(path); }

    std::filesystem::path path;
};

std::vector<PlyPoint> MakeSpherePoints() {
    constexpr int kLatitudeSamples = 12;
    constexpr int kLongitudeSamples = 24;
    constexpr double kPi = 3.14159265358979323846;
    std::vector<PlyPoint> points;
    points.reserve((kLatitudeSamples - 1) * kLongitudeSamples + 2);

    auto append_point = [&points](double x, double y, double z) {
        PlyPoint point;
        point.x = static_cast<float>(x);
        point.y = static_cast<float>(y);
        point.z = static_cast<float>(z);
        point.nx = point.x;
        point.ny = point.y;
        point.nz = point.z;
        point.r = 128;
        point.g = 160;
        point.b = 192;
        points.push_back(point);
    };

    append_point(0.0, 0.0, 1.0);
    for (int latitude = 1; latitude < kLatitudeSamples; ++latitude) {
        const double phi = kPi * latitude / kLatitudeSamples;
        for (int longitude = 0; longitude < kLongitudeSamples; ++longitude) {
            const double theta = 2.0 * kPi * longitude / kLongitudeSamples;
            append_point(std::sin(phi) * std::cos(theta),
                         std::sin(phi) * std::sin(theta), std::cos(phi));
        }
    }
    append_point(0.0, 0.0, -1.0);
    return points;
}

size_t ReadPlyFaceCount(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    BOOST_REQUIRE(file.is_open());
    std::string line;
    while (std::getline(file, line) && line != "end_header") {
        constexpr char kPrefix[] = "element face ";
        if (line.rfind(kPrefix, 0) == 0) {
            return std::stoull(line.substr(sizeof(kPrefix) - 1));
        }
    }
    return 0;
}

}  // namespace

BOOST_AUTO_TEST_CASE(ReconstructsSurfaceWithoutVisibility) {
    TemporaryDirectory temporary_directory;
    const auto fused_path = temporary_directory.path / "fused.ply";
    const auto output_path = temporary_directory.path / "mesh.ply";
    WriteBinaryPlyPoints(fused_path.string(), MakeSpherePoints());

    AdvancingFrontMeshingOptions options;
    options.max_edge_length = 1.0;
    options.visibility_filtering = false;
    options.num_threads = 1;
    AdvancingFrontMeshing(options, temporary_directory.path, output_path);

    BOOST_REQUIRE(std::filesystem::is_regular_file(output_path));
    BOOST_CHECK_GT(std::filesystem::file_size(output_path), 0);
    BOOST_CHECK_GT(ReadPlyFaceCount(output_path), 0);
}

BOOST_AUTO_TEST_CASE(RejectsInvalidOptions) {
    AdvancingFrontMeshingOptions options;
    options.block_overlap = 1.1;
    BOOST_CHECK(!options.Check());
    options.block_overlap = 0.2;
    options.num_threads = 0;
    BOOST_CHECK(!options.Check());
}

}  // namespace mvs
}  // namespace colmap
