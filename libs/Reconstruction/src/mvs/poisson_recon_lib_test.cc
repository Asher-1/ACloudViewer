// SPDX-License-Identifier: BSD-3-Clause
//
// Runtime gate for the unified PoissonReconLib wrapper (Adaptive Multigrid
// Solvers v18.75): drives the in-memory reconstruction end to end for both
// the colored and colorless paths and checks the IMesh callback contract.

#include "PoissonReconLib.h"

#include "util/testing.h"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <vector>

namespace {

// A small synthetic blob with outward normals and per-vertex colors.
template <typename Real>
class TestCloud : public PoissonReconLib::ICloud<Real> {
public:
    TestCloud(bool withColors, bool withNormals) {
        const int k = 12;
        for (int i = 0; i < k; ++i) {
            for (int j = 0; j < k; ++j) {
                const double theta = 2.0 * M_PI * i / k;
                const double phi = M_PI * j / (k - 1);
                points_.push_back({static_cast<Real>(std::sin(phi) *
                                                     std::cos(theta)),
                                   static_cast<Real>(std::sin(phi) *
                                                     std::sin(theta)),
                                   static_cast<Real>(std::cos(phi))});
                normals_.push_back(points_.back());
                if (withColors) {
                    colors_.push_back({static_cast<Real>(i) / k,
                                       static_cast<Real>(j) / k,
                                       static_cast<Real>(1)});
                }
            }
        }
        (void)withNormals;
    }

    size_t size() const override { return points_.size(); }
    bool hasNormals() const override { return true; }
    bool hasColors() const override { return !colors_.empty(); }
    void getPoint(size_t index, Real* coords) const override {
        for (int k = 0; k < 3; ++k) coords[k] = points_[index][k];
    }
    void getNormal(size_t index, Real* coords) const override {
        for (int k = 0; k < 3; ++k) coords[k] = normals_[index][k];
    }
    void getColor(size_t index, Real* rgb) const override {
        for (int k = 0; k < 3; ++k) rgb[k] = colors_[index][k];
    }

private:
    std::vector<std::array<Real, 3>> points_;
    std::vector<std::array<Real, 3>> normals_;
    std::vector<std::array<Real, 3>> colors_;
};

template <typename Real>
class TestMesh : public PoissonReconLib::IMesh<Real> {
public:
    void addVertex(const Real* coords) override {
        vertices_.push_back({coords[0], coords[1], coords[2]});
    }
    void addNormal(const Real* coords) override {
        normals_.push_back({coords[0], coords[1], coords[2]});
    }
    void addColor(const Real* rgb) override {
        colors_.push_back({rgb[0], rgb[1], rgb[2]});
    }
    void addDensity(double d) override { densities_.push_back(d); }
    void addTriangle(size_t i1, size_t i2, size_t i3) override {
        triangles_.push_back({i1, i2, i3});
    }

    std::vector<std::array<Real, 3>> vertices_;
    std::vector<std::array<Real, 3>> normals_;
    std::vector<std::array<Real, 3>> colors_;
    std::vector<double> densities_;
    std::vector<std::array<size_t, 3>> triangles_;
};

}  // namespace

TEST(PoissonReconLib, ReconstructColoredFloat) {
    PoissonReconLib::Parameters params;
    params.depth = 4;
    params.threads = 1;
    params.withColors = true;
    params.density = true;

    TestCloud<float> cloud(/*withColors=*/true, /*withNormals=*/true);
    TestMesh<float> mesh;
    EXPECT_TRUE(PoissonReconLib::Reconstruct(params, cloud, mesh));
    EXPECT_GE(mesh.vertices_.size(), 3u);
    EXPECT_GE(mesh.triangles_.size(), 1u);
    // Density is requested: one value per extracted vertex.
    EXPECT_EQ(mesh.densities_.size(), mesh.vertices_.size());
    // Colors are requested and the input carries them.
    EXPECT_EQ(mesh.colors_.size(), mesh.vertices_.size());
}

TEST(PoissonReconLib, ReconstructColorlessFloat) {
    PoissonReconLib::Parameters params;
    params.depth = 4;
    params.threads = 1;
    params.withColors = false;
    params.density = true;

    TestCloud<float> cloud(/*withColors=*/false, /*withNormals=*/true);
    TestMesh<float> mesh;
    EXPECT_TRUE(PoissonReconLib::Reconstruct(params, cloud, mesh));
    EXPECT_GE(mesh.vertices_.size(), 3u);
    EXPECT_GE(mesh.triangles_.size(), 1u);
    EXPECT_EQ(mesh.densities_.size(), mesh.vertices_.size());
    // No color auxiliary channel: the mesh must stay colorless.
    EXPECT_TRUE(mesh.colors_.empty());
}

TEST(PoissonReconLib, ReconstructColoredDouble) {
    PoissonReconLib::Parameters params;
    params.depth = 4;
    params.threads = 1;
    params.withColors = true;
    params.density = false;

    TestCloud<double> cloud(/*withColors=*/true, /*withNormals=*/true);
    TestMesh<double> mesh;
    EXPECT_TRUE(PoissonReconLib::Reconstruct(params, cloud, mesh));
    EXPECT_GE(mesh.vertices_.size(), 3u);
    EXPECT_GE(mesh.triangles_.size(), 1u);
    // Density was not requested.
    EXPECT_TRUE(mesh.densities_.empty());
    EXPECT_EQ(mesh.colors_.size(), mesh.vertices_.size());
}

TEST(PoissonReconLib, RejectsShallowDepth) {
    PoissonReconLib::Parameters params;
    params.depth = 1;
    params.threads = 1;

    TestCloud<float> cloud(/*withColors=*/false, /*withNormals=*/true);
    TestMesh<float> mesh;
    EXPECT_FALSE(PoissonReconLib::Reconstruct(params, cloud, mesh));
}
