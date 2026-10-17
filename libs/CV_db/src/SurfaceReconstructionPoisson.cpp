// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Screened Poisson surface reconstruction entry point (ccMesh factory). The
// engine lives in the unified PoissonReconLib wrapper (Adaptive Multigrid
// Solvers v18.75, the upstream COLMAP vendored tree, shared with the
// qPoissonRecon plugin); this file is a thin ccPointCloud/ccMesh adapter that
// preserves the historical public API of ccMesh::CreateFromPointCloudPoisson.

#include <Logging.h>

#include <Eigen/Dense>
#include <tuple>
#include <vector>

#include "PoissonReconLib.h"
#include "ecvHObjectCaster.h"
#include "ecvMesh.h"
#include "ecvPointCloud.h"

using namespace cloudViewer;

namespace {

// Adapts ccPointCloud to the PoissonReconLib input interface.
template <typename Real>
class CloudAdapter : public PoissonReconLib::ICloud<Real> {
public:
    explicit CloudAdapter(const ccPointCloud& cloud) : cloud_(cloud) {}

    size_t size() const override { return cloud_.size(); }
    bool hasNormals() const override { return cloud_.hasNormals(); }
    bool hasColors() const override { return cloud_.hasColors(); }

    void getPoint(size_t index, Real* coords) const override {
        const Eigen::Vector3d& point = cloud_.getEigenPoint(index);
        for (int k = 0; k < 3; ++k) {
            coords[k] = static_cast<Real>(point[k]);
        }
    }

    void getNormal(size_t index, Real* coords) const override {
        // getEigenNormal returns the decoded unit normal ([-1, 1] range).
        const Eigen::Vector3d& normal = cloud_.getEigenNormal(index);
        for (int k = 0; k < 3; ++k) {
            coords[k] = static_cast<Real>(normal[k]);
        }
    }

    void getColor(size_t index, Real* rgb) const override {
        const Eigen::Vector3d& color = cloud_.getEigenColor(index);
        for (int k = 0; k < 3; ++k) {
            rgb[k] = static_cast<Real>(color[k]);
        }
    }

private:
    const ccPointCloud& cloud_;
};

// Collects the reconstructed surface into ccMesh vertex/normal/color/density
// buffers and the triangle index list.
template <typename Real>
class MeshAdapter : public PoissonReconLib::IMesh<Real> {
public:
    void addVertex(const Real* coords) override {
        vertices_.emplace_back(coords[0], coords[1], coords[2]);
    }
    void addNormal(const Real* coords) override {
        normals_.emplace_back(coords[0], coords[1], coords[2]);
    }
    void addColor(const Real* rgb) override {
        colors_.emplace_back(rgb[0], rgb[1], rgb[2]);
    }
    void addDensity(double d) override { densities_.push_back(d); }
    void addTriangle(size_t i1, size_t i2, size_t i3) override {
        triangles_.emplace_back(static_cast<Eigen::Vector3i::Index>(i1),
                                static_cast<Eigen::Vector3i::Index>(i2),
                                static_cast<Eigen::Vector3i::Index>(i3));
    }

    // Materializes the collected buffers into the ccMesh created by the
    // caller (mirrors the historical assembly: points + normals + colors on
    // the associated cloud, triangles on the mesh).
    bool Materialize(const std::shared_ptr<ccMesh>& mesh,
                     std::vector<double>& densities) const {
        ccPointCloud* cloud =
                ccHObjectCaster::ToPointCloud(mesh->getAssociatedCloud());
        if (!cloud) {
            utility::LogError(
                    "[CreateFromPointCloudPoisson] ccMesh should set "
                    "associated cloud before using!");
            return false;
        }
        if (!cloud->reserveThePointsTable(
                    static_cast<unsigned>(vertices_.size()))) {
            utility::LogError(
                    "[CreateFromPointCloudPoisson] reserve points failed, not "
                    "enough memory!");
            return false;
        }
        if (!cloud->reserveTheNormsTable()) {
            utility::LogError(
                    "[CreateFromPointCloudPoisson] reserve normals failed, not "
                    "enough memory!");
            return false;
        }
        const bool withColors = !colors_.empty();
        if (withColors && !cloud->reserveTheRGBTable()) {
            utility::LogError(
                    "[CreateFromPointCloudPoisson] reserve rgbs failed, not "
                    "enough memory!");
            return false;
        }

        for (const Eigen::Vector3d& point : vertices_) {
            cloud->addEigenPoint(point);
        }
        for (const Eigen::Vector3d& normal : normals_) {
            cloud->addEigenNorm(normal);
        }
        if (withColors) {
            for (const Eigen::Vector3d& color : colors_) {
                cloud->addEigenColor(color);
            }
        }

        if (!mesh->reserve(triangles_.size())) {
            utility::LogError(
                    "[CreateFromPointCloudPoisson] reserve triangles failed, "
                    "not enough memory!");
            return false;
        }
        for (const Eigen::Vector3i& triangle : triangles_) {
            mesh->addTriangle(triangle);
        }

        densities = densities_;
        return true;
    }

private:
    std::vector<Eigen::Vector3d> vertices_;
    std::vector<Eigen::Vector3d> normals_;
    std::vector<Eigen::Vector3d> colors_;
    std::vector<double> densities_;
    std::vector<Eigen::Vector3i> triangles_;
};

}  // namespace

std::tuple<std::shared_ptr<ccMesh>, std::vector<double>>
ccMesh::CreateFromPointCloudPoisson(const ccPointCloud& pcd,
                                    size_t depth,
                                    size_t width,
                                    float scale,
                                    bool linear_fit,
                                    float point_weight,
                                    float samples_per_node,
                                    int boundary_type,
                                    int n_threads) {
    if (!pcd.hasNormals()) {
        utility::LogError("[CreateFromPointCloudPoisson] pcd has no normals");
    }
    if (depth < 2) {
        utility::LogError(
                "[CreateFromPointCloudPoisson] depth (={}) has to be >= 2",
                depth);
    }

    if (n_threads <= 0) {
        n_threads = (int)std::thread::hardware_concurrency();
    }

    PoissonReconLib::Parameters params;
    params.depth = static_cast<int>(depth);
    params.finestCellWidth = static_cast<float>(width);
    params.scale = scale;
    params.linearFit = linear_fit;
    params.pointWeight = point_weight;
    params.samplesPerNode = samples_per_node;
    params.threads = n_threads;
    // The historical integer contract maps onto the wrapper's boundary enum
    // (0=FREE, 1=DIRICHLET, 2=NEUMANN); unknown values fall back to NEUMANN.
    switch (boundary_type) {
        case 0:
            params.boundary = PoissonReconLib::Parameters::FREE;
            break;
        case 1:
            params.boundary = PoissonReconLib::Parameters::DIRICHLET;
            break;
        case 2:
        default:
            params.boundary = PoissonReconLib::Parameters::NEUMANN;
            break;
    }
    params.withColors = pcd.hasColors();
    // The densities are part of the returned tuple, so extraction always
    // carries the per-vertex sampling density.
    params.density = true;

    auto mesh = std::make_shared<ccMesh>();
    if (!mesh->CreateInternalCloud()) {
        utility::LogError(
                "[CreateFromPointCloudPoisson] creating internal cloud "
                "failed!");
    }

    CloudAdapter<float> cloudAdapter(pcd);
    MeshAdapter<float> meshAdapter;
    if (!PoissonReconLib::Reconstruct(params, cloudAdapter, meshAdapter)) {
        utility::LogError("[CreateFromPointCloudPoisson] reconstruction failed");
    }

    std::vector<double> densities;
    if (!meshAdapter.Materialize(mesh, densities)) {
        utility::LogError("[CreateFromPointCloudPoisson] mesh assembly failed");
    }

    return std::make_tuple(mesh, densities);
}
