// ----------------------------------------------------------------------------
// -                        CloudViewer: Asher-1.github.io                    -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 Asher-1.github.io
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#include "t/pipelines/slac/Visualization.h"
#include "visualization/utility/DrawGeometry.h"
#include <Eigen.h>

namespace cloudViewer {
namespace t {
namespace pipelines {
namespace slac {

static const Eigen::Vector3d kSourceColor = Eigen::Vector3d(0, 1, 0);
static const Eigen::Vector3d kTargetColor = Eigen::Vector3d(1, 0, 0);
static const Eigen::Vector3d kCorresColor = Eigen::Vector3d(0, 0, 1);

static Eigen::Vector3d Jet(double v, double vmin, double vmax) {
    Eigen::Vector3d c(1, 1, 1);
    double dv;

    if (v < vmin) v = vmin;
    if (v > vmax) v = vmax;
    dv = vmax - vmin;

    if (v < (vmin + 0.25 * dv)) {
        c(0) = 0;
        c(1) = 4 * (v - vmin) / dv;
    } else if (v < (vmin + 0.5 * dv)) {
        c(0) = 0;
        c(2) = 1 + 4 * (vmin + 0.25 * dv - v) / dv;
    } else if (v < (vmin + 0.75 * dv)) {
        c(0) = 4 * (v - vmin - 0.5 * dv) / dv;
        c(2) = 0;
    } else {
        c(1) = 1 + 4 * (vmin + 0.75 * dv - v) / dv;
        c(2) = 0;
    }

    return c;
}

void VisualizePointCloudCorrespondences(const t::geometry::PointCloud& tpcd_i,
                                        const t::geometry::PointCloud& tpcd_j,
                                        const core::Tensor correspondences,
                                        const core::Tensor& T_ij) {
    Eigen::Matrix4d flip;
    flip << 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1;

    core::Tensor correspondences_host =
            correspondences.To(core::Device("CPU:0"));

    auto pcd_i_corres = cloudViewer::make_shared<ccPointCloud>(
            tpcd_i.Clone().Transform(T_ij).ToLegacy());
    pcd_i_corres->paintUniformColor(kSourceColor);
    pcd_i_corres->transform(flip);

    auto pcd_j_corres = cloudViewer::make_shared<ccPointCloud>(
            tpcd_j.ToLegacy());
    pcd_j_corres->paintUniformColor(kTargetColor);
    pcd_j_corres->transform(flip);

    std::vector<std::pair<int, int>> corres_lines;
    for (int i = 0; i < correspondences_host.GetLength(); ++i) {
        corres_lines.push_back(
                std::make_pair(correspondences_host[i][0].Item<int64_t>(),
                               correspondences_host[i][1].Item<int64_t>()));
    }
    auto lineset =
            cloudViewer::geometry::LineSet::CreateFromPointCloudCorrespondences(
                    *pcd_i_corres, *pcd_j_corres, corres_lines);
    lineset->paintUniformColor(kCorresColor);

    visualization::DrawGeometries({pcd_i_corres, pcd_j_corres, lineset});
}

void VisualizePointCloudEmbedding(t::geometry::PointCloud& tpcd_param,
                                  ControlGrid& ctr_grid,
                                  bool show_lines) {
    Eigen::Matrix4d flip;
    flip << 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1;

    // Prepare all ctr grid point cloud for lineset
    auto pcd = cloudViewer::make_shared<ccPointCloud>(
            tpcd_param.ToLegacy());
    pcd->transform(flip);

    t::geometry::PointCloud tpcd_grid(
            ctr_grid.GetCurrPositions().Slice(0, 0, ctr_grid.Size()));
    auto pcd_grid = cloudViewer::make_shared<ccPointCloud>(
            tpcd_grid.ToLegacy());
    pcd_grid->transform(flip);

    // Prepare nb point cloud for visualization
    core::Tensor corres = tpcd_param.GetPointAttr(ControlGrid::kGrid8NbIndices)
                                  .To(core::Device("CPU:0"), core::Int64);
    t::geometry::PointCloud tpcd_grid_nb(
            tpcd_grid.GetPoints().IndexGet({corres.View({-1})}));

    auto pcd_grid_nb = cloudViewer::make_shared<ccPointCloud>(
            tpcd_grid_nb.ToLegacy());
    pcd_grid_nb->paintUniformColor(kSourceColor);
    pcd_grid_nb->transform(flip);

    visualization::DrawGeometries({pcd, pcd_grid_nb}, "Point cloud embedding");

    // Prepare n x 8 corres for visualization
    std::vector<std::pair<int, int>> corres_lines;
    for (int64_t i = 0; i < corres.GetLength(); ++i) {
        for (int k = 0; k < 8; ++k) {
            std::pair<int, int> pair = {i, corres[i][k].Item<int64_t>()};
            corres_lines.push_back(pair);
        }
    }
    auto lineset =
            cloudViewer::geometry::LineSet::CreateFromPointCloudCorrespondences(
                    *pcd, *pcd_grid, corres_lines);

    core::Tensor corres_interp =
            tpcd_param.GetPointAttr(ControlGrid::kGrid8NbVertexInterpRatios)
                    .To(core::Device("CPU:0"));
    for (int64_t i = 0; i < corres.GetLength(); ++i) {
        for (int k = 0; k < 8; ++k) {
            float ratio = corres_interp[i][k].Item<float>();
            Eigen::Vector3d color = Jet(ratio, 0, 0.5);
            lineset->colors_.push_back(color);
        }
    }

    // Ensure raw pcd is visible
    pcd->paintUniformColor({0, 0, 0});
    visualization::DrawGeometries({lineset, pcd, pcd_grid_nb},
                                  "Point cloud embedding");
}

void VisualizePointCloudDeformation(const geometry::PointCloud& tpcd_param,
                                    ControlGrid& ctr_grid) {
    Eigen::Matrix4d flip;
    flip << 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1;

    core::Tensor corres = tpcd_param.GetPointAttr(ControlGrid::kGrid8NbIndices)
                                  .To(core::Device("CPU:0"), core::Int64)
                                  .View({-1});

    core::Tensor prev = ctr_grid.GetInitPositions().IndexGet({corres});
    core::Tensor curr = ctr_grid.GetCurrPositions().IndexGet({corres});

    t::geometry::PointCloud tpcd_init_grid(prev);
    auto pcd_init_grid = cloudViewer::make_shared<ccPointCloud>(
            tpcd_init_grid.ToLegacy());
    pcd_init_grid->paintUniformColor({0, 1, 0});
    pcd_init_grid->transform(flip);

    auto pcd = cloudViewer::make_shared<ccPointCloud>(
            tpcd_param.ToLegacy());
    pcd->paintUniformColor({0, 1, 0});
    pcd->transform(flip);

    t::geometry::PointCloud tpcd_curr_grid(curr);
    auto pcd_curr_grid = cloudViewer::make_shared<ccPointCloud>(
            tpcd_curr_grid.ToLegacy());
    pcd_curr_grid->paintUniformColor({1, 0, 0});
    pcd_curr_grid->transform(flip);

    auto tpcd_warped = ctr_grid.Deform(tpcd_param);
    auto pcd_warped = cloudViewer::make_shared<ccPointCloud>(
            tpcd_warped.ToLegacy());
    pcd_warped->paintUniformColor({1, 0, 0});
    pcd_warped->transform(flip);

    std::vector<std::pair<int, int>> deform_lines;
    for (size_t i = 0; i < pcd_init_grid->size(); ++i) {
        deform_lines.push_back(std::make_pair(i, i));
    }
    auto lineset =
            cloudViewer::geometry::LineSet::CreateFromPointCloudCorrespondences(
                    *pcd_init_grid, *pcd_curr_grid, deform_lines);

    visualization::DrawGeometries(
            {pcd, pcd_warped, pcd_init_grid, pcd_curr_grid, lineset},
            "Point cloud deformation");
}

void VisualizeGridDeformation(ControlGrid& cgrid) {
    core::Tensor indices, indices_nb, masks_nb;
    std::tie(indices, indices_nb, masks_nb) = cgrid.GetNeighborGridMap();

    core::Device host("CPU:0");
    indices = indices.To(host);
    indices_nb = indices_nb.To(host);
    masks_nb = masks_nb.To(host);

    int64_t n = cgrid.Size();
    core::Tensor prev = cgrid.GetInitPositions().Slice(0, 0, n);
    core::Tensor curr = cgrid.GetCurrPositions().Slice(0, 0, n);

    t::geometry::PointCloud tpcd_init_grid(prev);
    auto pcd_init_grid = cloudViewer::make_shared<ccPointCloud>(
            tpcd_init_grid.ToLegacy());
    pcd_init_grid->paintUniformColor({0, 1, 0});

    t::geometry::PointCloud tpcd_curr_grid(curr);
    auto pcd_curr_grid = cloudViewer::make_shared<ccPointCloud>(
            tpcd_curr_grid.ToLegacy());
    pcd_curr_grid->paintUniformColor({1, 0, 0});

    std::vector<std::pair<int, int>> nb_lines;
    for (int64_t i = 0; i < indices.GetLength(); ++i) {
        for (int j = 0; j < 6; ++j) {
            if (masks_nb[i][j].Item<bool>()) {
                nb_lines.push_back(std::make_pair(
                        indices[i].Item<int>(), indices_nb[i][j].Item<int>()));
            }
        }
    }

    {
        auto lineset_init =
                cloudViewer::geometry::LineSet::CreateFromPointCloudCorrespondences(
                        *pcd_init_grid, *pcd_init_grid, nb_lines);
        auto lineset_curr =
                cloudViewer::geometry::LineSet::CreateFromPointCloudCorrespondences(
                        *pcd_curr_grid, *pcd_curr_grid, nb_lines);
        visualization::DrawGeometries(
                {pcd_init_grid, pcd_curr_grid, lineset_init, lineset_curr},
                "Grid Deformation");
    }
}

}  // namespace slac
}  // namespace pipelines
}  // namespace t
}  // namespace cloudViewer
