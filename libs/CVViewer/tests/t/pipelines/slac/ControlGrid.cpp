// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "cloudViewer/t/pipelines/slac/ControlGrid.h"

#include "cloudViewer/core/Tensor.h"
#include "cloudViewer/data/Dataset.h"
#include "cloudViewer/t/io/PointCloudIO.h"
#include "cloudViewer/t/pipelines/slac/Visualization.h"
#include "core/CoreTest.h"
#include "tests/Tests.h"

namespace cloudViewer {
namespace tests {

static t::geometry::PointCloud CreateTPCDFromFile(
        const std::string& fname,
        const core::Device& device = core::Device("CPU:0")) {
    auto pcd = io::CreatePointCloudFromFile(fname);
    return t::geometry::PointCloud::FromLegacy(*pcd, core::Float32, device);
}

class ControlGridPermuteDevices : public PermuteDevices {};
INSTANTIATE_TEST_SUITE_P(ControlGrid,
                         ControlGridPermuteDevices,
                         testing::ValuesIn(PermuteDevices::TestCases()));

// TODO(wei): more well-designed test cases
TEST_P(ControlGridPermuteDevices, Touch) {
    core::Device device = GetParam();
    t::pipelines::slac::ControlGrid cgrid(0.5, 1000, device);

    data::PCDPointCloud sample_pcd;
    t::geometry::PointCloud pcd =
            CreateTPCDFromFile(sample_pcd.GetPath(), device);
    cgrid.Touch(pcd);

    t::geometry::PointCloud pcd_param = cgrid.Parameterize(pcd);
}

TEST_P(ControlGridPermuteDevices, Deform) {
    core::Device device = GetParam();
    t::pipelines::slac::ControlGrid cgrid(0.5, 1000, device);

    data::PCDPointCloud sample_pcd;
    t::geometry::PointCloud pcd =
            CreateTPCDFromFile(sample_pcd.GetPath(), device);
    cgrid.Touch(pcd);
    cgrid.Compactify();

    t::geometry::PointCloud pcd_param = cgrid.Parameterize(pcd);

    core::Tensor prev = cgrid.GetInitPositions();
    core::Tensor curr = cgrid.GetCurrPositions();
    curr[0][0] += 0.5;
    curr[1][2] -= 0.5;
    curr[2][1] += 0.5;
}

TEST_P(ControlGridPermuteDevices, Regularizer) {
    core::Device device = GetParam();
    t::pipelines::slac::ControlGrid cgrid(0.5, 1000, device);

    data::PCDPointCloud sample_pcd;
    t::geometry::PointCloud pcd =
            CreateTPCDFromFile(sample_pcd.GetPath(), device);
    cgrid.Touch(pcd);
    cgrid.Compactify();
    core::Tensor prev = cgrid.GetInitPositions();
    core::Tensor curr = cgrid.GetCurrPositions();
    curr[0][0] += 0.2;
    curr[1][2] -= 0.2;
    curr[2][1] += 0.2;
}

}  // namespace tests
}  // namespace cloudViewer
