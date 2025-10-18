// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <CVPointCloud.h>
#include <GenericIndexedCloud.h>
#include <PointCloudTpl.h>

#include "wrappers.h"

void define_PointCloud(py::module &cccorelib)
{

    using cloudViewer::GenericIndexedCloudPersist;
    using cloudViewer::PointCloud;
    using cloudViewer::PointCloudTpl;

    DEFINE_POINTCLOUDTPL(GenericIndexedCloudPersist, cccorelib, "__pointCloudTplCCCoreLib");
    py::class_<PointCloud, cloudViewer::PointCloudTpl<GenericIndexedCloudPersist>>(cccorelib, "PointCloud")
        .def(py::init<>())
        .def("reserveNormals", &PointCloud::reserveNormals, "newCount"_a)
        .def("addNormal", &PointCloud::addNormal, "normal"_a);
}
