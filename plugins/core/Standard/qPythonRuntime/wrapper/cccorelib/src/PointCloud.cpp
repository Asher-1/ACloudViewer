// ##########################################################################
// #                                                                        #
// #                ACLOUDVIEWER PLUGIN: PythonRuntime                       #
// #                                                                        #
// #  This program is free software; you can redistribute it and/or modify  #
// #  it under the terms of the GNU General Public License as published by  #
// #  the Free Software Foundation; version 2 of the License.               #
// #                                                                        #
// #  This program is distributed in the hope that it will be useful,       #
// #  but WITHOUT ANY WARRANTY; without even the implied warranty of        #
// #  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         #
// #  GNU General Public License for more details.                          #
// #                                                                        #
// #                   COPYRIGHT: Thomas Montaigu                           #
// #                                                                        #
// ##########################################################################

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
