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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <GenericCloud.h>
#include <GenericDistribution.h>
#include <GenericIndexedCloudPersist.h>
#include <GenericProgressCallback.h>
#include <StatisticalTestingTools.h>

namespace py = pybind11;
using namespace pybind11::literals;

void define_StatisticalTestingTools(py::module &cvcorelib)
{
    py::class_<cloudViewer::StatisticalTestingTools>(cvcorelib, "StatisticalTestingTools")
        .def_static("computeAdaptativeChi2Dist",
                    &cloudViewer::StatisticalTestingTools::computeAdaptativeChi2Dist,
                    "distrib"_a,
                    "cloud"_a,
                    "numberOfClasses"_a,
                    "finalNumberOfClasses"_a,
                    "noClassCompression"_a = false,
                    "histoMin"_a = nullptr,
                    "histoMax"_a = nullptr,
                    "histoValues"_a = nullptr,
                    "npis"_a = nullptr)
        .def_static(
            "computeChi2Fractile", &cloudViewer::StatisticalTestingTools::computeChi2Fractile, "p"_a, "d"_a)
        .def_static("computeChi2Probability",
                    &cloudViewer::StatisticalTestingTools::computeChi2Probability,
                    "chi2result"_a,
                    "d"_a)
        .def_static("testCloudWithStatisticalModel",
                    &cloudViewer::StatisticalTestingTools::testCloudWithStatisticalModel,
                    "distrib"_a,
                    "theCloud"_a,
                    "numberOfNeighbours"_a,
                    "pTrust"_a,
                    "progressCb"_a = nullptr,
                    "inputOctree"_a = nullptr);
}
