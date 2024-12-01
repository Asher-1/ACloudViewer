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

#include <AutoSegmentationTools.h>
#include <DgmOctree.h>
#include <GenericIndexedCloudPersist.h>
#include <GenericProgressCallback.h>

namespace py = pybind11;
using namespace pybind11::literals;

void define_AutoSegmentationTools(py::module &cccorelib)
{
    py::class_<cloudViewer::AutoSegmentationTools> AutoSegmentationTools(cccorelib, "AutoSegmentationTools");

    AutoSegmentationTools.def_static("labelConnectedComponents",
                                     &cloudViewer::AutoSegmentationTools::labelConnectedComponents,
                                     "theCloud"_a,
                                     "level"_a,
                                     "sixConnexity"_a = false,
                                     "progressCb"_a = nullptr,
                                     "inputOctree"_a = nullptr);

    AutoSegmentationTools.def_static("extractConnectedComponents",
                                     &cloudViewer::AutoSegmentationTools::extractConnectedComponents,
                                     "theCloud"_a,
                                     "ccc"_a);

    AutoSegmentationTools.def_static("frontPropagationBasedSegmentation",
                                     &cloudViewer::AutoSegmentationTools::frontPropagationBasedSegmentation,
                                     "theCloud"_a,
                                     "radius"_a,
                                     "minSeeDist"_a,
                                     "octreeLevel"_a,
                                     "theSegmentedLists"_a,
                                     "progressCb"_a = nullptr,
                                     "inputOctree"_a = nullptr,
                                     "applyGaussianFilter"_a = false,
                                     "alpha"_a = 2.0f);
}
