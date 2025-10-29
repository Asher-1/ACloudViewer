// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

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
