// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <pybind11/pybind11.h>

#include <DistanceComputationTools.h>

namespace py = pybind11;
using namespace pybind11::literals;

void define_DistanceComputationTools(py::module &cccorelib)
{
    py::class_<cloudViewer::DistanceComputationTools> DistanceComputationTools(cccorelib,
                                                                               "DistanceComputationTools");

    py::enum_<cloudViewer::DistanceComputationTools::ERROR_MEASURES>(DistanceComputationTools,
                                                                     "ERRPOR_MEASURES")
        .value("RMS", cloudViewer::DistanceComputationTools::ERROR_MEASURES::RMS)
        .value("MAX_DIST_68_PERCENT",
               cloudViewer::DistanceComputationTools::ERROR_MEASURES::MAX_DIST_68_PERCENT)
        .value("MAX_DIST_95_PERCENT",
               cloudViewer::DistanceComputationTools::ERROR_MEASURES::MAX_DIST_95_PERCENT)
        .value("MAX_DIST_99_PERCENT",
               cloudViewer::DistanceComputationTools::ERROR_MEASURES::MAX_DIST_99_PERCENT)
        .value("MAX_DIST", cloudViewer::DistanceComputationTools::ERROR_MEASURES::MAX_DIST)
        .export_values();
}
