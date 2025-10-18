// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <GenericCloud.h>
#include <GenericDistribution.h>

namespace py = pybind11;
using namespace pybind11::literals;

#include "cccorelib.h"

void define_GenericDistribution(py::module &cccorelib)
{

    py::class_<cloudViewer::GenericDistribution> GenericDistribution(cccorelib, "GenericDistribution");
    GenericDistribution
        .def("getName", &cloudViewer::GenericDistribution::getName, py::return_value_policy::reference)
        .def("isValid", &cloudViewer::GenericDistribution::isValid)
        .def("computeParameters", &cloudViewer::GenericDistribution::computeParameters, "values"_a)
        .def("computeP",
             (double(cloudViewer::GenericDistribution::*)(ScalarType)
                  const)(&cloudViewer::GenericDistribution::computeP),
             "x"_a)
        .def("computeP",
             (double(cloudViewer::GenericDistribution::*)(ScalarType, ScalarType)
                  const)(&cloudViewer::GenericDistribution::computeP),
             "x1"_a,
             "x2"_a)
        .def("computePfromZero", &cloudViewer::GenericDistribution::computePfromZero, "x"_a)
        .def("computeChi2Dist",
             &cloudViewer::GenericDistribution::computeChi2Dist,
             "GenericDistribution",
             "Yk"_a,
             "numberOfClasses"_a,
             "histo"_a = nullptr);
}
