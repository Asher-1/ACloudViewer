// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <pybind11/pybind11.h>

#include <WeibullDistribution.h>

namespace py = pybind11;
using namespace pybind11::literals;

void define_WeibullDistribution(py::module &cccorelib)
{
    py::class_<cloudViewer::WeibullDistribution, cloudViewer::GenericDistribution>(cccorelib,
                                                                                   "WeibullDistribution")
        .def(py::init<>())
        .def(py::init<ScalarType, ScalarType, ScalarType>(), "a"_a, "b"_a, "valueShift"_a = 0)
        .def("getParameters", &cloudViewer::WeibullDistribution::getParameters, "a"_a, "b"_a)
        .def("getOtherParameters", &cloudViewer::WeibullDistribution::getOtherParameters, "mu"_a, "sigma2"_a)
        .def("computeMode", &cloudViewer::WeibullDistribution::computeMode)
        .def("computeSkewness", &cloudViewer::WeibullDistribution::computeSkewness)
        .def("setParameters",
             &cloudViewer::WeibullDistribution::setParameters,
             "a"_a,
             "b"_a,
             "valueshift"_a = 0)
        .def("setValueShift", &cloudViewer::WeibullDistribution::setValueShift, "vs"_a)
        .def("getValueShift", &cloudViewer::WeibullDistribution::getValueShift);
}
