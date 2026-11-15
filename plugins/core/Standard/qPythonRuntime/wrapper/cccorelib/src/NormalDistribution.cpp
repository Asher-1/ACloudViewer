// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <pybind11/pybind11.h>

#include <GenericCloud.h>
#include <NormalDistribution.h>
#include <iostream>

#include "wrappers.h"

namespace py = pybind11;
using namespace pybind11::literals;

void define_NormalDistribution(py::module &cccorelib)
{
    py::class_<cloudViewer::NormalDistribution, cloudViewer::GenericDistribution>(cccorelib,
                                                                                  "NormalDistribution")
        .def(py::init<>())
        .def(py::init<ScalarType, ScalarType>(), "_mu"_a, "_sigma2"_a)
        .def("getParameters",
             [](const cloudViewer::NormalDistribution &self)
             {
                 ScalarType mu, sigma2 = 0.0;
                 self.getParameters(mu, sigma2);
                 return std::make_pair(mu, sigma2);
             })
        .def("setParameters", &cloudViewer::NormalDistribution::setParameters, "_mu"_a, "_sigma2"_a)
        .def("getMu", &cloudViewer::NormalDistribution::getMu)
        .def("getSigma2", &cloudViewer::NormalDistribution::getSigma2)
        .def("computeParameters",
             (bool(cloudViewer::NormalDistribution::*)(const cloudViewer::GenericCloud *))(
                 &cloudViewer::NormalDistribution::computeParameters),
             "cloud"_a)
        .def(
            "computeRobustParameters",
            [](cloudViewer::NormalDistribution &self, py::handle values, double nSigma)
            {
                return PyCC::WithContainer<cloudViewer::GenericDistribution::ScalarContainer>(
                    values,
                    [&](const auto &samples) { return self.computeRobustParameters(samples, nSigma); });
            },
            "values"_a,
            "nSigma"_a);
}
