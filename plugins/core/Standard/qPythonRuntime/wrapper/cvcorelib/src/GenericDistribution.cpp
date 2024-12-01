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

namespace py = pybind11;
using namespace pybind11::literals;

#include "cvcorelib.h"

void define_GenericDistribution(py::module &cvcorelib)
{

    py::class_<cloudViewer::GenericDistribution> GenericDistribution(cvcorelib, "GenericDistribution");
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

    py::bind_vector<cloudViewer::GenericDistribution::ScalarContainer>(GenericDistribution, "ScalarContainer");
}
