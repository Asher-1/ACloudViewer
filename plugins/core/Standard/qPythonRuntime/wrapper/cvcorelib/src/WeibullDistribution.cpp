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

#include <WeibullDistribution.h>

namespace py = pybind11;
using namespace pybind11::literals;

void define_WeibullDistribution(py::module &cvcorelib)
{
    py::class_<cloudViewer::WeibullDistribution, cloudViewer::GenericDistribution>(cvcorelib,
                                                                               "WeibullDistribution")
        .def(py::init<>())
        .def(py::init<ScalarType, ScalarType, ScalarType>(), "a"_a, "b"_a, "valueShift"_a = 0)
        .def("getParameters", &cloudViewer::WeibullDistribution::getParameters, "a"_a, "b"_a)
        .def("getOtherParameters", &cloudViewer::WeibullDistribution::getOtherParameters, "mu"_a, "sigma2"_a)
        .def("computeMode", &cloudViewer::WeibullDistribution::computeMode)
        .def("computeSkewness", &cloudViewer::WeibullDistribution::computeSkewness)
        .def(
            "setParameters", &cloudViewer::WeibullDistribution::setParameters, "a"_a, "b"_a, "valueshift"_a = 0)
        .def("setValueShift", &cloudViewer::WeibullDistribution::setValueShift, "vs"_a)
        .def("getValueShift", &cloudViewer::WeibullDistribution::getValueShift);
}
