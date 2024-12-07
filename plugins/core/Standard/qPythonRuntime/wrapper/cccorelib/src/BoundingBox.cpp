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

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>

#include <BoundingBox.h>

namespace py = pybind11;
using namespace pybind11::literals;

void define_BoundingBox(py::module &cccorelib)
{
    py::class_<cloudViewer::BoundingBox>(cccorelib, "BoundingBox")
        .def(py::init<>())
        .def(py::init(
            [](const CCVector3 &minCorner, const CCVector3 &maxCorner, bool valid = true)
            {
                auto bbox = new cloudViewer::BoundingBox(minCorner, maxCorner);
                if (bbox)
                {
                    bbox->setValidity(valid);
                }
                return bbox;
            }))
        .def("clear", &cloudViewer::BoundingBox::clear)
        .def("add", &cloudViewer::BoundingBox::add, "aPoint"_a)
        .def("minCorner", [](const cloudViewer::BoundingBox &self) { return self.minCorner(); })
        .def("maxCorner", [](const cloudViewer::BoundingBox &self) { return self.maxCorner(); })
        .def("getCenter", &cloudViewer::BoundingBox::getCenter)
        .def("getDiagVec", &cloudViewer::BoundingBox::getDiagVec)
        .def("getDiagNorm", &cloudViewer::BoundingBox::getDiagNorm)
        .def("getDiagNormd", &cloudViewer::BoundingBox::getDiagNormd)
        .def("getMinBoxDim", &cloudViewer::BoundingBox::getMinBoxDim)
        .def("getMaxBoxDim", &cloudViewer::BoundingBox::getMaxBoxDim)
        .def("computeVolume", &cloudViewer::BoundingBox::computeVolume)
        .def("setValidity", &cloudViewer::BoundingBox::setValidity, "state"_a)
        .def("isValid", &cloudViewer::BoundingBox::isValid)
        .def("minDistTo", &cloudViewer::BoundingBox::minDistTo, "box"_a)
        .def("contains", &cloudViewer::BoundingBox::contains, "P"_a)
        .def(py::self + py::self)
        .def(py::self += py::self)
        .def(py::self += CCVector3())
        .def(py::self -= CCVector3())
        .def(py::self *= double());
    // TODO operator *= square matrix
}
