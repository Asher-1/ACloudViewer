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

#include <ecvViewportParameters.h>

#include <QRect>

#include "../casters.h"

namespace py = pybind11;
using namespace pybind11::literals;

void define_ccViewPortParameters(py::module &m)
{
    py::class_<ecvViewportParameters>(m, "ccViewportParameters")
        .def_readwrite("viewMat", &ecvViewportParameters::viewMat)
        .def_readwrite("defaultPointSize", &ecvViewportParameters::defaultPointSize)
        .def_readwrite("defaultLineWidth", &ecvViewportParameters::defaultLineWidth)
        .def_readwrite("perspectiveView", &ecvViewportParameters::perspectiveView)
        .def_readwrite("objectCenteredView", &ecvViewportParameters::objectCenteredView)
        .def_readwrite("zNearCoef", &ecvViewportParameters::zNearCoef)
        .def_readwrite("zNearCoef", &ecvViewportParameters::zNearCoef)
        .def_readwrite("zNear", &ecvViewportParameters::zNear)
        .def_readwrite("zFar", &ecvViewportParameters::zFar)
        .def_readwrite("fov_deg", &ecvViewportParameters::fov_deg)
        .def_readwrite("cameraAspectRatio", &ecvViewportParameters::cameraAspectRatio)
        .def(py::init<>())
        .def("setPivotPoint", &ecvViewportParameters::setPivotPoint, "P"_a, "autoUpdateFocal"_a)
        .def("getPivotPoint", &ecvViewportParameters::getPivotPoint)
        .def("setCameraCenter", &ecvViewportParameters::setCameraCenter, "C"_a, "autoUpdateFocal"_a)
        .def("getCameraCenter", &ecvViewportParameters::getCameraCenter)
        .def("setFocalDistance", &ecvViewportParameters::setFocalDistance, "distance"_a)
        .def("getFocalDistance", &ecvViewportParameters::getFocalDistance)
        .def("computeViewMatrix", &ecvViewportParameters::computeViewMatrix)
        .def("computeScaleMatrix",
             [](const ecvViewportParameters &self, const py::sequence &glViewPort)
             {
                 const QRect vp(glViewPort[0].cast<int>(),
                                glViewPort[1].cast<int>(),
                                glViewPort[2].cast<int>(),
                                glViewPort[3].cast<int>());
                 return self.computeScaleMatrix(vp);
             })
        .def("getViewDir", &ecvViewportParameters::getViewDir)
        .def("getUpDir", &ecvViewportParameters::getUpDir)
        .def("getRotationCenter", &ecvViewportParameters::getRotationCenter)
        .def("computeDistanceToHalfWidthRation",
             &ecvViewportParameters::computeDistanceToHalfWidthRatio)
        .def("computeDistanceToWidthRatio", &ecvViewportParameters::computeDistanceToWidthRatio)
        .def("computeWidthAtFocalDist", &ecvViewportParameters::computeWidthAtFocalDist)
        .def("computePixelSize", &ecvViewportParameters::computePixelSize);
}
