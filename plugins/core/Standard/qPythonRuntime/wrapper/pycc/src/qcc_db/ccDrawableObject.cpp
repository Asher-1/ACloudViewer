// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <ecvDrawableObject.h>
#include <ecvGenericDisplayTools.h>

namespace py = pybind11;
using namespace pybind11::literals;

void define_ccDrawableObject(py::module &m)
{
    py::class_<ccDrawableObject>(m, "ccDrawableObject", R"pbdoc(
    Base class of all objects that ACloudViewer is able to render/draw/display on the screen
)pbdoc")
        .def("draw", &ccDrawableObject::draw, "context"_a)
        .def("isVisible", &ccDrawableObject::isVisible)
        .def("setVisible", &ccDrawableObject::setVisible, "state"_a)
        .def("toggleVisibility", &ccDrawableObject::toggleVisibility)
        .def("isVisibilityLocked", &ccDrawableObject::isVisibilityLocked)
        .def("lockVisibility", &ccDrawableObject::lockVisibility)
        .def("isSelected", &ccDrawableObject::isSelected)
        .def("setSelected", &ccDrawableObject::setSelected, "state"_a)
        .def("hasColors", &ccDrawableObject::hasColors)
        .def("colorsShown", &ccDrawableObject::colorsShown)
        .def("showColors", &ccDrawableObject::showColors, "state"_a)
        .def("toggleColors", &ccDrawableObject::toggleColors)
        .def("hasNormals", &ccDrawableObject::hasNormals)
        .def("normalsShown", &ccDrawableObject::normalsShown)
        .def("showNormals", &ccDrawableObject::showNormals, "state"_a)
        .def("toggleNormals", &ccDrawableObject::toggleNormals)
        .def("hasDisplayedScalarField", &ccDrawableObject::hasDisplayedScalarField)
        .def("hasScalarFields", &ccDrawableObject::hasScalarFields)
        .def("showSF", &ccDrawableObject::showSF, "state"_a)
        .def("toggleSF", &ccDrawableObject::toggleSF)
        .def("sfShown", &ccDrawableObject::sfShown)
        .def("toggleMaterials", &ccDrawableObject::toggleMaterials)
        .def("showNameIn3D", &ccDrawableObject::showNameIn3D, "state"_a)
        .def("nameShownIn3D", &ccDrawableObject::nameShownIn3D)
        .def("toggleShowName", &ccDrawableObject::toggleShowName)
        // temporary color
        .def("isColorOverridden", &ccDrawableObject::isColorOverridden)
        .def("getTempColor", &ccDrawableObject::getTempColor)
        .def("setTempColor",
             (void(ccDrawableObject::*)(const ecvColor::Rgba &, bool))(
                 &ccDrawableObject::setTempColor),
             "col"_a,
             "autoActivate"_a = true)

        .def("setTempColor",
             (void(ccDrawableObject::*)(const ecvColor::Rgb &, bool))(
                 &ccDrawableObject::setTempColor),
             "col"_a,
             "autoActivate"_a = true)
        // Transformation matrix management
        .def("setGLTransformation", &ccDrawableObject::setGLTransformation, "trans"_a)
        .def("enableGLTransformation", &ccDrawableObject::enableGLTransformation, "state"_a)
        .def("isGLTransEnabled", &ccDrawableObject::isGLTransEnabled)
        .def("getGLTransformation", &ccDrawableObject::getGLTransformation)
        .def("resetGLTransformation", &ccDrawableObject::resetGLTransformation)
        .def("rotateGL", &ccDrawableObject::rotateGL, "rotMat"_a)
        .def("translateGL", &ccDrawableObject::translateGL, "rotMat"_a)
        // clipping plane
        .def("removeAllClipPlanes", &ccDrawableObject::removeAllClipPlanes)
        .def("addClipPlanes", &ccDrawableObject::addClipPlanes, "plane"_a)
        .def("toggleClipPlanes", &ccDrawableObject::toggleClipPlanes, "context"_a, "enable"_a);
}
