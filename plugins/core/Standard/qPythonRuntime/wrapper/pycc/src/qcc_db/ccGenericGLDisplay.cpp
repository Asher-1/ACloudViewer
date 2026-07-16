// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <ecvBBox.h>
#include <ecvGenericGLDisplay.h>
#include <ecvHObject.h>
#include <ecvViewportParameters.h>

namespace py = pybind11;
using namespace pybind11::literals;

void define_ccGenericGLDisplay(py::module &m)
{
    py::class_<ecvGenericGLDisplay> PyGLDisplay(m,
                                                "ccGenericGLDisplay",
                                                R"doc(
Per-view GL display interface.

Obtained from ccViewManager (getActiveView, getEffectiveView, findView, getAllViews).
Provides per-view operations: viewport parameters, redraw, own-DB management, etc.

Example:
    >>> vm = cloudViewer.ccViewManager.instance()
    >>> view = vm.getEffectiveView()
    >>> print(view.getUniqueID(), view.getTitle())
    >>> vp = view.getViewportParameters()
    >>> view.redraw()
)doc");

    PyGLDisplay
        .def("getUniqueID",
             &ecvGenericGLDisplay::getUniqueID,
             R"doc(Get the unique identifier of this view.)doc")
        .def("getTitle",
             &ecvGenericGLDisplay::getTitle,
             R"doc(Get the title/label of this view.)doc")

        .def("redraw",
             &ecvGenericGLDisplay::redraw,
             "only2D"_a = false,
             "forceRedraw"_a = true,
             R"doc(Redraw this specific view.)doc")
        .def("refresh",
             &ecvGenericGLDisplay::refresh,
             "only2D"_a = false,
             R"doc(Refresh this specific view.)doc")

        .def("getViewportParameters",
             &ecvGenericGLDisplay::getViewportParameters,
             py::return_value_policy::reference,
             R"doc(Get the viewport parameters for this view.)doc")
        .def("setViewportParameters",
             &ecvGenericGLDisplay::setViewportParameters,
             "params"_a,
             R"doc(Set the viewport parameters for this view.)doc")
        .def("setPerspectiveState",
             &ecvGenericGLDisplay::setPerspectiveState,
             "state"_a,
             "objectCenteredView"_a,
             "redraw"_a = true,
             R"doc(Set perspective/ortho mode for this view.)doc")

        .def("glWidth", &ecvGenericGLDisplay::glWidth, R"doc(Get the GL viewport width.)doc")
        .def("glHeight", &ecvGenericGLDisplay::glHeight, R"doc(Get the GL viewport height.)doc")
        .def("getDevicePixelRatio",
             &ecvGenericGLDisplay::getDevicePixelRatio,
             R"doc(Get the device pixel ratio.)doc")

        .def("getOwnDB",
             &ecvGenericGLDisplay::getOwnDB,
             py::return_value_policy::reference,
             R"doc(Get the per-view own database root.)doc")
        .def("addToOwnDB",
             &ecvGenericGLDisplay::addToOwnDB,
             "obj"_a,
             "noDependency"_a = true,
             R"doc(Add an object to this view's own database.)doc")
        .def("removeFromOwnDB",
             &ecvGenericGLDisplay::removeFromOwnDB,
             "obj"_a,
             R"doc(Remove an object from this view's own database.)doc")

        .def("getGLCameraParameters",
             &ecvGenericGLDisplay::getGLCameraParameters,
             "params"_a,
             R"doc(Get the GL camera parameters for this view.)doc")

        .def("invalidateViewport",
             &ecvGenericGLDisplay::invalidateViewport,
             R"doc(Invalidate the viewport for this view.)doc")
        .def("deprecate3DLayer",
             &ecvGenericGLDisplay::deprecate3DLayer,
             R"doc(Deprecate the 3D layer for this view.)doc")

        .def("setFov",
             &ecvGenericGLDisplay::setFov,
             "fov"_a,
             R"doc(Set the field of view for this view.)doc")
        .def("getFov",
             &ecvGenericGLDisplay::getFov,
             R"doc(Get the field of view (degrees) for this view.)doc")

        .def("setView",
             &ecvGenericGLDisplay::setView,
             "orientation"_a,
             R"doc(Set a preset camera orientation for this view.)doc")

        .def("updateConstellationCenterAndZoom",
             &ecvGenericGLDisplay::updateConstellationCenterAndZoom,
             "box"_a = nullptr,
             R"doc(Fit the view to the given bounding box (or all visible objects).)doc")

        .def("toBeRefreshed",
             &ecvGenericGLDisplay::toBeRefreshed,
             R"doc(Flag this view as needing a refresh on next refresh() call.)doc")

        .def("perspectiveView",
             &ecvGenericGLDisplay::perspectiveView,
             R"doc(Whether this view is in perspective mode.)doc")
        .def("objectCenteredView",
             &ecvGenericGLDisplay::objectCenteredView,
             R"doc(Whether the view is object-centered (vs viewer-based).)doc")

        .def("getCurrentViewDir",
             &ecvGenericGLDisplay::getCurrentViewDir,
             R"doc(Get the current camera view direction as a 3D vector.)doc")

        .def("moveCamera",
             &ecvGenericGLDisplay::moveCamera,
             "dx"_a,
             "dy"_a,
             "dz"_a,
             R"doc(Move the camera by the given translation offsets.)doc")
        .def("resetCamera",
             static_cast<void (ecvGenericGLDisplay::*)()>(&ecvGenericGLDisplay::resetCamera),
             R"doc(Reset the camera to the default position.)doc")

        .def("setPivotPoint",
             &ecvGenericGLDisplay::setPivotPoint,
             "P"_a,
             "autoRedraw"_a = true,
             "verbose"_a = false,
             R"doc(Set the rotation pivot point for this view.)doc")

        .def("renderToFile",
             &ecvGenericGLDisplay::renderToFile,
             "filename"_a,
             "zoomFactor"_a = 1.0f,
             "dontScale"_a = false,
             R"doc(Render the view to an image file. Returns True on success.)doc")

        .def("displayNewMessage",
             &ecvGenericGLDisplay::displayNewMessage,
             "message"_a,
             "pos"_a,
             "append"_a = false,
             "displayMaxDelay_sec"_a = 2,
             "type"_a = ecvGenericGLDisplay::CUSTOM_MESSAGE,
             R"doc(Display a temporary message on this view.)doc")

        .def("setPointSizeOnView",
             &ecvGenericGLDisplay::setPointSizeOnView,
             "size"_a,
             R"doc(Set the default point size for this specific view.)doc")
        .def("setViewportDefaultPointSize",
             &ecvGenericGLDisplay::setViewportDefaultPointSize,
             "size"_a,
             R"doc(Set the viewport default point size.)doc")
        .def("setViewportDefaultLineWidth",
             &ecvGenericGLDisplay::setViewportDefaultLineWidth,
             "width"_a,
             R"doc(Set the viewport default line width.)doc")

        .def("getLightIntensity",
             &ecvGenericGLDisplay::getLightIntensity,
             R"doc(Get the global light intensity for this view.)doc")
        .def("setLightIntensity",
             &ecvGenericGLDisplay::setLightIntensity,
             "intensity"_a,
             R"doc(Set the global light intensity for this view.)doc");
}
