// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "../casters.h"
#include <pybind11/native_enum.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <QPointF>
#include <QSize>

#include <ecvGenericDisplayTools.h>

namespace py = pybind11;
using namespace pybind11::literals;

void define_ccGenericDisplayTools(py::module &m)
{
    py::class_<ccGLCameraParameters>(m, "ccGLCameraParameters")
        .def(py::init<>())
        .def("project",
             (bool(ccGLCameraParameters::*)(const CCVector3d &, CCVector3d &, bool *)
                  const)(&ccGLCameraParameters::project),
             "input3D"_a,
             "output2D"_a,
             "checkInFrustrum"_a = false)
        .def("project",
             (bool(ccGLCameraParameters::*)(const CCVector3 &, CCVector3d &, bool *)
                  const)(&ccGLCameraParameters::project),
             "input3D"_a,
             "output2D"_a,
             "checkInFrustrum"_a = false)
        .def("unproject",
             (bool(ccGLCameraParameters::*)(const CCVector3d &, CCVector3d &)
                  const)(&ccGLCameraParameters::unproject),
             "input3D"_a,
             "output2D"_a)

        .def("unproject",
             (bool(ccGLCameraParameters::*)(const CCVector3 &, CCVector3d &)
                  const)(&ccGLCameraParameters::unproject),
             "input3D"_a,
             "output2D"_a)

        .def_readwrite("modelViewMat", &ccGLCameraParameters::modelViewMat)
        .def_readwrite("projectionMat", &ccGLCameraParameters::projectionMat)
        //        .def_readwrite("viewport", &ccGLCameraParameters::viewport)
        .def_readwrite("perspective", &ccGLCameraParameters::perspective)
        .def_readwrite("fov_deg", &ccGLCameraParameters::fov_deg)
        .def_readwrite("pixelSize", &ccGLCameraParameters::pixelSize);

    py::class_<ecvGenericDisplayTools> PyecvGenericDisplayTools(m, "ccGenericDisplayTools");

    py::native_enum<ecvGenericDisplayTools::TextAlign>(
        PyecvGenericDisplayTools, "TextAlign", "enum.Enum", "ecvGenericDisplayTools::TextAlign.")
        .value("ALIGN_HLEFT", ecvGenericDisplayTools::TextAlign::ALIGN_HLEFT)
        .value("ALIGN_HMIDDLE", ecvGenericDisplayTools::TextAlign::ALIGN_HMIDDLE)
        .value("ALIGN_HRIGHT", ecvGenericDisplayTools::TextAlign::ALIGN_HRIGHT)
        .value("ALIGN_VTOP", ecvGenericDisplayTools::TextAlign::ALIGN_VTOP)
        .value("ALIGN_VMIDDLE", ecvGenericDisplayTools::TextAlign::ALIGN_VMIDDLE)
        .value("ALIGN_VBOTTOM", ecvGenericDisplayTools::TextAlign::ALIGN_VBOTTOM)
        .value("ALIGN_DEFAULT", ecvGenericDisplayTools::TextAlign::ALIGN_DEFAULT)
        .export_values()
        .finalize();

    // TODO as widget
}
