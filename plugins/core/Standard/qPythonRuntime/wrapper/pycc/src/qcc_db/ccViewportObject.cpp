// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <ecv2DViewportLabel.h>
#include <ecv2DViewportObject.h>

#include "../casters.h"

namespace py = pybind11;
using namespace pybind11::literals;

void define_ccViewportObject(py::module &m)
{
    py::class_<cc2DViewportObject, ccHObject>(m, "cc2DViewportObject")
        .def(py::init<QString>(), "name"_a = QString())
        .def("setParameters", &cc2DViewportObject::setParameters, "params"_a)
        .def("getParameters", &cc2DViewportObject::getParameters);

    py::class_<cc2DViewportLabel, cc2DViewportObject>(m, "cc2DViewportLabel")
        .def("roi",
             [](const cc2DViewportLabel &self)
             {
                 py::list roi;
                 roi.append(self.roi()[0]);
                 roi.append(self.roi()[1]);
                 roi.append(self.roi()[2]);
                 roi.append(self.roi()[3]);
                 return roi;
             })
        .def("setRoi",
             [](const cc2DViewportLabel &self, py::sequence &roi)
             {
                 float newRoi[4];
                 newRoi[0] = roi[0].cast<float>();
                 newRoi[1] = roi[1].cast<float>();
                 newRoi[2] = roi[2].cast<float>();
                 newRoi[3] = roi[3].cast<float>();
             });
}
