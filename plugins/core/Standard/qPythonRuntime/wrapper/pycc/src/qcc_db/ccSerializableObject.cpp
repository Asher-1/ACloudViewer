// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "../casters.h"
#include "ecvSerializableObject.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;
using namespace pybind11::literals;

void define_ccSerializableObject(py::module &m)
{
    py::class_<ccSerializableObject>(m, "ccSerializableObject")
        .def(py::init<>())
        .def("isSerializable", &ccSerializableObject::isSerializable)
        .def("toFile", &ccSerializableObject::toFile, py::arg("out"), py::arg("dataVersion"))
        .def("fromFile", &ccSerializableObject::fromFile)
        .def("minimumFileVersion", &ccSerializableObject::minimumFileVersion)
        .def_static("WriteError", &ccSerializableObject::WriteError)
        .def_static("ReadError", &ccSerializableObject::ReadError)
        .def_static("MemoryError", &ccSerializableObject::MemoryError)
        .def_static("CorruptError", &ccSerializableObject::CorruptError);
}

void define_ccSerializationHelper(py::module &m)
{
    py::class_<ccSerializationHelper>(m, "ccSerializationHelper")
        .def_static("CoordsFromDataStream", &ccSerializationHelper::CoordsFromDataStream)
        .def_static("ScalarsFromDataStream", &ccSerializationHelper::ScalarsFromDataStream);
}