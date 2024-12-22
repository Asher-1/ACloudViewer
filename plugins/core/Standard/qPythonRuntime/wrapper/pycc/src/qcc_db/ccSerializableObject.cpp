// ##########################################################################
// #                                                                        #
// #                ACLOUDVIEWER PLUGIN: PythonRuntime                      #
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
        .def("toFile", &ccSerializableObject::toFile)
        .def("fromFile", &ccSerializableObject::fromFile)
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