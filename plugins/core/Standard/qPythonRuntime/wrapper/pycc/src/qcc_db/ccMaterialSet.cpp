// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "../casters.h"
#include "ecvMaterialSet.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;
using namespace pybind11::literals;

void define_ccMaterialSet(py::module &m)
{
    py::class_<ccMaterialSet, std::vector<ccMaterial::CShared>, ccHObject, CCShareable>(
        m, "ccMaterialSet")
        .def(py::init<const QString &>(), py::arg("name") = QString("default"))
        .def("getClassID", &ccMaterialSet::getClassID)
        .def("isShareable", &ccMaterialSet::isShareable)
        .def("findMaterialByName", &ccMaterialSet::findMaterialByName, py::arg("mtlName"))
        .def("findMaterialByUniqueID", &ccMaterialSet::findMaterialByUniqueID, py::arg("uniqueID"))
        .def("addMaterial",
             &ccMaterialSet::addMaterial,
             py::arg("mat"),
             py::arg("allowDuplicateNames") = false)
        .def_static("ParseMTL",
                    &ccMaterialSet::ParseMTL,
                    py::arg("path"),
                    py::arg("filename"),
                    py::arg("materials"),
                    py::arg("errors"))
        .def("saveAsMTL",
             &ccMaterialSet::saveAsMTL,
             py::arg("path"),
             py::arg("baseFilename"),
             py::arg("errors"))
        .def("clone", &ccMaterialSet::clone)
        .def("append", &ccMaterialSet::append, py::arg("source"))
        .def("isSerializable", &ccMaterialSet::isSerializable);
}
