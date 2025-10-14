// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "../casters.h"
#include "ecvMaterial.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <QImage>

namespace py = pybind11;
using namespace pybind11::literals;

void define_ccMaterial(py::module &m)
{
    py::class_<ccMaterial, ccSerializableObject>(m, "ccMaterial")
        .def(py::init<const QString &>(), py::arg("name") = QString("default"))
        .def("getName", &ccMaterial::getName)
        .def("getTextureFilename", &ccMaterial::getTextureFilename)
        .def("setName", &ccMaterial::setName)
        .def("setDiffuse", &ccMaterial::setDiffuse)
        .def("setDiffuseFront", &ccMaterial::setDiffuseFront)
        .def("setDiffuseBack", &ccMaterial::setDiffuseBack)
        .def("getDiffuseFront", &ccMaterial::getDiffuseFront)
        .def("getDiffuseBack", &ccMaterial::getDiffuseBack)
        .def("setAmbient", &ccMaterial::setAmbient)
        .def("getAmbient", &ccMaterial::getAmbient)
        .def("setIllum", &ccMaterial::setIllum)
        .def("getIllum", &ccMaterial::getIllum)
        .def("setSpecular", &ccMaterial::setSpecular)
        .def("getSpecular", &ccMaterial::getSpecular)
        .def("setEmission", &ccMaterial::setEmission)
        .def("getEmission", &ccMaterial::getEmission)
        .def("setShininess", &ccMaterial::setShininess)
        .def("setShininessFront", &ccMaterial::setShininessFront)
        .def("setShininessBack", &ccMaterial::setShininessBack)
        .def("getShininessFront", &ccMaterial::getShininessFront)
        .def("getShininessBack", &ccMaterial::getShininessBack)
        .def("setTransparency", &ccMaterial::setTransparency)
        .def("hasTexture", &ccMaterial::hasTexture)
        .def("setTexture",
             &ccMaterial::setTexture,
             py::arg("image"),
             py::arg("absoluteFilename") = QString(),
             py::arg("mirrorImage") = false)
        .def("loadAndSetTexture", &ccMaterial::loadAndSetTexture, py::arg("absoluteFilename"))
        .def("getTexture", &ccMaterial::getTexture)
        .def("getTextureID", &ccMaterial::getTextureID)
        .def("releaseTexture", &ccMaterial::releaseTexture)
        .def("compare", &ccMaterial::compare, py::arg("mtl"))
        .def("isSerializable", &ccMaterial::isSerializable)
        .def("toFile", &ccMaterial::toFile)
        .def("fromFile", &ccMaterial::fromFile)
        .def("getUniqueIdentifier", &ccMaterial::getUniqueIdentifier);

    m.def("GetTexture", &ccMaterial::GetTexture, py::arg("absoluteFilename"));
    m.def("AddTexture", &ccMaterial::AddTexture, py::arg("image"), py::arg("absoluteFilename"));
    m.def("ReleaseTextures", &ccMaterial::ReleaseTextures);
}