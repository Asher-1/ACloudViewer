// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <pybind11/native_enum.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <ecvColorScalesManager.h>

#include <QColor>

#include "../casters.h"

namespace py = pybind11;
using namespace pybind11::literals;

void define_ccColorScalesManager(py::module &m)
{
    py::class_<ccColorScalesManager> pyColorScalesManager(m, "ccColorScalesManager");

    py::native_enum<ccColorScalesManager::DEFAULT_SCALES>(pyColorScalesManager,
                                                          "DEFAULT_SCALES",
                                                          "enum.Enum",
                                                          "ccColorScalesManager::DEFAULT_SCALES.")
        .value("BGYR", ccColorScalesManager::DEFAULT_SCALES::BGYR)
        .value("GREY", ccColorScalesManager::DEFAULT_SCALES::GREY)
        .value("BWR", ccColorScalesManager::DEFAULT_SCALES::BWR)
        .value("RY", ccColorScalesManager::DEFAULT_SCALES::RY)
        .value("RW", ccColorScalesManager::DEFAULT_SCALES::RW)
        .value("ABS_NORM_GREY", ccColorScalesManager::DEFAULT_SCALES::ABS_NORM_GREY)
        .value("HSV_360_DEG", ccColorScalesManager::DEFAULT_SCALES::HSV_360_DEG)
        .value("VERTEX_QUALITY", ccColorScalesManager::DEFAULT_SCALES::VERTEX_QUALITY)
        .value("DIP_BRYW", ccColorScalesManager::DEFAULT_SCALES::DIP_BRYW)
        .value("DIP_DIR_REPEAT", ccColorScalesManager::DEFAULT_SCALES::DIP_DIR_REPEAT)
        .value("VIRIDIS", ccColorScalesManager::DEFAULT_SCALES::VIRIDIS)
        .value("BROWN_YELLOW", ccColorScalesManager::DEFAULT_SCALES::BROWN_YELLOW)
        .value("YELLOW_BROWN", ccColorScalesManager::DEFAULT_SCALES::YELLOW_BROWN)
        .value("TOPO_LANDSERF", ccColorScalesManager::DEFAULT_SCALES::TOPO_LANDSERF)
        .value("HIGH_CONTRAST", ccColorScalesManager::DEFAULT_SCALES::HIGH_CONTRAST)
        .value("CIVIDIS", ccColorScalesManager::DEFAULT_SCALES::CIVIDIS)
        .export_values()
        .finalize();

    pyColorScalesManager
        .def_static("GetUniqueInstance",
                    &ccColorScalesManager::GetUniqueInstance,
                    py::return_value_policy::reference)
        .def_static("ReleaseUniqueInstance", &ccColorScalesManager::ReleaseUniqueInstance)
        .def_static("GetDefaultScaleUUID", &ccColorScalesManager::GetDefaultScaleUUID, "scale"_a)
        .def_static("GetDefaultScale",
                    &ccColorScalesManager::GetDefaultScale,
                    "scale"_a = ccColorScalesManager::DEFAULT_SCALES::BGYR)
        .def("getDefaultScale", &ccColorScalesManager::getDefaultScale, "scale"_a)
        .def("getScale", &ccColorScalesManager::getScale, "UUID"_a)
        .def("addScale", &ccColorScalesManager::addScale, "scale"_a)
        .def("removeScale", &ccColorScalesManager::removeScale, "UUID"_a)
        .def("removeScale", &ccColorScalesManager::removeScale, "UUID"_a)
        // TODO map()
        .def("fromPersistentSettings", &ccColorScalesManager::fromPersistentSettings)
        .def("toPersistentSettings", &ccColorScalesManager::toPersistentSettings);
}
