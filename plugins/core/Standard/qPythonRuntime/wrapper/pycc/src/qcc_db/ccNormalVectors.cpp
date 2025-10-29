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

#include <ecvNormalVectors.h>
#include <ecvProgressDialog.h>

namespace py = pybind11;
using namespace pybind11::literals;

void define_ccNormalVectors(py::module &m)
{
    py::class_<ccNormalVectors> pyNormalVectors(m, "ccNormalVectors");

    pyNormalVectors.def_static("GetUniqueInstance", &ccNormalVectors::GetUniqueInstance)
        .def_static("ReleaseUniqueInstance", &ccNormalVectors::ReleaseUniqueInstance)
        .def_static("GetNumberOfVectors", &ccNormalVectors::GetNumberOfVectors)
        .def_static("GetNormal", &ccNormalVectors::GetNormal, "normIndex"_a)
        .def(
            "getNormal",
            [](const ccNormalVectors &self, unsigned normIndex) -> const CCVector3 &
            { return self.getNormal(normIndex); },
            "normIndex"_a)
        .def_static(
            "GetNormIndex",
            static_cast<CompressedNormType (*)(const CCVector3 &N)>(&ccNormalVectors::GetNormIndex),
            "N"_a);

    py::native_enum<ccNormalVectors::Orientation>(
        pyNormalVectors, "Orientation", "enum.Enum", "ccNormalVectors::Orientation.")
        .value("PLUS_X", ccNormalVectors::Orientation::PLUS_X)
        .value("MINUS_X", ccNormalVectors::Orientation::MINUS_X)
        .value("PLUS_Y", ccNormalVectors::Orientation::PLUS_Y)
        .value("MINUS_Y", ccNormalVectors::Orientation::MINUS_Y)
        .value("PLUS_Z", ccNormalVectors::Orientation::MINUS_Z)
        .value("PLUS_BARYCENTER", ccNormalVectors::Orientation::PLUS_BARYCENTER)
        .value("MINUS_BARYCENTER", ccNormalVectors::Orientation::MINUS_BARYCENTER)
        .value("PLUS_ORIGIN", ccNormalVectors::Orientation::PLUS_ORIGIN)
        .value("MINUS_ORIGIN", ccNormalVectors::Orientation::MINUS_ORIGIN)
        .value("PREVIOUS", ccNormalVectors::Orientation::PREVIOUS)
        .value("PLUS_SENSOR_ORIGIN", ccNormalVectors::Orientation::PLUS_SENSOR_ORIGIN)
        .value("MINUS_SENSOR_ORIGIN", ccNormalVectors::Orientation::MINUS_SENSOR_ORIGIN)
        .value("UNDEFINED", ccNormalVectors::Orientation::UNDEFINED)
        .export_values()
        .finalize();

    // TODO NormsIndexTable type cause symbols problems at runtime when the .so of the plugin is
    // loaded pyNormalVectors
    //     .def_static("ComputeCloudNormals",
    //                 &ccNormalVectors::ComputeCloudNormals,
    //                 "cloud"_a,
    //                 "theNormCodes"_a,
    //                 "localModel"_a,
    //                 "localRadius"_a,
    //                 "preferredOrientation"_a = ccNormalVectors::Orientation::UNDEFINED,
    //                 "progressCb"_a = nullptr,
    //                 "inputOctree"_a = nullptr);
    //     .def_static("UpdateNormalOrientations",
    //                 &ccNormalVectors::UpdateNormalOrientations,
    //                 "cloud"_a,
    //                 "theNormsCodes"_a,
    //                 "preferredOrientation"_a)
    //     .def_static("ConvertNormalToStrikeAndDip",
    //                 &ccNormalVectors::ConvertNormalToStrikeAndDip,
    //                 "N"_a,
    //                 "dip_deg"_a,
    //                 "dipDir_deg"_a)
    //     .def_static("ConvertNormalToDipAndDipDir",
    //                 &ccNormalVectors::ConvertNormalToDipAndDipDir,
    //                 "N"_a,
    //                 "dip_deg"_a,
    //                 "dip_deg"_a)
    //     .def_static("ConvertDipAndDipDirToNormal",
    //                 &ccNormalVectors::ConvertNormalToDipAndDipDir,
    //                 "dip_deg"_a,
    //                 "dipDir_deg"_a,
    //                 "upward"_a = true)
    //     .def_static("ConvertStrikeAndDipToString",
    //                 &ccNormalVectors::ConvertStrikeAndDipToString,
    //                 "strike_deg"_a,
    //                 "dip_deg"_a)
    //     .def_static("ConvertDipAndDipDirToString",
    //                 &ccNormalVectors::ConvertDipAndDipDirToString,
    //                 "strike_deg"_a,
    //                 "dipDir_deg"_a)
    //     .def_static("ConvertNormalToHSV",
    //                 [](const CCVector3 &N)
    //                 {
    //                     float H, S, V;
    //                     ccNormalVectors::ConvertNormalToHSV(N, H, S, V);
    //                     return py::make_tuple(H, S, V);
    //                 })
    //     .def_static("ConvertNormalToRGB", &ccNormalVectors::ConvertNormalToRGB, "N"_a)
    //     .def("enableNormalHSVColorsArray", &ccNormalVectors::enableNormalHSVColorsArray)
    //     .def("getNormalHSVColor", &ccNormalVectors::getNormalHSVColor, "index"_a)
    //     .def("getNormalHSVColorArray", &ccNormalVectors::getNormalHSVColorArray);
    // TODO 3 helpers
}
