// ##########################################################################
// #                                                                        #
// #                ACLOUDVIEWER PLUGIN: PythonRuntime                       #
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

#include <pybind11/pybind11.h>

#include <CVConst.h>

namespace py = pybind11;
using namespace pybind11::literals;

void define_CCConst(py::module &cvcorelib)
{
    /* Constants */
    cvcorelib.attr("SQRT_3") = SQRT_3;
    // Python's float are doubles
    cvcorelib.attr("ZERO_TOLERANCE_F") = ZERO_TOLERANCE_F;
    cvcorelib.attr("ZERO_TOLERANCE_D") = ZERO_TOLERANCE_D;
    cvcorelib.attr("ZERO_TOLERANCE_SCALAR") = ZERO_TOLERANCE_SCALAR;
    cvcorelib.attr("ZERO_TOLERANCE_POINT_COORDINATE") = ZERO_TOLERANCE_POINT_COORDINATE;
    cvcorelib.attr("PC_ONE") = PC_ONE;
    cvcorelib.attr("PC_NAN") = PC_NAN;
    cvcorelib.attr("NAN_VALUE") = NAN_VALUE;

    // visibility
    cvcorelib.attr("POINT_VISIBLE") = POINT_VISIBLE;
    cvcorelib.attr("POINT_HIDDEN") = POINT_HIDDEN;
    cvcorelib.attr("POINT_OUT_OF_RANGE") = POINT_OUT_OF_RANGE;
    cvcorelib.attr("POINT_OUT_OF_FOV") = POINT_OUT_OF_FOV;

    py::enum_<CHAMFER_DISTANCE_TYPE>(cvcorelib, "CHAMFER_DISTANCE_TYPE")
        .value("CHAMFER_111", CHAMFER_DISTANCE_TYPE::CHAMFER_111)
        .value("CHAMFER_345", CHAMFER_DISTANCE_TYPE::CHAMFER_345);

    py::enum_<CV_LOCAL_MODEL_TYPES>(cvcorelib, "LOCAL_MODEL_TYPES")
        .value("NO_MODEL", CV_LOCAL_MODEL_TYPES::NO_MODEL)
        .value("LS", CV_LOCAL_MODEL_TYPES::LS)
        .value("TRI", CV_LOCAL_MODEL_TYPES::TRI)
        .value("QUADRIC", CV_LOCAL_MODEL_TYPES::QUADRIC);

    cvcorelib.attr("CV_LOCAL_MODEL_MIN_SIZE") = CV_LOCAL_MODEL_MIN_SIZE;
}
