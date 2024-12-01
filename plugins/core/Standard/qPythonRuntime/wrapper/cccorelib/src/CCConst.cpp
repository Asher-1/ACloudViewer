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

void define_CCConst(py::module &cccorelib)
{
    /* Constants */
    cccorelib.attr("SQRT_3") = SQRT_3;
    // Python's float are doubles
    cccorelib.attr("ZERO_TOLERANCE_F") = ZERO_TOLERANCE_F;
    cccorelib.attr("ZERO_TOLERANCE_D") = ZERO_TOLERANCE_D;
    cccorelib.attr("ZERO_TOLERANCE_SCALAR") = ZERO_TOLERANCE_SCALAR;
    cccorelib.attr("ZERO_TOLERANCE_POINT_COORDINATE") = ZERO_TOLERANCE_POINT_COORDINATE;
    cccorelib.attr("PC_ONE") = PC_ONE;
    cccorelib.attr("PC_NAN") = PC_NAN;
    cccorelib.attr("NAN_VALUE") = NAN_VALUE;

    // visibility
    cccorelib.attr("POINT_VISIBLE") = POINT_VISIBLE;
    cccorelib.attr("POINT_HIDDEN") = POINT_HIDDEN;
    cccorelib.attr("POINT_OUT_OF_RANGE") = POINT_OUT_OF_RANGE;
    cccorelib.attr("POINT_OUT_OF_FOV") = POINT_OUT_OF_FOV;

    py::enum_<CHAMFER_DISTANCE_TYPE>(cccorelib, "CHAMFER_DISTANCE_TYPE")
        .value("CHAMFER_111", CHAMFER_DISTANCE_TYPE::CHAMFER_111)
        .value("CHAMFER_345", CHAMFER_DISTANCE_TYPE::CHAMFER_345);

    py::enum_<CV_LOCAL_MODEL_TYPES>(cccorelib, "LOCAL_MODEL_TYPES")
        .value("NO_MODEL", CV_LOCAL_MODEL_TYPES::NO_MODEL)
        .value("LS", CV_LOCAL_MODEL_TYPES::LS)
        .value("TRI", CV_LOCAL_MODEL_TYPES::TRI)
        .value("QUADRIC", CV_LOCAL_MODEL_TYPES::QUADRIC);

    cccorelib.attr("CV_LOCAL_MODEL_MIN_SIZE") = CV_LOCAL_MODEL_MIN_SIZE;
}
