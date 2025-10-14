// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <pybind11/pybind11.h>

#include <CVMiscTools.h>

namespace py = pybind11;
using namespace pybind11::literals;

void define_CCMiscTools(py::module &cccorelib)
{
    py::class_<cloudViewer::CCMiscTools> CCMiscTools(cccorelib, "CCMiscTools");
    CCMiscTools.def_static(
        "EnlargeBox", &cloudViewer::CCMiscTools::EnlargeBox, "dimMin"_a, "dimMax"_a, "coef"_a);
    CCMiscTools.def_static("MakeMinAndMaxCubical",
                           &cloudViewer::CCMiscTools::MakeMinAndMaxCubical,
                           "dimMin"_a,
                           "dimMax"_a,
                           "enlargeFactor"_a = 0.01);

    CCMiscTools.def_static("ComputeBaseVectors",
                           (void (*)(const CCVector3 &, CCVector3 &, CCVector3 &))(
                               &cloudViewer::CCMiscTools::ComputeBaseVectors),
                           "N"_a,
                           "X"_a,
                           "Y"_a);

    CCMiscTools.def_static("ComputeBaseVectors",
                           (void (*)(const CCVector3d &, CCVector3d &, CCVector3d &))(
                               &cloudViewer::CCMiscTools::ComputeBaseVectors),
                           "N"_a,
                           "X"_a,
                           "Y"_a);

    CCMiscTools.def_static(
        "TriBoxOverlap",
        [](const CCVector3 &boxCenter, const CCVector3 boxhalfsize, py::list &triverts)
        {
            const CCVector3 *trueTrivers[3] = {
                triverts[0].cast<CCVector3 *>(),
                triverts[1].cast<CCVector3 *>(),
                triverts[2].cast<CCVector3 *>(),
            };

            return cloudViewer::CCMiscTools::TriBoxOverlap(boxCenter, boxhalfsize, trueTrivers);
        },
        "boxcenter"_a,
        "boxhalfsize"_a,
        "triverts"_a);

    CCMiscTools.def_static("TriBoxOverlapd",
                           cloudViewer::CCMiscTools::TriBoxOverlapd,
                           "boxcenter"_a,
                           "boxhalfsize"_a,
                           "triverts"_a);
}
