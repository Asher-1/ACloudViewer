// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <ecvScalarField.h>

#include "../../cccorelib/src/wrappers.h"

namespace py = pybind11;
using namespace pybind11::literals;

void define_ccScalarField(py::module &m)
{
    py::class_<ccScalarField, cloudViewer::ScalarField, CCShareableHolder<ccScalarField>>
        pyScalarField(m, "ccScalarField");

    py::class_<ccScalarField::Range>(pyScalarField, "Range")
        .def(py::init<>())
        .def("min", &ccScalarField::Range::min)
        .def("start", &ccScalarField::Range::start)
        .def("stop", &ccScalarField::Range::stop)
        .def("max", &ccScalarField::Range::max)
        .def("range", &ccScalarField::Range::range)
        .def("maxRange", &ccScalarField::Range::maxRange)
        .def("setBounds",
             &ccScalarField::Range::setBounds,
             "minVal"_a,
             "maxVal"_a,
             "resetStartStop"_a = true)
        .def("setStart", &ccScalarField::Range::setStart, "value"_a)
        .def("setStop", &ccScalarField::Range::setStop, "value"_a)
        .def("inbound", &ccScalarField::Range::inbound, "value"_a)
        .def("isInbound", &ccScalarField::Range::isInbound, "value"_a)
        .def("isInRange", &ccScalarField::Range::isInRange, "value"_a);

    pyScalarField.def("displayRange", &ccScalarField::displayRange)
        .def("saturationRange", &ccScalarField::saturationRange)
        .def("logSaturationRange", &ccScalarField::logSaturationRange)
        .def("setMinDisplayed", &ccScalarField::setMinDisplayed, "value"_a)
        .def("setMaxDisplayed", &ccScalarField::setMaxDisplayed, "value"_a)
        .def("setSaturationStart", &ccScalarField::setSaturationStart, "value"_a)
        .def("setSaturationStop", &ccScalarField::setSaturationStop, "value"_a)
        .def("getColor",
             &ccScalarField::getColor,
             "value"_a,
             py::return_value_policy::reference_internal)
        .def("getValueColor",
             &ccScalarField::getValueColor,
             "index"_a,
             py::return_value_policy::reference_internal)
        .def("showNaNValuesInGrey", &ccScalarField::showNaNValuesInGrey, "state"_a)
        .def("areNaNValuesShownInGrey", &ccScalarField::areNaNValuesShownInGrey)
        .def("alwaysShowZero", &ccScalarField::alwaysShowZero, "state"_a)
        .def("isZeroAlwaysShown", &ccScalarField::isZeroAlwaysShown)
        .def("setSymmetricalScale", &ccScalarField::setSymmetricalScale, "state"_a)
        .def("symmetricalScale", &ccScalarField::symmetricalScale)
        .def("setLogScale", &ccScalarField::setLogScale, "state"_a)
        .def("logScale", &ccScalarField::logScale)
        .def("getColorScale", &ccScalarField::getColorScale)
        .def("setColorScale", &ccScalarField::setColorScale, "scale"_a)
        .def("getColorRampSteps", &ccScalarField::getColorRampSteps)
        .def("setColorRampSteps", &ccScalarField::setColorRampSteps, "steps"_a)
        .def("getHistogram",
             [](const ccScalarField &sf)
             {
                 // Return an independent histogram, not KDTree's opaque index container.
                 return PyCC::CopyToPythonList(sf.getHistogram());
             })
        .def("mayHaveHiddenValues", &ccScalarField::mayHaveHiddenValues)
        .def("setModificationFlag", &ccScalarField::setModificationFlag, "state"_a)
        .def("getModificationFlag", &ccScalarField::getModificationFlag)
        .def("importParametersFrom", &ccScalarField::importParametersFrom, "sf"_a)
        .def("getGlobalShift", &ccScalarField::getGlobalShift)
        .def("setGlobalShift", &ccScalarField::setGlobalShift, "state"_a);
}
