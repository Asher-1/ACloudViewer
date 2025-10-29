// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <GenericCloud.h>
#include <GenericIndexedCloudPersist.h>
#include <GenericProgressCallback.h>
#include <ScalarFieldTools.h>

namespace py = pybind11;
using namespace pybind11::literals;

using cloudViewer::ScalarFieldTools;

#include "cccorelib.h"

void define_ScalarFieldTools(py::module &cccorelib)
{
    py::class_<cloudViewer::KMeanClass> pyKMeanClass(cccorelib, "KMeanClass");
    py::class_<ScalarFieldTools> pyScalarFieldTools(cccorelib, "ScalarFieldTools");

    pyKMeanClass.def_readwrite("mean", &cloudViewer::KMeanClass::mean)
        .def_readwrite("minValue", &cloudViewer::KMeanClass::minValue)
        .def_readwrite("maxValue", &cloudViewer::KMeanClass::maxValue);

    pyScalarFieldTools
        .def_static("computeMeanScalarValue", &ScalarFieldTools::computeMeanScalarValue, "theCloud"_a)
        .def_static(
            "computeMeanSquareScalarValue", &ScalarFieldTools::computeMeanSquareScalarValue, "theCloud"_a)
        .def_static("computeScalarFieldGradient",
                    &ScalarFieldTools::computeScalarFieldGradient,
                    "theCloud"_a,
                    "radius"_a,
                    "euclideanDistances"_a,
                    "sameInAndOutScalarField"_a = false,
                    "progressCb"_a = nullptr,
                    "theOctree"_a = nullptr)
        .def_static("applyScalarFieldGaussianFilter",
                    &ScalarFieldTools::applyScalarFieldGaussianFilter,
                    "sigma"_a,
                    "theCloud"_a,
                    "sigmaSF"_a,
                    "progressCb"_a = nullptr,
                    "theOctree"_a = nullptr)
        .def_static("multiplyScalarFields",
                    &ScalarFieldTools::multiplyScalarFields,
                    "firstCloud"_a,
                    "secondCloud"_a,
                    "progressCb"_a = nullptr)
        .def_static(
            "computeScalarFieldHistogram",
            [](const cloudViewer::GenericCloud *theCloud, unsigned numberOfClasses)
            {
                std::vector<int> histo;
                ScalarFieldTools::computeScalarFieldHistogram(theCloud, numberOfClasses, histo);
                return py::array_t<int>(histo.size(), histo.data());
            },
            "theCloud"_a,
            "numberOfClasses"_a)
        .def_static(
            "computeScalarFieldExtremas",
            [](const cloudViewer::GenericCloud *theCloud)
            {
                ScalarType minV{};
                ScalarType maxV{};
                cloudViewer::ScalarFieldTools::computeScalarFieldExtremas(theCloud, minV, maxV);
                return py::make_tuple(minV, maxV);
            },
            "theCloud"_a)
        .def_static(
            "countScalarFieldValidValues", &ScalarFieldTools::countScalarFieldValidValues, "theCloud"_a)
        .def_static("computeKmeans",
                    &ScalarFieldTools::computeKmeans,
                    "theCloud"_a,
                    "K"_a,
                    "kmcc"_a,
                    "progressCb"_a = nullptr)
        .def_static("SetScalarValueToNaN",
                    &cloudViewer::ScalarFieldTools::SetScalarValueToNaN,
                    "P"_a,
                    "scalarValue"_a)
        .def_static(
            "SetScalarValueInverted", &ScalarFieldTools::SetScalarValueInverted, "P"_a, "scalarValue"_a)
        .def_static(
            "SetScalarValueInverted", &ScalarFieldTools::SetScalarValueInverted, "P"_a, "scalarValue"_a);
}
