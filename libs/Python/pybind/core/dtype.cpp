// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "core/Dtype.h"

#include "pybind/cloudViewer_pybind.h"
#include "pybind/core/core.h"
#include "pybind/docstring.h"

namespace cloudViewer {
namespace core {

void pybind_core_dtype(py::module &m) {
    // cloudViewer.core.Dtype class
    py::class_<Dtype, std::shared_ptr<Dtype>> dtype(m, "Dtype",
                                                    "CloudViewer data types.");
    py::native_enum<Dtype::DtypeCode>(dtype, "DtypeCode", "enum.Enum")
            .value("Undefined", Dtype::DtypeCode::Undefined)
            .value("Bool", Dtype::DtypeCode::Bool)
            .value("Int", Dtype::DtypeCode::Int)
            .value("UInt", Dtype::DtypeCode::UInt)
            .value("Float", Dtype::DtypeCode::Float)
            .value("Object", Dtype::DtypeCode::Object)
            .finalize();

    dtype.def(py::init<Dtype::DtypeCode, int64_t, const std::string &>());
    dtype.def_readonly_static("Undefined", &core::Undefined);
    dtype.def_readonly_static("Float32", &core::Float32);
    dtype.def_readonly_static("Float64", &core::Float64);
    dtype.def_readonly_static("Int8", &core::Int8);
    dtype.def_readonly_static("Int16", &core::Int16);
    dtype.def_readonly_static("Int32", &core::Int32);
    dtype.def_readonly_static("Int64", &core::Int64);
    dtype.def_readonly_static("UInt8", &core::UInt8);
    dtype.def_readonly_static("UInt16", &core::UInt16);
    dtype.def_readonly_static("UInt32", &core::UInt32);
    dtype.def_readonly_static("UInt64", &core::UInt64);
    dtype.def_readonly_static("Bool", &core::Bool);
    dtype.def("byte_size", &Dtype::ByteSize);
    dtype.def("byte_code", &Dtype::GetDtypeCode);
    dtype.def("__eq__", &Dtype::operator==);
    dtype.def("__hash__", [](const Dtype &dt) {
        using DtypeTuple = std::tuple<size_t, size_t, std::string>;
        return utility::hash_tuple<DtypeTuple>()(
                std::make_tuple(static_cast<size_t>(dt.GetDtypeCode()),
                                dt.ByteSize(), dt.ToString()));
    });
    dtype.def("__ene__", &Dtype::operator!=);
    dtype.def("__repr__", &Dtype::ToString);
    dtype.def("__str__", &Dtype::ToString);

    // Dtype shortcuts.
    // E.g. cloudViewer.core.float32
    m.attr("undefined") = &core::Undefined;
    m.attr("float32") = core::Float32;
    m.attr("float64") = core::Float64;
    m.attr("int8") = core::Int8;
    m.attr("int16") = core::Int16;
    m.attr("int32") = core::Int32;
    m.attr("int64") = core::Int64;
    m.attr("uint8") = core::UInt8;
    m.attr("uint16") = core::UInt16;
    m.attr("uint32") = core::UInt32;
    m.attr("uint64") = core::UInt64;
    m.attr("bool") = core::Bool;
    m.attr("bool8") = core::Bool;
}

}  // namespace core
}  // namespace cloudViewer
