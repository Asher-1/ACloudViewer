// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Check real cross-translation-unit conversion and mutation, not template names.
#include "../wrapper/cccorelib/src/cccorelib.h"
#include "../wrapper/pycc/src/casters.h"
#include <pybind11/embed.h>
#include <pybind11/stl_bind.h>
// The shared converter header must preserve slots for actual Qt consumer headers.
#include <cstdio>
#include <ecvDisplayTools.h>

bool consumer_accepts_scalar_list(pybind11::handle value);
void consumer_mutates_scalar_container(pybind11::handle value);

int main()
{
    namespace py = pybind11;
    py::scoped_interpreter interpreter;
    py::module_ module = py::module_::import("__main__");
    using Scalars = cloudViewer::GenericDistribution::ScalarContainer;
    py::bind_vector<Scalars>(module, "ScalarContainer");
    py::list copied_values;
    copied_values.append(1.0);
    if (consumer_accepts_scalar_list(copied_values))
    {
        std::fprintf(
            stderr,
            "FAIL: ScalarContainer became a copied Python list in another translation unit\n");
        return 1;
    }
    Scalars values{ScalarType(1.0)};
    py::object borrowed = py::cast(&values, py::return_value_policy::reference);
    consumer_mutates_scalar_container(borrowed);
    if (values.at(0) != ScalarType(7.0))
    {
        std::fprintf(stderr, "FAIL: ScalarContainer mutation did not reach its C++ owner\n");
        return 1;
    }
    const QString input = QString::fromUtf8(u8"\u70b9\u4e91\u4e0e\u6807\u6ce8");
    if (py::cast(input).cast<QString>() != input)
    {
        std::fprintf(stderr, "FAIL: QString UTF-8 conversion changed\n");
        return 1;
    }
    const QByteArray bytes("a\0b\xff", 4);
    if (py::cast(bytes).cast<QByteArray>() != bytes)
    {
        std::fprintf(stderr, "FAIL: QByteArray conversion changed binary or NUL data\n");
        return 1;
    }
    if (!py::cast(QVariant(true)).cast<bool>() || py::cast(QVariant(42)).cast<int>() != 42 ||
        py::cast(QVariant(input)).cast<QString>() != input || !py::cast(QVariant()).is_none())
    {
        std::fprintf(stderr, "FAIL: QVariant conversion changed supported value semantics\n");
        return 1;
    }
    std::puts(
        "PASS: cross-translation-unit opaque reference, QString, bytes and QVariant conversion");
    return 0;
}
