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
#include <ecvDisplayTools.h>
#include <gtest/gtest.h>

#include <memory>

bool consumer_accepts_scalar_list(pybind11::handle value);
void consumer_mutates_scalar_container(pybind11::handle value);

namespace
{
namespace py = pybind11;
using Scalars = cloudViewer::GenericDistribution::ScalarContainer;

class BindingTypes : public ::testing::Test
{
  protected:
    static void SetUpTestSuite()
    {
        interpreter = std::make_unique<py::scoped_interpreter>();
        py::module_ module = py::module_::import("__main__");
        py::bind_vector<Scalars>(module, "ScalarContainer");
    }
    static void TearDownTestSuite()
    {
        interpreter.reset();
    }
    static std::unique_ptr<py::scoped_interpreter> interpreter;
};
std::unique_ptr<py::scoped_interpreter> BindingTypes::interpreter;

TEST_F(BindingTypes, OpaqueScalarContainerPreservesCrossTranslationUnitOwnership)
{
    py::list copiedValues;
    copiedValues.append(1.0);
    EXPECT_FALSE(consumer_accepts_scalar_list(copiedValues))
        << "ScalarContainer must not become a copied Python list";
    Scalars values{ScalarType(1.0)};
    py::object borrowed = py::cast(&values, py::return_value_policy::reference);
    consumer_mutates_scalar_container(borrowed);
    EXPECT_EQ(values.at(0), ScalarType(7.0)) << "Mutation must reach the C++ owner";
}

TEST_F(BindingTypes, QStringPreservesUtf8AndBytesPreserveEmbeddedNul)
{
    const QString input = QString::fromUtf8(u8"\u70b9\u4e91\u4e0e\u6807\u6ce8");
    EXPECT_EQ(py::cast(input).cast<QString>(), input);
    const QByteArray bytes("a\0b\xff", 4);
    EXPECT_EQ(py::cast(bytes).cast<QByteArray>(), bytes);
}

TEST_F(BindingTypes, QVariantPreservesSupportedValueAndNoneSemantics)
{
    const QString input = QString::fromUtf8(u8"\u70b9\u4e91\u4e0e\u6807\u6ce8");
    EXPECT_TRUE(py::cast(QVariant(true)).cast<bool>());
    EXPECT_EQ(py::cast(QVariant(42)).cast<int>(), 42);
    EXPECT_EQ(py::cast(QVariant(input)).cast<QString>(), input);
    EXPECT_TRUE(py::cast(QVariant()).is_none());
}
} // namespace
