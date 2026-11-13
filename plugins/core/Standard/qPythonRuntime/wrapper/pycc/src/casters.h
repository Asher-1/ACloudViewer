// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QByteArray>
#include <QSharedPointer>
#include <QString>
#include <QVariant>
#include <QtCompat.h>

#include <CVLog.h>
#include <ecvColorScale.h>
#include <limits>

// clang-format off
#pragma push_macro("slots")
#undef slots
#include <pybind11/pybind11.h>
#include <pybind11/native_enum.h>
#include <Python.h>
#pragma pop_macro("slots")
// clang-format on

PYBIND11_DECLARE_HOLDER_TYPE(T, QSharedPointer<T>);
PYBIND11_MAKE_OPAQUE(QSharedPointer<ccColorScale>);

namespace pybind11
{
namespace detail
{
template <> struct type_caster<QString>
{
    PYBIND11_TYPE_CASTER(QString, _("QString"));

    bool load(handle src, bool)
    {
        if (!src || !PyUnicode_Check(src.ptr()))
            return false;

        Py_ssize_t size = 0;
        const char *str = PyUnicode_AsUTF8AndSize(src.ptr(), &size);
        if (!str)
            throw error_already_set();
        // Qt5 accepts an int length; check narrowing and preserve embedded NUL bytes.
        if (size > std::numeric_limits<int>::max())
            throw value_error("QString input exceeds the supported length");
        value = QString::fromUtf8(str, static_cast<int>(size));
        return true;
    }

    static handle cast(const QString &src, return_value_policy policy, handle parent)
    {
        const QByteArray byteArray = src.toUtf8();
        // TODO find how to use this error parameter.
        const char *errors = nullptr;
        PyObject *obj = PyUnicode_DecodeUTF8(byteArray.constData(), byteArray.size(), errors);
        return obj;
    }
};

template <> struct type_caster<QByteArray>
{
    PYBIND11_TYPE_CASTER(QByteArray, _("QByteArray"));

    bool load(handle src, bool)
    {
        if (!src || !isinstance<pybind11::bytes>(src))
        {
            return false;
        }

        char *buffer = nullptr;
        Py_ssize_t length = 0;
        if (PyBytes_AsStringAndSize(src.ptr(), &buffer, &length) == -1)
            throw error_already_set();
        // Qt5 uses int lengths; reject narrowing before copying binary payloads.
        if (length > std::numeric_limits<int>::max())
            throw value_error("QByteArray input exceeds the supported length");
        value = QByteArray(buffer, static_cast<int>(length));
        return true;
    }

    static handle cast(const QByteArray &src, return_value_policy /* policy */, handle /* handle */)
    {
        return PyBytes_FromStringAndSize(src.constData(), src.size());
    }
};

template <> struct type_caster<QVariant>
{
    PYBIND11_TYPE_CASTER(QVariant, _("QVariant"));

    bool load(handle src, bool)
    {
        if (!src)
        {
            return false;
        }
        if (src.is_none())
        {
            return true;
        }

        if (isinstance<pybind11::int_>(src))
        {
            value = src.cast<int>();
        }
        else if (isinstance<pybind11::str>(src))
        {
            value = QString::fromStdString(src.cast<std::string>());
        }
        else if (isinstance<pybind11::float_>(src))
        {
            value = src.cast<double>();
        }
        else
        {
            return false;
        }
        return true;
    }

    static handle cast(const QVariant &src, return_value_policy policy, handle handle)
    {
        pybind11::object h = none();
        // QMetaType ids are stable across Qt5/Qt6; the shared wrapper owns API differences.
        switch (static_cast<int>(qtCompatVariantType(src)))
        {
        case QMetaType::UnknownType:
            break;
        case QMetaType::Bool:
            h = pybind11::cast(src.toBool(), policy, handle);
            break;
        case QMetaType::Int:
            h = pybind11::cast(src.toInt(), policy, handle);
            break;
        case QMetaType::UInt:
            h = pybind11::cast(src.toUInt(), policy, handle);
            break;
        case QMetaType::LongLong:
            h = pybind11::cast(src.toLongLong(), policy, handle);
            break;
        case QMetaType::ULongLong:
            h = pybind11::cast(src.toULongLong(), policy, handle);
            break;
        case QMetaType::Double:
            h = pybind11::cast(src.toDouble(), policy, handle);
            break;
        case QMetaType::QChar:
            h = pybind11::cast(src.toChar(), policy, handle);
            break;
        case QMetaType::QString:
            h = pybind11::cast(src.toString(), policy, handle);
            break;
        case QMetaType::QVariantMap:
        case QMetaType::QVariantList:
        case QMetaType::QStringList:
        case QMetaType::QByteArray:
            // Preserve the established conversion at this value boundary.
            h = pybind11::cast(src.toByteArray(), policy, handle);
            break;
        default:
            throw std::runtime_error("Cannot convert this QVariant to python object");
        }
        return h.release();
    }
};

} // namespace detail
} // namespace pybind11
