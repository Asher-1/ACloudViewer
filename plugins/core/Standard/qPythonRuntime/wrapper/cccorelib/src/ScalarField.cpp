// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>

#include "wrappers.h"

#include <ScalarField.h>
#include <cstring>
#include <limits>
#include <memory>
#include <stdio.h>
#include <unordered_map>

namespace py = pybind11;
using namespace pybind11::literals;

static py::array_t<ScalarType> CopyScalarFieldArray(const cloudViewer::ScalarField &field)
{
    const auto size = field.size();
    if (size > static_cast<size_t>(std::numeric_limits<py::ssize_t>::max()))
        throw py::value_error("Scalar field exceeds the supported array length");
    // Allocate first, then reacquire native storage; never export a vector pointer.
    // O(n * sizeof(ScalarType)) bytes, with no Python objects per element.
    py::array_t<ScalarType> result(static_cast<py::ssize_t>(size));
    if (field.size() != size)
        throw py::value_error("Scalar field size changed while creating an array");
    if (size)
        std::copy_n(field.data(), size, result.mutable_data());
    return result;
}

class ScalarFieldView
{
  public:
    explicit ScalarFieldView(cloudViewer::ScalarField &field) : m_size(field.size())
    {
        if (m_size > static_cast<size_t>(std::numeric_limits<py::ssize_t>::max()))
            throw py::value_error("Scalar field exceeds the supported array length");
        m_count = static_cast<py::ssize_t>(m_size);
        field.link();
        m_owner = std::shared_ptr<cloudViewer::ScalarField>(
            &field, [](cloudViewer::ScalarField *owner) { owner->release(); });
    }

    py::object get_item(const py::object &raw_index, bool python_value = false) const
    {
        if (m_indices.empty())
        {
            if (const auto index = basic_slice_index(raw_index))
                return py::cast(slice(index));
        }
        const auto index = freeze_index(raw_index);
        if (m_indices.empty() && is_integer(index))
        {
            const auto position = integer(index);
            // O(1): resolve Python indices before reading the current native value.
            const auto value = m_owner->at(m_start + position * m_step);
            if (python_value)
                return py::float_(value);
            return py::dtype::of<ScalarType>().attr("type")(value);
        }
        auto data = selected_local(snapshot());
        py::object result = data.attr("__getitem__")(index);
        if (py::isinstance<py::array>(result) &&
            (py::isinstance<py::slice>(index) ||
             py::module_::import("numpy").attr("shares_memory")(data, result).cast<bool>()))
        {
            ScalarFieldView child(*this);
            py::tuple indices(m_indices.size() + 1);
            for (py::ssize_t i = 0; i < m_indices.size(); ++i)
                indices[i] = m_indices[i];
            indices[m_indices.size()] = index;
            child.m_indices = std::move(indices);
            return py::cast(std::move(child));
        }
        return result;
    }

    void set_item(const py::object &raw_index, const py::object &value) const
    {
        if (m_indices.empty())
        {
            if (const auto index = basic_slice_index(raw_index))
            {
                slice(index).assign(value);
                return;
            }
        }
        const auto index = freeze_index(raw_index);
        if (m_indices.empty() && is_integer(index))
        {
            ScalarFieldView target(*this);
            target.m_start += integer(index) * m_step;
            target.m_count = 1;
            target.m_step = 1;
            target.assign(value);
            return;
        }
        // Resolve arbitrary indexing once with NumPy, including callback indices.
        auto np = py::module_::import("numpy");
        auto positions =
            selected_local(np.attr("arange")(m_count).cast<py::array>()).attr("__getitem__")(index);
        auto before = snapshot();
        auto work = before.attr("copy")().cast<py::array_t<ScalarType>>();
        work.attr("__setitem__")(positions, value);
        py::array_t<bool> mask(m_count);
        mask.attr("fill")(false);
        mask.attr("__setitem__")(positions, true);
        check_commit(work, before, mask.data());
        write(work, mask.data());
    }

    py::object array(const py::object &dtype, const py::object &copy) const
    {
        if (!copy.is_none() && !copy.cast<bool>())
            throw py::value_error("A scalar field cannot export a safe zero-copy ndarray");
        auto data = selected_local(snapshot());
        py::object result = m_indices.empty() ? py::object(data) : data.attr("copy")();
        if (!dtype.is_none())
            result = result.attr("astype")(dtype, "copy"_a = false);
        return result;
    }

    py::ssize_t len() const
    {
        validate();
        return m_indices.empty() ? m_count : py::len(selected_local(snapshot()));
    }

    py::object array_ufunc(const py::object &ufunc,
                           const std::string &method,
                           const py::args &inputs,
                           const py::kwargs &kwargs) const
    {
        // Follow NumPy's dispatch contract before exposing any private work arrays.
        auto np = py::module_::import("numpy");
        if (!py::isinstance(ufunc, np.attr("ufunc")))
            return py::reinterpret_borrow<py::object>(Py_NotImplemented);
        for (py::ssize_t i = 0; i < inputs.size(); ++i)
            if (!(method == "at" && i == 1 ? supported_index(inputs[i]) : supported(inputs[i])))
                return py::reinterpret_borrow<py::object>(Py_NotImplemented);
        py::tuple outputs;
        if (kwargs.contains("out"))
        {
            outputs = kwargs["out"].cast<py::tuple>();
            for (auto output : outputs)
                if (!supported(output))
                    return py::reinterpret_borrow<py::object>(Py_NotImplemented);
        }
        if (kwargs.contains("where") && !supported(kwargs["where"]))
            return py::reinterpret_borrow<py::object>(Py_NotImplemented);

        struct WorkArray
        {
            ScalarFieldView field;
            py::array_t<ScalarType> before;
            py::array_t<ScalarType> root;
            py::array_t<bool> mask;
            bool writes = false;
            bool dense = false;
        };
        std::unordered_map<cloudViewer::ScalarField *, WorkArray> work;
        auto include = [&work](py::handle value, bool output)
        {
            if (!py::isinstance<ScalarFieldView>(value))
                return;
            const auto &view = value.cast<const ScalarFieldView &>();
            view.validate();
            // Basic views have bounded endpoints; sparse/distant aliases may still span N.
            const auto last = view.m_count ? view.m_start + (view.m_count - 1) * view.m_step : 0;
            const auto begin = view.m_count ? std::min(view.m_start, last) : 0;
            const auto end = view.m_count ? std::max(view.m_start, last) + 1 : 0;
            auto found = work.find(view.m_owner.get());
            if (found == work.end())
            {
                ScalarFieldView field(view);
                field.m_indices = py::tuple();
                field.m_start = begin;
                field.m_step = 1;
                field.m_count = end - begin;
                found = work.emplace(view.m_owner.get(), WorkArray{std::move(field)}).first;
            }
            else if (end > begin)
            {
                auto &field = found->second.field;
                const auto stop = field.m_count ? std::max(field.m_start + field.m_count, end) : end;
                field.m_start = field.m_count ? std::min(field.m_start, begin) : begin;
                field.m_count = stop - field.m_start;
            }
            found->second.writes |= output;
        };
        for (py::ssize_t i = 0; i < inputs.size(); ++i)
            include(inputs[i], method == "at" && i == 0);
        for (auto output : outputs)
            include(output, true);
        if (kwargs.contains("where"))
            include(kwargs["where"], false);
        // A continuous output covering W makes every element a target; no mask is needed.
        // Partial outputs, where controls and at operations retain their explicit masks.
        if (method == "__call__" && !kwargs.contains("where"))
            for (auto output : outputs)
                if (py::isinstance<ScalarFieldView>(output))
                {
                    const auto &view = output.cast<const ScalarFieldView &>();
                    auto &entry = work.at(view.m_owner.get());
                    entry.dense |= view.m_indices.empty() && view.m_step == 1 &&
                                   view.m_start == entry.field.m_start && view.m_count == entry.field.m_count;
                }
        for (auto &item : work)
        {
            auto &entry = item.second;
            // W * sizeof(ScalarType) bytes for reads; only writers need preflight and masks.
            entry.root = entry.field.snapshot();
            if (entry.writes)
            {
                entry.before = entry.root.attr("copy")().cast<py::array_t<ScalarType>>();
                if (!entry.dense)
                {
                    entry.mask = py::array_t<bool>(entry.field.m_count);
                    entry.mask.attr("fill")(false);
                }
            }
        }
        auto unwrap = [&work](py::handle value) -> py::object
        {
            if (!py::isinstance<ScalarFieldView>(value))
                return py::reinterpret_borrow<py::object>(value);
            const auto &view = value.cast<const ScalarFieldView &>();
            const auto &entry = work.at(view.m_owner.get());
            return view.selected(entry.root, entry.field.m_start);
        };
        struct ControlInput
        {
            py::object source;
            py::array before;
        };
        std::vector<ControlInput> controls;
        auto freeze_control = [&np, &controls](auto &&self, py::handle value) -> py::object
        {
            if (py::isinstance<py::tuple>(value))
            {
                auto source = py::reinterpret_borrow<py::tuple>(value);
                py::tuple frozen(source.size());
                for (py::ssize_t i = 0; i < source.size(); ++i)
                    frozen[i] = self(self, source[i]);
                return frozen;
            }
            if (py::isinstance<py::array>(value) || py::isinstance<py::list>(value))
            {
                auto before = np.attr("asarray")(value).attr("copy")().cast<py::array>();
                controls.push_back({py::reinterpret_borrow<py::object>(value), before});
                return before;
            }
            return freeze_index(py::reinterpret_borrow<py::object>(value));
        };
        py::tuple arguments(inputs.size());
        for (py::ssize_t i = 0; i < inputs.size(); ++i)
            arguments[i] = method == "at" && i == 1 ? freeze_control(freeze_control, unwrap(inputs[i]))
                                                    : unwrap(inputs[i]);
        if (method == "at" && inputs.size() > 1 && py::isinstance<ScalarFieldView>(inputs[0]))
        {
            const auto &view = inputs[0].cast<const ScalarFieldView &>();
            auto &entry = work.at(view.m_owner.get());
            view.selected(entry.mask, entry.field.m_start).attr("__setitem__")(arguments[1], true);
        }
        py::dict options(kwargs);
        if (options.contains("out"))
        {
            py::tuple converted(outputs.size());
            for (py::ssize_t i = 0; i < outputs.size(); ++i)
                converted[i] = unwrap(outputs[i]);
            options["out"] = std::move(converted);
        }
        if (options.contains("where"))
        {
            auto source = unwrap(options["where"]);
            auto frozen = freeze_control(freeze_control, source);
            // NumPy coerces sequences to bool, but rejects integer ndarray where controls.
            if (py::isinstance<py::list>(source) || py::isinstance<py::tuple>(source))
                frozen = np.attr("asarray")(frozen, "dtype"_a = py::dtype::of<bool>());
            options["where"] = std::move(frozen);
        }
        for (auto output : outputs)
        {
            if (!py::isinstance<ScalarFieldView>(output))
                continue;
            const auto &view = output.cast<const ScalarFieldView &>();
            auto &entry = work.at(view.m_owner.get());
            if (entry.dense)
                continue;
            auto target = view.selected(entry.mask, entry.field.m_start);
            // Reduction where filters inputs, not the destination of the reduced result.
            if (method == "__call__" && options.contains("where"))
                np.attr("logical_or")(target, options["where"], "out"_a = target);
            else
                target.attr("fill")(true);
        }

        py::object result = ufunc.attr(method.c_str())(*arguments, **options);
        for (const auto &control : controls)
        {
            auto current = np.attr("asarray")(control.source).attr("copy")().cast<py::array>();
            if (!current.dtype().is(control.before.dtype()) || current.ndim() != control.before.ndim() ||
                !std::equal(current.shape(), current.shape() + current.ndim(), control.before.shape()) ||
                current.nbytes() != control.before.nbytes() ||
                std::memcmp(current.data(), control.before.data(), current.nbytes()) != 0)
                throw py::value_error("NumPy control input changed during scalar field transaction");
        }
        // No Python callbacks or allocations are permitted between preflight and writes.
        // Validate every destination before modifying even the first one.
        for (const auto &item : work)
            if (item.second.writes)
                item.second.field.check_commit(item.second.root,
                                               item.second.before,
                                               item.second.dense ? nullptr : item.second.mask.data());
        for (const auto &item : work)
            if (item.second.writes)
                item.second.field.write(item.second.root,
                                        item.second.dense ? nullptr : item.second.mask.data());
        if (outputs.size() == 1 && !outputs[0].is_none())
            return py::reinterpret_borrow<py::object>(outputs[0]);
        if (outputs.size() > 1)
        {
            auto results = result.cast<py::tuple>();
            py::tuple returned(outputs.size());
            for (py::ssize_t i = 0; i < outputs.size(); ++i)
                returned[i] = outputs[i].is_none() ? results[i] : outputs[i];
            return returned;
        }
        return result;
    }

  private:
    void validate() const
    {
        if (m_owner->size() != m_size)
            throw py::value_error("Scalar field size changed; obtain a new asArray() view");
    }

    static py::handle basic_slice_index(py::handle index)
    {
        // Native basic slices retain only resolved offsets, so no frozen index copy is needed.
        if (PySlice_Check(index.ptr()))
            return index;
        if (PyTuple_Check(index.ptr()) && PyTuple_GET_SIZE(index.ptr()) == 1)
        {
            auto *item = PyTuple_GET_ITEM(index.ptr(), 0);
            if (PySlice_Check(item))
                return py::handle(item);
        }
        return py::handle();
    }

    static py::object freeze_index(const py::object &index)
    {
        // Builtin integers and non-indexing sentinels need no callbacks or normalization.
        if (PyLong_CheckExact(index.ptr()) || index.is_none() || index.is(py::ellipsis()))
            return index;
        if (PySlice_Check(index.ptr()))
        {
            const auto *value = reinterpret_cast<const PySliceObject *>(index.ptr());
            auto builtin_bound = [](PyObject *bound) { return bound == Py_None || PyLong_CheckExact(bound); };
            // An immutable builtin slice has no callbacks to freeze; reuse it without allocation.
            if (builtin_bound(value->start) && builtin_bound(value->stop) && builtin_bound(value->step))
                return index;
        }
        if (Py_EnterRecursiveCall(" while normalizing NumPy indices"))
            throw py::error_already_set();
        struct RecursionGuard
        {
            ~RecursionGuard()
            {
                Py_LeaveRecursiveCall();
            }
        } guard;
        if (py::isinstance<py::slice>(index))
        {
            Py_ssize_t start, stop, step;
            // Use the native slice protocol for callback order, clipping and zero-step errors.
            if (PySlice_Unpack(index.ptr(), &start, &stop, &step) < 0)
                throw py::error_already_set();
            const auto *value = reinterpret_cast<const PySliceObject *>(index.ptr());
            return py::slice(value->start == Py_None ? py::object(py::none()) : py::object(py::int_(start)),
                             value->stop == Py_None ? py::object(py::none()) : py::object(py::int_(stop)),
                             value->step == Py_None ? py::object(py::none()) : py::object(py::int_(step)));
        }
        if (py::isinstance<py::tuple>(index))
        {
            auto source = index.cast<py::tuple>();
            py::tuple frozen(source.size());
            for (py::ssize_t i = 0; i < source.size(); ++i)
                frozen[i] = freeze_index(py::reinterpret_borrow<py::object>(source[i]));
            return frozen;
        }
        if (is_integer(index))
        {
            auto frozen = py::reinterpret_steal<py::object>(PyNumber_Index(index.ptr()));
            if (!frozen)
                throw py::error_already_set();
            return frozen;
        }
        return index;
    }

    static bool supported_index(py::handle index)
    {
        // Index arrays must retain the same foreign-dispatch boundary as numeric arrays.
        if (py::isinstance<py::array>(index))
            return supported(index);
        if (index.is_none() || index.is(py::ellipsis()) || py::isinstance<py::slice>(index) ||
            is_integer(index))
            return true;
        if (py::isinstance<py::tuple>(index))
        {
            if (Py_EnterRecursiveCall(" while validating NumPy indices"))
                throw py::error_already_set();
            struct RecursionGuard
            {
                ~RecursionGuard()
                {
                    Py_LeaveRecursiveCall();
                }
            } guard;
            for (auto item : py::reinterpret_borrow<py::tuple>(index))
                if (!supported_index(item))
                    return false;
            return true;
        }
        return supported(index);
    }

    static bool is_integer(py::handle value)
    {
        if (PyLong_Check(value.ptr()))
            return !PyBool_Check(value.ptr());
        if (py::isinstance<py::array>(value))
        {
            const auto array = py::reinterpret_borrow<py::array>(value);
            const auto kind = array.dtype().kind();
            // NumPy integer scalars represented as 0-D arrays still address one native value.
            return array.ndim() == 0 && (kind == 'i' || kind == 'u');
        }
        return PyIndex_Check(value.ptr()) && !PyBool_Check(value.ptr()) &&
               !py::isinstance(value, py::module_::import("numpy").attr("bool_"));
    }

    py::ssize_t integer(py::handle index) const
    {
        auto value = py::reinterpret_steal<py::object>(PyNumber_Index(index.ptr()));
        if (!value)
            throw py::error_already_set();
        auto position = PyLong_AsSsize_t(value.ptr());
        if (position == -1 && PyErr_Occurred())
        {
            if (!PyErr_ExceptionMatches(PyExc_OverflowError))
                throw py::error_already_set();
            PyErr_Clear();
            throw py::index_error("Scalar field index is out of range");
        }
        if (position < 0)
            position += m_count;
        if (position < 0 || position >= m_count)
            throw py::index_error("Scalar field index is out of range");
        validate();
        return position;
    }

    ScalarFieldView slice(py::handle index) const
    {
        py::ssize_t start, stop, step, count;
        if (!py::reinterpret_borrow<py::slice>(index).compute(m_count, &start, &stop, &step, &count))
            throw py::error_already_set();
        validate();
        ScalarFieldView result(*this);
        // Canonicalize empty/singleton slices to avoid irrelevant stride overflow.
        result.m_start = count ? m_start + start * m_step : 0;
        result.m_step = count > 1 ? m_step * step : 1;
        result.m_count = count;
        return result;
    }

    static bool supported(py::handle value)
    {
        if (value.is_none() || py::isinstance<ScalarFieldView>(value))
            return true;
        auto np = py::module_::import("numpy");
        if (py::isinstance<py::array>(value))
            return reinterpret_cast<PyObject *>(Py_TYPE(value.ptr())) == np.attr("ndarray").ptr();
        if (py::hasattr(value, "__array_ufunc__"))
            return false;
        if (PyLong_Check(value.ptr()) || PyFloat_Check(value.ptr()) || PyComplex_Check(value.ptr()) ||
            py::isinstance(value, np.attr("generic")))
            return true;
        if (py::isinstance<py::list>(value) || py::isinstance<py::tuple>(value))
        {
            // Honor Python's recursion limit for cyclic/nested operand containers.
            if (Py_EnterRecursiveCall(" while validating NumPy operands"))
                throw py::error_already_set();
            struct RecursionGuard
            {
                ~RecursionGuard()
                {
                    Py_LeaveRecursiveCall();
                }
            } guard;
            for (auto item : py::reinterpret_borrow<py::iterable>(value))
                if (!supported(item))
                    return false;
            return true;
        }
        return false;
    }

    py::array_t<ScalarType> snapshot() const
    {
        validate();
        py::array_t<ScalarType> result(m_count);
        validate();
        // Only gather storage: NumPy still owns numerical operations and broadcasting.
        // O(k) time and k * sizeof(ScalarType) bytes, even for a reversed short slice.
        auto *destination = result.mutable_data();
        const auto *source = m_owner->data();
        for (py::ssize_t i = 0; i < m_count; ++i)
            destination[i] = source[m_start + i * m_step];
        return result;
    }

    py::array selected_local(const py::array &root) const
    {
        py::object result = root;
        for (auto index : m_indices)
            result = result.attr("__getitem__")(index);
        return result.cast<py::array>();
    }

    py::array selected(const py::array &root, py::ssize_t offset = 0) const
    {
        const auto *data = static_cast<const char *>(root.data());
        py::array selection(root.dtype(),
                            {m_count},
                            {m_step * root.itemsize()},
                            m_count ? data + (m_start - offset) * root.itemsize() : data,
                            root);
        return selected_local(selection);
    }

    void check_commit(const py::array_t<ScalarType> &data,
                      const py::array_t<ScalarType> &before,
                      const bool *mask = nullptr) const
    {
        validate();
        // Inspect actual metadata, not the array_t template retained across callbacks.
        const auto dtype = data.dtype();
        const auto native_dtype = py::dtype::of<ScalarType>();
        // Identity with NumPy's native builtin descriptor proves type and byte order.
        // Distinct descriptors retain the full original checks, including explicit native byte order.
        if (data.ndim() != 1 || data.shape(0) != m_count || data.itemsize() != sizeof(ScalarType) ||
            (!dtype.is(native_dtype) &&
             (dtype.num() != native_dtype.num() || !dtype.attr("isnative").cast<bool>())) ||
            !(data.flags() & py::array::c_style) || (m_count && data.strides(0) != sizeof(ScalarType)))
            throw py::value_error("NumPy work array layout changed before scalar field commit");
        const auto *original = before.data();
        const auto *current = m_owner->data();
        for (py::ssize_t i = 0; i < m_count; ++i)
            if ((!mask || mask[i]) &&
                std::memcmp(current + m_start + i * m_step, original + i, sizeof(ScalarType)) != 0)
                throw py::value_error("Scalar field write target changed during NumPy conversion");
    }

    void write(const py::array_t<ScalarType> &data, const bool *mask = nullptr) const
    {
        auto *destination = m_owner->data();
        const auto *source = data.data();
        for (py::ssize_t i = 0; i < m_count; ++i)
            if (!mask || mask[i])
                destination[m_start + i * m_step] = source[i];
    }

    void assign(const py::object &value) const
    {
        auto before = snapshot();
        auto work = before.attr("copy")().cast<py::array_t<ScalarType>>();
        work.attr("__setitem__")(py::ellipsis(), value);
        check_commit(work, before);
        write(work);
    }

    std::shared_ptr<cloudViewer::ScalarField> m_owner;
    size_t m_size;
    py::ssize_t m_start = 0;
    py::ssize_t m_step = 1;
    py::ssize_t m_count = 0;
    py::tuple m_indices;
};

void define_ScalarField(py::module &cccorelib)
{

    auto mixin = py::module_::import("numpy.lib.mixins").attr("NDArrayOperatorsMixin");
    auto view_type = py::class_<ScalarFieldView>(cccorelib, "ScalarFieldView");
    view_type.def("__len__", &ScalarFieldView::len)
        .def("__getitem__",
             [](const ScalarFieldView &view, const py::object &index) { return view.get_item(index); })
        .def("__setitem__", &ScalarFieldView::set_item)
        .def("__array__", &ScalarFieldView::array, "dtype"_a = py::none(), "copy"_a = py::none())
        .def("__array_ufunc__", &ScalarFieldView::array_ufunc)
        .def("__bool__",
             [](const ScalarFieldView &view)
             { return view.array(py::none(), py::none()).attr("__bool__")().cast<bool>(); })
        .def_property_readonly("dtype", [](const ScalarFieldView &) { return py::dtype::of<ScalarType>(); });
    // Reuse NumPy's operators directly: a Python mixin is not a registered C++ base.
    for (auto entry : mixin.attr("__dict__").attr("items")())
    {
        auto item = entry.cast<py::tuple>();
        if (PyCallable_Check(item[1].ptr()))
            view_type.attr(item[0].cast<std::string>().c_str()) = item[1];
    }

    py::class_<cloudViewer::ScalarField, CCShareable, CCShareableHolder<cloudViewer::ScalarField>>(
        cccorelib,
        "ScalarField",
        R"doc(
    ScalarField

    .. note::
        Scalar storage uses the core build's ScalarType (float32 or float64).
        NumPy operations on asArray() preserve that dtype unless explicitly cast.

    Getting / Setting values can be done via :meth:`cccorelib.ScalarField.getValue`
    or :meth:`cccorelib.ScalarField.setValue`.
    Alternatively, it is possible to use bracket operator.

    Use :meth:`cccorelib.ScalarField.asArray` to be able to use the
    scalar field as a normal numpy array.


    Example
    -------

    .. code:: Python

        scalar_field = cccorelib.ScalarField("Codification")
)doc")
        .def_static("NaN", &cloudViewer::ScalarField::NaN)

        .def(py::init<const char *>(), "name"_a, R"doc(
    Creates a ScalarField with the given name

    Example
    -------

    .. code:: Python

        scalar_field = cccorelib.ScalarField("Codification")
        scalar_field.resize(3)

        scalar_field.setValue(1, 1.0)
        scalar_field[2] = 2.0

        assert scalar_field.getValue(0) == scalar_field[0]
        assert scalar_field.getValue(1) == scalar_field[1]
        assert scalar_field.getValue(2) == scalar_field[2]
)doc")
        .def("setName", &cloudViewer::ScalarField::setName, R"doc(
    Sets the name of the scalar field

    >>> import cccorelib
    >>> sf = cccorelib.ScalarField("name")
    >>> sf.setName("other_name")
    >>> sf.getName() == "other_name"
    True
)doc")
        .def("getName", &cloudViewer::ScalarField::getName, R"doc(
    Returns the name of the scalar field

    >>> import cccorelib
    >>> sf = cccorelib.ScalarField("name")
    >>> sf.getName()
    'name'
 )doc")
        .def("size",
             &cloudViewer::ScalarField::currentSize,
             "Returns the number of elements (values) in the scalar field")
        .def("computeMeanAndVariance",
             &cloudViewer::ScalarField::computeMeanAndVariance,
             "mean"_a,
             "variance"_a = nullptr)
        .def("computeMinAndMax", &cloudViewer::ScalarField::computeMinAndMax, R"doc(
    Computes the Min and Max, this needs to be called before retrieving the values
    with getMin or getMax
)doc")
        .def("getMin", &cloudViewer::ScalarField::getMin, R"doc(
    Returns the lowest value in the scalar field

    You need to call :meth:`cccorelib.ScalarField.computeMinAndMax` if the array has changed
    since last call.
)doc")
        .def("getMax", &cloudViewer::ScalarField::getMax, R"doc(
    Returns the highest value in the scalar field

    You need to call :meth:`cccorelib.ScalarField.computeMinAndMax` if the array has changed
    since last call.
)doc")
        .def_static("ValidValue", &cloudViewer::ScalarField::ValidValue, "value"_a)
        .def("flagValueAsInvalid", &cloudViewer::ScalarField::flagValueAsInvalid, "index"_a)
        .def("fill",
             &cloudViewer::ScalarField::fill,
             "fillValue"_a = 0,
             "Fills the scalar field with the given value")
        .def(
            "reserve",
            [](cloudViewer::ScalarField &field, const size_t size) { field.reserve(size); },
            "count"_a,
            R"doc(
    Reserves space for ``count`` element, but does not change the size.

    Will raise an exception if allocation failed
)doc")
        .def("reserveSafe", &cloudViewer::ScalarField::reserveSafe, "count"_a, R"doc(
    Reserves space for ``count`` element, but does not change the size.

    Will NOT raise an exception if allocation failed.
    Instead, it returns a bool to indicate success.

    Prefer use of :meth:`cccorelib.ScalarField.reserve`.
)doc")
        .def(
            "resize",
            [](cloudViewer::ScalarField &self, const size_t count, const ScalarType valueForNewElements)
            {
                // pybind11 will convert exceptions
                self.resizeSafe(count, true /* always init values */, valueForNewElements);
            },
            "count"_a,
            "valueForNewElements"_a = 0,
            R"doc(
    Resize the scalar field

    Will raise an exception if allocation failed
)doc")
        .def("resizeSafe",
             &cloudViewer::ScalarField::resizeSafe,
             "count"_a,
             "initNewElements"_a = false,
             "valueForNewElements"_a = 0,
             R"doc(
    Resize the scalar field

    Will NOT raise an exception if allocation failed.
    Instead, it returns a bool to indicate success.

    Prefer use of :meth:`cccorelib.ScalarField.resize`.
)doc")

        .def(
            "getValue",
            [](cloudViewer::ScalarField &self, const size_t index) { return self.getValue(index); },
            "index"_a,
            R"doc(
    Returns the value at the given index.

    Only supports index in [0..self.size()[

    Raises
    ------
    IndexError on invalid index
)doc")
        .def("setValue",
             &cloudViewer::ScalarField::setValue,
             "index"_a,
             "value"_a,
             R"doc(
    Sets the value at the given index.

    Only supports index in [0..self.size()[

    Raises
    ------
    IndexError on invalid index
)doc")
        .def("addElement", &cloudViewer::ScalarField::addElement, "value"_a, R"doc(
    Appends a value

    Example
    -------
    >>> import cccorelib
    >>> sf = cccorelib.ScalarField("name")
    >>> sf.size()
    0
    >>> sf.addElement(1)
    >>> sf.size()
    1
)doc")
        .def(
            "asArray",
            [](cloudViewer::ScalarField &self) { return ScalarFieldView(self); },
            R"doc(
    Returns a writable NumPy-compatible scalar field view.

    Operators, indexing, broadcasting and ufuncs are implemented by NumPy.
    Changes through this view are written back to the scalar field, and later
    operations see field changes. Work arrays own their storage, so Python
    callbacks and vector reallocation cannot leave exported dangling pointers.
    np.asarray(view) returns an independent snapshot in the core ScalarType;
    copy=False is rejected. Views reject access when the current field size differs
    from its creation size; obtain a new view after resizing. Concurrent native
    field mutation is unsupported.

    .. code:: Python

        scalar_field = cccorelib.ScalarField("Codification")
        scalar_field.resize(10)

        array = scalar_field.asArray()

        assert np.all(array == 0)
        # Changes made to the scalar_field reflects on the array
        scalar_field.fill(1.0)
        assert np.all(array == 1.0)

        # and vice-versa
        array[:] = 2.0
        assert scalar_field[0] == 2.0
)doc")
        .def(
            "__getitem__",
            [](cloudViewer::ScalarField &self, const py::object &index)
            { return ScalarFieldView(self).get_item(index, true); },
            py::return_value_policy::reference_internal)
        .def(
            "toArray",
            [](const cloudViewer::ScalarField &self)
            { return CopyScalarFieldArray(self).attr("astype")(py::dtype::of<double>(), "copy"_a = false); },
            R"doc(
        Copies the content into an array
)doc")
        .def(
            "__array__",
            [](cloudViewer::ScalarField &self, const py::object &dtype, const py::object &copy) {
                return ScalarFieldView(self).array(
                    dtype.is_none() ? py::object(py::dtype::of<double>()) : dtype, copy);
            },
            "dtype"_a = py::none(),
            "copy"_a = py::none(),
            R"doc(
        Returns an independent NumPy array; default dtype is float64 for compatibility.
        An explicit dtype is honored. copy=False is rejected for resizable storage.
)doc")
        .def("__setitem__",
             [](cloudViewer::ScalarField &self, const py::object &index, const py::object &value)
             { return ScalarFieldView(self).set_item(index, value); })
        .def("__len__", &cloudViewer::ScalarField::currentSize)
        .def("__repr__",
             [](const cloudViewer::ScalarField &self)
             { return std::string("<ScalarField(name=") + self.getName() + ")>"; });
}
