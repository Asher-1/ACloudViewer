// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef PYTHON_PLUGIN_WRAPPERS_H
#define PYTHON_PLUGIN_WRAPPERS_H

#include <CVShareable.h>
#include <CVTypes.h>
#include <PointCloudTpl.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <stdexcept>
#include <type_traits>

namespace py = pybind11;
using namespace pybind11::literals;

/// A unique_ptr that never free its ptr.
template <class T> using observer_ptr = std::unique_ptr<T, py::nodelete>;

/// A holder type for any type that inherits CCShareable
///
/// CCShareable is CC's ref counted mechanism.
/// It works by inheritance + manual call to link/release.
template <class T> class CCShareableHolder
{
  public:
    static_assert(std::is_base_of<CCShareable, T>::value == true, "T must be a subclass of CCShareable");

    CCShareableHolder() = default;

    explicit CCShareableHolder(T *obj) : m_ptr(obj)
    {
        if (m_ptr)
        {
            m_ptr->link();
        }
    }

    T *get()
    {
        return m_ptr;
    }

    const T *get() const
    {
        return m_ptr;
    }

    ~CCShareableHolder()
    {
        if (m_ptr)
        {
            m_ptr->release();
            m_ptr = nullptr;
        }
    }

  private:
    T *m_ptr = nullptr;
};

PYBIND11_DECLARE_HOLDER_TYPE(T, CCShareableHolder<T>);

namespace PyCC
{
inline void NoOpDelete(void *) {}

template <class T> py::array_t<T> SpanAsNumpyArray(T *data, py::array::ShapeContainer shape)
{
    auto capsule = py::capsule(data, NoOpDelete);
    return py::array(shape, data, capsule);
}

template <class T> py::array_t<T> SpanAsNumpyArray(T *data, size_t len)
{
    auto capsule = py::capsule(data, NoOpDelete);
    return py::array(len, data, capsule);
}

template <class T> py::array_t<T> VectorAsNumpyArray(std::vector<T> &vector)
{
    // https://stackoverflow.com/questions/44659924/returning-numpy-arrays-via-pybind11
    // https://github.com/pybind/pybind11/issues/1042
    return SpanAsNumpyArray(vector.data(), vector.size());
}

template <class PointCloudType>
void addPointsFromArrays(PointCloudType &self,
                         py::array_t<PointCoordinateType> &xs,
                         py::array_t<PointCoordinateType> &ys,
                         py::array_t<PointCoordinateType> &zs)
{
    if (xs.size() != ys.size() || xs.size() != zs.size())
    {
        throw py::value_error("xs, ys, zs must have the same size");
    }

    const py::ssize_t numToReserve = self.size() + xs.size();
    if (numToReserve > std::numeric_limits<unsigned int>::max())
    {
        throw std::out_of_range(std::to_string(numToReserve) + " cannot be casted to unsigned int");
    }
    self.reserve(static_cast<unsigned int>(numToReserve));

    auto xs_it = xs.begin();
    auto ys_it = ys.begin();
    auto zs_it = zs.begin();

    for (; xs_it != xs.end();)
    {
        self.addPoint({xs_it->cast<PointCoordinateType>(),
                       ys_it->cast<PointCoordinateType>(),
                       zs_it->cast<PointCoordinateType>()});
        ++xs_it;
        ++ys_it;
        ++zs_it;
    }
}
} // namespace PyCC

static const constexpr char ADD_SCALAR_FIELD_DOCSTRING[] = R"doc(
     Adds a scalar field with the given name to the point cloud.

    Parameters
    ----------
    name: str
        name of the scalar field that will be added.
    values: optional, numpy.array, list of float
        values to use when initializing the new scalar field

    Raises
    ------
    RuntimeError if the point cloud already has a scalar field with the given ``name``
    ValueError if values are provided don't have the same length (size) as the point cloud

)doc";

static const constexpr char SIZE_SCALAR_FIELD_DOCSTRING[] = R"doc(
    Returns the size (number of points) in the point cloud.

    ``len`` also works as an alias to size.

    .. code:: Python

        pc = pycc.ccPointCloud("name")
        assert len(pc) == pc.size()
)doc";

#define DEFINE_POINTCLOUDTPL(T, module, name)                                                                \
    py::class_<cloudViewer::PointCloudTpl<T>, T>(module, name)                                               \
        .def("size", &cloudViewer::PointCloudTpl<T>::size, SIZE_SCALAR_FIELD_DOCSTRING)                      \
        .def("forEach", &cloudViewer::PointCloudTpl<T>::forEach, "action"_a)                                 \
        .def("getBoundingBox", &cloudViewer::PointCloudTpl<T>::getBoundingBox, "bbMin"_a, "bbMax"_a)         \
        .def("getNextPoint",                                                                                 \
             &cloudViewer::PointCloudTpl<T>::getNextPoint,                                                   \
             py::return_value_policy::reference)                                                             \
        .def("enableScalarField", &cloudViewer::PointCloudTpl<T>::enableScalarField)                         \
        .def("isScalarFieldEnabled", &cloudViewer::PointCloudTpl<T>::isScalarFieldEnabled)                   \
        .def("setPointScalarValue",                                                                          \
             &cloudViewer::PointCloudTpl<T>::setPointScalarValue,                                            \
             "pointIndex"_a,                                                                                 \
             "value"_a)                                                                                      \
        .def("getPointScalarValue", &cloudViewer::PointCloudTpl<T>::getPointScalarValue, "pointIndex"_a)     \
        .def("resize", &cloudViewer::PointCloudTpl<T>::resize, "newCount"_a)                                 \
        .def("reserve", &cloudViewer::PointCloudTpl<T>::reserve, "newCapacity"_a)                            \
        .def("reset", &cloudViewer::PointCloudTpl<T>::reset)                                                 \
        .def("invalidateBoundingBox", &cloudViewer::PointCloudTpl<T>::invalidateBoundingBox)                 \
        .def("getNumberOfScalarFields",                                                                      \
             &cloudViewer::PointCloudTpl<T>::getNumberOfScalarFields,                                        \
             "Returns the number of scalar field of the point cloud")                                        \
        .def(                                                                                                \
            "getScalarField",                                                                                \
            &cloudViewer::PointCloudTpl<T>::getScalarField,                                                  \
            "index"_a,                                                                                       \
            R"doc(                                                                                           \
    Returns the scalar field identified by its index.                                                        \
                                                                                                             \
    If index is invalid, None is returned                                                                    \
)doc")                                                                                                       \
        .def(                                                                                                \
            "getScalarFieldName",                                                                            \
            &cloudViewer::PointCloudTpl<T>::getScalarFieldName,                                              \
            "index"_a,                                                                                       \
            R"doc(                                                                                           \
    Returns the name of the scalar field identified by the index.                                            \
                                                                                                             \
    If index is invalid, -1 is returned                                                                      \
)doc")                                                                                                       \
        .def(                                                                                                \
            "getScalarFieldIndexByName",                                                                     \
            &cloudViewer::PointCloudTpl<T>::getScalarFieldIndexByName,                                       \
            "name"_a,                                                                                        \
            R"doc(                                                                                           \
     Returns the scalar field identified by its name.                                                        \
                                                                                                             \
    If no scalar field has the given name, None is returned                                                  \
)doc")                                                                                                       \
        .def("getCurrentInScalarField", &cloudViewer::PointCloudTpl<T>::getCurrentInScalarField)             \
        .def("getCurrentOutScalarField", &cloudViewer::PointCloudTpl<T>::getCurrentOutScalarField)           \
        .def("setCurrentInScalarField", &cloudViewer::PointCloudTpl<T>::setCurrentInScalarField, "index"_a)  \
        .def("getCurrentInScalarFieldIndex", &cloudViewer::PointCloudTpl<T>::getCurrentInScalarFieldIndex)   \
        .def(                                                                                                \
            "setCurrentOutScalarField", &cloudViewer::PointCloudTpl<T>::setCurrentOutScalarField, "index"_a) \
        .def("getCurrentOutScalarFieldIndex", &cloudViewer::PointCloudTpl<T>::getCurrentOutScalarFieldIndex) \
        .def("setCurrentScalarField", &cloudViewer::PointCloudTpl<T>::setCurrentScalarField, "index"_a)      \
        .def("renameScalarField", &cloudViewer::PointCloudTpl<T>::renameScalarField, "index"_a, "newName"_a) \
        .def(                                                                                                \
            "addScalarField",                                                                                \
            [](cloudViewer::PointCloudTpl<T> &self,                                                          \
               const char *sfName,                                                                           \
               const py::object &maybe_values = py::none())                                                  \
            {                                                                                                \
                int idx = self.addScalarField(sfName);                                                       \
                if (idx == -1)                                                                               \
                {                                                                                            \
                    throw std::runtime_error("Failed to add scalar field");                                  \
                }                                                                                            \
                if (!maybe_values.is_none())                                                                 \
                {                                                                                            \
                    py::array_t<ScalarType> values(maybe_values);                                            \
                    if (values.size() != self.size())                                                        \
                    {                                                                                        \
                        throw py::value_error("value must have the same len as the poinc cloud size");       \
                    }                                                                                        \
                    auto values_u = values.unchecked<1>();                                                   \
                    cloudViewer::ScalarField *sf = self.getScalarField(idx);                                 \
                    for (py::ssize_t i{0}; i < values.size(); ++i)                                           \
                    {                                                                                        \
                        sf->setValue(i, values_u(i));                                                        \
                    }                                                                                        \
                }                                                                                            \
                return idx;                                                                                  \
            },                                                                                               \
            "name"_a,                                                                                        \
            "values"_a = py::none(),                                                                         \
            ADD_SCALAR_FIELD_DOCSTRING)                                                                      \
        .def(                                                                                                \
            "deleteScalarField",                                                                             \
            &cloudViewer::PointCloudTpl<T>::deleteScalarField,                                               \
            "index"_a,                                                                                       \
            R"doc(                                                                                           \
     Removes the scalar field identified by the index          .                                             \
                                                                                                             \
    .. warning::                                                                                             \
        This operation may modify the scalar fields order                                                    \
        (especially if the deleted SF is not the last one).                                                  \
        However current IN & OUT scalar fields will stay up-to-date                                          \
        (while their index may change).                                                                      \
                                                                                                             \
    Does nothing if index is invalid                                                                         \
)doc")                                                                                                       \
        .def("deleteAllScalarFields",                                                                        \
             &cloudViewer::PointCloudTpl<T>::deleteAllScalarFields,                                          \
             "Deletes all scalar fields associated to this cloud")                                           \
        .def(                                                                                                \
            "addPoint",                                                                                      \
            [](cloudViewer::PointCloudTpl<T> &self, const CCVector3 &P) { self.addPoint(P); },               \
            "P"_a,                                                                                           \
            R"doc(                                                                                           \
    Adds a 3D point to the point cloud                                                                       \
                                                                                                             \
    .. note::                                                                                                \
        For better performances it is better to use :meth:`.addPoints`.                                      \
)doc")                                                                                                       \
        .def(                                                                                                \
            "addPoints",                                                                                     \
            &PyCC::addPointsFromArrays<cloudViewer::PointCloudTpl<T>>,                                       \
            "xs"_a,                                                                                          \
            "ys"_a,                                                                                          \
            "zs"_a,                                                                                          \
            R"doc(                                                                                           \
    Takes values from xs, yz, zs array and add them as points of the point cloud.                            \
                                                                                                             \
    Raises                                                                                                   \
    ------                                                                                                   \
    Value error if xs,ys,zs do not have the same length                                                      \
)doc")                                                                                                       \
        .def("__len__", &cloudViewer::PointCloudTpl<T>::size);

#endif // PYTHON_PLUGIN_WRAPPERS_H
