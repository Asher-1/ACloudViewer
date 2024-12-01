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
#include <pybind11/stl_bind.h>

#include "wrappers.h"

#include <ScalarField.h>

namespace py = pybind11;
using namespace pybind11::literals;

void define_ScalarField(py::module &cccorelib)
{
    py::class_<cloudViewer::ScalarField, CCShareable, CCShareableHolder<cloudViewer::ScalarField>>(cccorelib,
                                                                                               "ScalarField",
                                                                                               R"doc(
    ScalarField

    .. note::
        Note that cccorelib.ScalarField uses float while Python only uses doubles
        which means a loss of precision will happen.

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
             &cloudViewer::ScalarField::size,
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
        .def("reserve", &cloudViewer::ScalarField::reserve, "count"_a, R"doc(
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
            [](cloudViewer::ScalarField &self, size_t count, ScalarType valueForNewElements)
            {
                // pybind11 will convert exceptions
                self.resize(count, valueForNewElements);
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
        .def("getValue",
             static_cast<ScalarType &(cloudViewer::ScalarField::*)(std::size_t)>(
                 &cloudViewer::ScalarField::getValue),
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
            [](cloudViewer::ScalarField &self) { return PyCC::VectorAsNumpyArray(self); },
            R"doc(
    Returns the scalar field viewed as a numpy array.

    This does not return a copy, so changes made to the array a reflected
    in the scalar field (and vice-versa).

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
        .def("__getitem__",
             static_cast<ScalarType &(cloudViewer::ScalarField::*)(std::size_t)>(
                 &cloudViewer::ScalarField::getValue))
        .def("__setitem__", &cloudViewer::ScalarField::setValue)
        .def("__repr__",
             [](const cloudViewer::ScalarField &self)
             { return std::string("<ScalarField(name=") + self.getName() + ")>"; });
}
