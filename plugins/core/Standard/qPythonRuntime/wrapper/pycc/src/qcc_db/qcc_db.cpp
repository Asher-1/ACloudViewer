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

#include <ecvAdvancedTypes.h>
#include <ecvColorTypes.h>

#undef slots
#include <pybind11/pybind11.h>

#include "../../cccorelib/src/wrappers.h"

namespace py = pybind11;
using namespace pybind11::literals;

void define_ccDrawableObject(py::module &);
void define_ccScalarField(py::module &);
void define_ccObject(py::module &);
void define_ccGenericPointCloud(py::module &);
void define_ccPointCloud(py::module &);
void define_ccGenericDisplayTools(py::module &);
void define_ccDisplayTools(py::module &);
void define_ccGenericMesh(py::module &);
void define_ccMesh(py::module &);
// void define_ccSerializableObject(py::module &);
// void define_ccSerializationHelper(py::module &);
// void define_ccMaterial(py::module &);
// void define_ccMaterialSet(py::module &);
void define_ccGBLSensor(py::module &);
void define_ccGenericPrimitive(py::module &);
void define_ccSphere(py::module &);
void define_ccGLMatrix(py::module &);
void define_ccIndexedTransformation(py::module &m);
void define_ccPlane(py::module &);
void define_ccTorus(py::module &);
void define_ccBox(py::module &);
void define_ccDish(py::module &);
void define_ccCone(py::module &);
void define_ccCylinder(py::module &);
void define_ccProgressDialog(py::module &);
void define_ccLog(py::module &);
void define_ccNormalVectors(py::module &);
void define_ccInteractor(py::module &);
void define_cc2DLabel(py::module &);
void define_ccViewPortParameters(py::module &);
void define_ccViewportObject(py::module &);
void define_ccBBox(py::module &);
void define_ccSensor(py::module &m);
void define_ccCameraSensor(py::module &);
void define_ccColorScale(py::module &);
void define_ccColorScalesManager(py::module &);
void define_ccWaveform(py::module &m);
void define_ccSubMesh(py::module &m);
void define_ccRasterGrid(py::module &m);
void define_ccQuadric(py::module &m);
void define_ccPolyline(py::module &m);

template <class T> using observer_ptr = std::unique_ptr<T, py::nodelete>;

#ifndef USE_EMBEDDED_MODULES
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#endif

void define_qcc_db(py::module &m)
{
    /**********************************
     * qCC_db
     **********************************/

    py::class_<ecvColor::Rgba>(m, "Rgba", R"doc(
    RGBA color structure
)doc")
        .def(py::init<>(), R"doc(Inits color to (0,0,0, 0))doc")
        .def(py::init<ColorCompType, ColorCompType, ColorCompType, ColorCompType>(),
             "r"_a,
             "g"_a,
             "b"_a,
             "a"_a)
        .def_readwrite("r", &ecvColor::Rgba::r)
        .def_readwrite("g", &ecvColor::Rgba::g)
        .def_readwrite("b", &ecvColor::Rgba::b)
        .def_readwrite("a", &ecvColor::Rgba::a)
        .def("__ne__", &ecvColor::Rgba::operator!=);

    py::class_<ecvColor::Rgb>(m, "Rgb", R"doc(
    RGB color structure
)doc")
        .def(py::init<>(), R"doc(Inits color to (0,0,0))doc")
        .def(py::init<ColorCompType, ColorCompType, ColorCompType>(), "r"_a, "g"_a, "b"_a)
        .def_readwrite("r", &ecvColor::Rgb::r)
        .def_readwrite("g", &ecvColor::Rgb::g)
        .def_readwrite("b", &ecvColor::Rgb::b)
        .def("__ne__", &ecvColor::Rgb::operator!=);

    define_ccScalarField(m);
    define_ccGLMatrix(m);
    define_ccIndexedTransformation(m);
    define_ccInteractor(m);

    define_ccDrawableObject(m);
    define_ccObject(m);

    using VectorCompressedNormType = std::vector<CompressedNormType>;
    using ccArrayCompressedNormType = ccArray<CompressedNormType, 1, CompressedNormType>;

    // FIXME: strange that this is needed when _not_ using embedded modules
    //      but breaks if present while using embedded modules
#ifndef USE_EMBEDDED_MODULES
    py::bind_vector<VectorCompressedNormType>(
        m, "_VectorCompressedNormType", py::module_local(true));
#endif

    // Cannot have CCShareable as we would require CCShareable holder, but
    // VectorCompressedNormType does not have 'non-default holder type'.
    // We need to do a proper ccArray wrapping
    py::class_<ccArrayCompressedNormType,
               VectorCompressedNormType,
               // CCShareable,
               ccHObject,
               observer_ptr<ccArrayCompressedNormType>>(m, "_NormsIndexesArrayType");
    // py::class_<NormsIndexesTableType, ccArrayCompressedNormType>(m, "NormsIndexesTableType")
    //       .def(py::init<>());

    define_ccGenericMesh(m);
    define_ccMesh(m);
    // define_ccSerializableObject(m);
    // define_ccSerializationHelper(m);
    // define_ccMaterial(m);
    // define_ccMaterialSet(m);
    define_ccGenericPrimitive(m);
    define_ccSphere(m);
    define_ccPlane(m);
    define_ccTorus(m);
    define_ccBox(m);
    define_ccDish(m);
    define_ccCone(m);
    define_ccCylinder(m);
    define_ccQuadric(m);
    define_cc2DLabel(m);
    define_ccColorScale(m);
    define_ccColorScalesManager(m);
    define_ccViewPortParameters(m);
    define_ccViewportObject(m);
    define_ccBBox(m);
    define_ccSensor(m);
    define_ccGBLSensor(m);
    define_ccCameraSensor(m);
    define_ccSubMesh(m);
    define_ccWaveform(m);
    define_ccRasterGrid(m);
    define_ccGenericPointCloud(m);
    define_ccPointCloud(m);
    define_ccPolyline(m);
    define_ccNormalVectors(m);

    define_ccLog(m);
    define_ccProgressDialog(m);

    define_ccGenericDisplayTools(m);
    define_ccDisplayTools(m);
}
