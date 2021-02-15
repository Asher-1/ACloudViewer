// ----------------------------------------------------------------------------
// -                        cloudViewer: www.erow.cn                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.erow.cn
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

// CV_CORE_LIB
#include <Console.h>

// ECV_DB_LIB
#include <ecvPointCloud.h>
#include <ecvMesh.h>
#include <ecvBox.h>
#include <ecvCone.h>
#include <ecvDish.h>
#include <ecvExtru.h>
#include <ecvFacet.h>
#include <ecvPlane.h>
#include <ecvSphere.h>
#include <ecvTorus.h>
#include <ecvQuadric.h>
#include <ecvCylinder.h>
#include <ecvGenericPrimitive.h>
#include <ecvPlanarEntityInterface.h>

// LOCAL
#include "pybind/docstring.h"
#include "pybind/geometry/geometry.h"
#include "pybind/geometry/geometry_trampoline.h"

namespace cloudViewer {
namespace geometry {

void pybind_primitives(py::module &m) {
	// cloudViewer.geometry.ccGenericPrimitive
	py::class_<ccGenericPrimitive, 
		PyGenericPrimitive<ccGenericPrimitive>,
		std::shared_ptr<ccGenericPrimitive>, ccMesh>
		primitive(m, "ccGenericPrimitive", "The base primitives class.");
		primitive.def("__repr__", [](const ccGenericPrimitive &prim) {
			std::string info = fmt::format(
				"ccGenericPrimitive with type {}, precision {}",
				prim.getTypeName().toStdString(), prim.getDrawingPrecision());
			return info;
		})
		.def("clone",
			[](const ccGenericPrimitive& prim) {
				return std::shared_ptr<ccGenericPrimitive>(prim.clone());
			}, "Clones primitive.")
		.def("get_type_name",
			[](const ccGenericPrimitive &prim) {
				return prim.getTypeName().toStdString(); 
			}, "Returns type name (sphere, cylinder, etc.)")
		.def("set_color", 
			[](ccGenericPrimitive& prim, const Eigen::Vector3d& color) {
				prim.setColor(ecvColor::Rgb::FromEigen(color));
			}, "Sets primitive color (shortcut).",
			"color"_a)
		.def("has_drawing_precision", &ccGenericPrimitive::hasDrawingPrecision,
			"Whether drawing is dependent on 'precision' parameter.")
		.def("set_drawing_precision", &ccGenericPrimitive::setDrawingPrecision,
			"Sets drawing precision.", "steps"_a)
		.def("get_drawing_precision", &ccGenericPrimitive::getDrawingPrecision,
			"Returns drawing precision (or 0 if feature is not supported).")
		.def("get_transformation", 
			[](const ccGenericPrimitive& prim) {
				return ccGLMatrixd::ToEigenMatrix4(prim.getTransformation());
			}, "Returns the transformation that is currently applied to the vertices.")
		.def("get_transformation_history",
			[](const ccGenericPrimitive& prim) {
				return ccGLMatrixd::ToEigenMatrix4(prim.getGLTransformationHistory());
			}, "inherited methods (ccHObject).");
	docstring::ClassMethodDocInject(m, "ccGenericPrimitive", "clone");
	docstring::ClassMethodDocInject(m, "ccGenericPrimitive", "get_type_name");
	docstring::ClassMethodDocInject(m, "ccGenericPrimitive", "set_color",
			{ {"color", "rgb color."} });
	docstring::ClassMethodDocInject(m, "ccGenericPrimitive", "has_drawing_precision");
	docstring::ClassMethodDocInject(m, "ccGenericPrimitive", "set_drawing_precision",
			{ {"steps", 
			"Warnings: "
			"- steps should always be >= ccGenericPrimitive::MIN_DRAWING_PRECISION"
			"- changes primitive content(calls ccGenericPrimitive::updateRepresentation)"
			"- may fail if not enough memory!"
			"- param steps drawing precision"
			"return success(false if not enough memory)"} } );
	docstring::ClassMethodDocInject(m, "ccGenericPrimitive", "get_drawing_precision");
	docstring::ClassMethodDocInject(m, "ccGenericPrimitive", "get_transformation");
	docstring::ClassMethodDocInject(m, "ccGenericPrimitive", "get_transformation_history");

	// cloudViewer.geometry.ccPlane
	py::class_<ccPlane, ccGenericPrimitive, std::shared_ptr<ccPlane>, ccPlanarEntityInterface>
		pyplane(m, "ccPlane", py::multiple_inheritance(), "The 3D plane primitive.");
	py::detail::bind_default_constructor<ccPlane>(pyplane);
	py::detail::bind_copy_functions<ccPlane>(pyplane);
	pyplane.def(py::init([](const std::string& name) {
			return new ccPlane(name.c_str());
		}), "Simplified constructor", "name"_a = "Plane")
		.def(py::init([](PointCoordinateType width, PointCoordinateType height,
			const Eigen::Matrix4d& trans_matrix, const std::string& name) {
			const ccGLMatrix matrix = ccGLMatrix::FromEigenMatrix(trans_matrix);
			auto prim = new ccPlane(width, height, &matrix, name.c_str());
			prim->clearTriNormals();
			return prim;
		}), "Plane normal corresponds to 'Z' dimension: "
			"\n"
			"-param width plane width along 'X' dimension; "
			"\n"
			"-param height plane width along 'Y' dimension; "
			"\n"
			"param trans_matrix optional 3D transformation (can be set afterwards with ccDrawableObject::setGLTransformation); "
			"\n"
			"-param name name.", 
			"width"_a, "height"_a,
			"trans_matrix"_a = Eigen::Matrix4d::Identity(), "name"_a = "Plane")
		.def("__repr__", [](ccPlane &plane) {
			const PointCoordinateType* eq = plane.getEquation();
			std::string info = fmt::format(
				"ccPlane with faces {}, width {}, height {} and equation {}x + {}y + {}z = {}",
				plane.size(), plane.getXWidth(), plane.getYWidth(), eq[0], eq[1], eq[2], eq[3]);
			return info;
		})
		.def("flip", &ccPlane::flip, "Flips the plane.")
		.def("get_width", &ccPlane::getXWidth, "Returns 'X' width.")
		.def("set_width", [](ccPlane& plane, PointCoordinateType width, bool auto_update) {
			plane.setXWidth(width, auto_update);
		}, "Sets 'X' width.", "width"_a, "auto_update"_a = true)
		.def("get_height", &ccPlane::getYWidth, "Returns 'Y' width.")
		.def("set_height", [](ccPlane& plane, PointCoordinateType height, bool auto_update) {
			plane.setYWidth(height, auto_update);
		}, "Sets 'Y' width.", "height"_a, "auto_update"_a = true)
		.def("get_center", [](const ccPlane& plane) {
			return CCVector3d::fromArray(plane.getCenter());
		}, "Returns the plane center.")
		.def("get_normal", [](const ccPlane& plane) {
			return CCVector3d::fromArray(plane.getNormal());
		}, "inherited from ccPlanarEntityInterface, returns the plane normal.")
		.def("get_equation", [](const ccPlane& plane) {
			CCVector3 N;
			PointCoordinateType constVal = 0;
			plane.getEquation(N, constVal);
			return std::make_tuple(CCVector3d::fromArray(N), constVal);
		}, 
			"Returns the equation of the plane. Equation: N.P + constVal = 0; "
			"i.e.Nx.x + Ny.y + Nz.z + constVal = 0");
	
	docstring::ClassMethodDocInject(m, "ccPlane", "flip");
	docstring::ClassMethodDocInject(m, "ccPlane", "get_width");
	docstring::ClassMethodDocInject(m, "ccPlane", "set_width",
		{ {"width", "plane 'X' width."}, {"auto_update", "auto update"} });
	docstring::ClassMethodDocInject(m, "ccPlane", "get_height");
	docstring::ClassMethodDocInject(m, "ccPlane", "set_height",
		{ {"height", "plane 'Y' width."}, {"auto_update", "auto update"} });
	docstring::ClassMethodDocInject(m, "ccPlane", "get_center");
	docstring::ClassMethodDocInject(m, "ccPlane", "get_normal");
	docstring::ClassMethodDocInject(m, "ccPlane", "get_equation");

	// cloudViewer.geometry.ccBox
	py::class_<ccBox, PyGenericPrimitive<ccBox>,
		std::shared_ptr<ccBox>, ccGenericPrimitive>
		pybox(m, "ccBox", "The 3D Box primitive.");
	py::detail::bind_default_constructor<ccBox>(pybox);
	py::detail::bind_copy_functions<ccBox>(pybox);
	pybox.def(py::init([](const std::string& name) {
		return new ccBox(name.c_str());
	}), "Simplified constructor", "name"_a = "Box")
	.def(py::init([](const Eigen::Vector3d& dims, 
			const Eigen::Matrix4d& trans_matrix, 
			const std::string& name) {
			const ccGLMatrix matrix = ccGLMatrix::FromEigenMatrix(trans_matrix);
			auto prim = new ccBox(CCVector3::fromArray(dims), &matrix, name.c_str());
			prim->clearTriNormals();
			return prim;
		}), "Box dimensions axis along each dimension are defined in a single 3D vector: "
		"A box is in fact composed of 6 planes (ccPlane)."
		"\n"
		"-param dims box dimensions; "
		"\n"
		"-param trans_matrix optional 3D transformation (can be set afterwards with ccDrawableObject::setGLTransformation); "
		"\n"
		"-param name name.",
		"dims"_a,
		"trans_matrix"_a = Eigen::Matrix4d::Identity(), "name"_a = "Box")
	.def("__repr__", [](const ccBox &box) {
			const CCVector3& dims = box.getDimensions();
			std::string info = fmt::format(
				"ccBox with faces {} and dimension ({}, {}, {})",
				box.size(), dims.x, dims.y, dims.z);
			return info;
		})
	.def("get_dimensions", [](ccBox& box) {
			return CCVector3d::fromArray(box.getDimensions());
		}, "Returns box dimensions.")
	.def("set_dimensions", [](ccBox& box, const Eigen::Vector3d& dims) {
            CCVector3 tempDims = CCVector3::fromArray(dims);
            box.setDimensions(tempDims);
		}, "Returns box dimensions.", "dims"_a);

	docstring::ClassMethodDocInject(m, "ccBox", "get_dimensions");
	docstring::ClassMethodDocInject(m, "ccBox", "set_dimensions",
		{ {"dims", "box dimensions (width, length, height)."} });

	// cloudViewer.geometry.ccSphere
	py::class_<ccSphere, PyGenericPrimitive<ccSphere>,
		std::shared_ptr<ccSphere>, ccGenericPrimitive>
		pysphere(m, "ccSphere", "The 3D sphere primitive.");
	py::detail::bind_default_constructor<ccSphere>(pysphere);
	py::detail::bind_copy_functions<ccSphere>(pysphere);
	pysphere.def(py::init([](const std::string& name) {
		return new ccSphere(name.c_str());
	}), "Simplified constructor", "name"_a = "Sphere")
	.def(py::init([](PointCoordinateType radius,
			const Eigen::Matrix4d& trans_matrix,
			unsigned precision, const std::string& name) {
		const ccGLMatrix matrix = ccGLMatrix::FromEigenMatrix(trans_matrix);
		auto prim = new ccSphere(radius, &matrix, name.c_str(), precision);
		prim->clearTriNormals();
		return prim;
	}), "-param radius sphere radius; "
		"\n"
		"-param trans_matrix optional 3D transformation (can be set afterwards with ccDrawableObject::setGLTransformation); "
		"\n"
		"-param precision drawing precision (angular step = 360/precision).",
		"\n"
		"-param name name.",
		"radius"_a,
		"trans_matrix"_a = Eigen::Matrix4d::Identity(),
		"precision"_a = 24,
		"name"_a = "Sphere")
	.def("__repr__", [](const ccSphere &sphere) {
		std::string info = fmt::format(
			"ccSphere with faces {} and radius {}",
			sphere.size(), sphere.getRadius());
		return info;
	})
	.def("get_radius", &ccSphere::getRadius, "Returns sphere radius.")
	.def("set_radius", &ccSphere::setRadius, "Sets sphere radius.", "radius"_a);

	docstring::ClassMethodDocInject(m, "ccSphere", "get_radius");
	docstring::ClassMethodDocInject(m, "ccSphere", "set_radius",
		{ {"radius", "sphere radius, warning changes primitive content "
		"(calls ccGenericPrimitive::updateRepresentation)."} });

	// cloudViewer.geometry.ccTorus
	py::class_<ccTorus, PyGenericPrimitive<ccTorus>,
		std::shared_ptr<ccTorus>, ccGenericPrimitive>
		pytorus(m, "ccTorus", "The 3D torus primitive.");
	py::detail::bind_default_constructor<ccTorus>(pytorus);
	py::detail::bind_copy_functions<ccTorus>(pytorus);
	pytorus.def(py::init([](const std::string& name) {
		return new ccTorus(name.c_str());
	}), "Simplified constructor", "name"_a = "Torus")
	.def(py::init([](PointCoordinateType inside_radius,
			PointCoordinateType outside_radius,
			double angle_rad,
			bool rectangular_section,
			PointCoordinateType rect_section_height,
			const Eigen::Matrix4d& trans_matrix,
			unsigned precision, const std::string& name) {
		const ccGLMatrix matrix = ccGLMatrix::FromEigenMatrix(trans_matrix);
		auto prim = new ccTorus(inside_radius, outside_radius,
			angle_rad, rectangular_section, rect_section_height,
			&matrix, name.c_str(), precision);
		prim->clearTriNormals();
		return prim;
	}), "Torus is defined in the XY plane by default: "
		"\n"
		"-param inside_radius inside radius.",
		"\n"
		"-param outside_radius outside radius.",
		"\n"
		"-param angle_rad subtended angle (in radians).",
		"\n"
		"-param rectangular_section whether section is rectangular or round.",
		"\n"
		"-param rect_section_height section height (if rectangular torus).",
		"\n"
		"-param trans_matrix optional 3D transformation (can be set afterwards with ccDrawableObject::setGLTransformation); "
		"\n"
		"-param precision drawing precision (angular step = 360/precision).",
		"\n"
		"-param name name.",
		"inside_radius"_a,
		"outside_radius"_a,
		"angle_rad"_a = 2.0*M_PI,
		"rectangular_section"_a = false,
		"rect_section_height"_a = 0,
		"trans_matrix"_a = Eigen::Matrix4d::Identity(),
		"precision"_a = 24,
		"name"_a = "Torus")
	.def("__repr__", [](const ccTorus &torus) {
		std::string info = fmt::format(
			"ccTorus with faces {}, inside-radius {}, outside-radius {} and height {}",
			torus.size(), torus.getInsideRadius(), torus.getOutsideRadius(), torus.getRectSectionHeight());
		return info;
	})
	.def("get_inside_radius", &ccTorus::getInsideRadius, "Returns the torus inside radius.")
	.def("get_outside_radius", &ccTorus::getOutsideRadius, "Returns the torus outside radius.")
	.def("get_rect_section_height", &ccTorus::getRectSectionHeight,
		"Returns the torus rectangular section height (along Y-axis) if applicable.")
	.def("get_rectangular_section", &ccTorus::getRectSection, 
		"Returns whether torus has a rectangular (true) or circular (false) section.")
	.def("get_angle_rad", &ccTorus::getAngleRad, "Returns the torus subtended angle (in radians).");

	docstring::ClassMethodDocInject(m, "ccTorus", "get_inside_radius");
	docstring::ClassMethodDocInject(m, "ccTorus", "get_outside_radius");
	docstring::ClassMethodDocInject(m, "ccTorus", "get_rect_section_height");
	docstring::ClassMethodDocInject(m, "ccTorus", "get_rectangular_section");
	docstring::ClassMethodDocInject(m, "ccTorus", "get_angle_rad");

	// cloudViewer.geometry.ccQuadric
	py::class_<ccQuadric, PyGenericPrimitive<ccQuadric>,
		std::shared_ptr<ccQuadric>, ccGenericPrimitive>
		pyquadric(m, "ccQuadric", "The 3D quadric primitive.");
	py::detail::bind_default_constructor<ccQuadric>(pyquadric);
	py::detail::bind_copy_functions<ccQuadric>(pyquadric);
	pyquadric.def(py::init([](const std::string& name) {
		return new ccQuadric(name.c_str());
	}), "Simplified constructor", "name"_a = "Quadric")
	.def(py::init([](
			const Eigen::Vector2d& min_corner,
			const Eigen::Vector2d& max_corner,
			const Eigen::Vector6d& equation,
			const Eigen::Vector3i& dimensions,
			const Eigen::Matrix4d& trans_matrix,
			unsigned precision, const std::string& name) {

		const ccGLMatrix matrix = ccGLMatrix::FromEigenMatrix(trans_matrix);
		CCVector2 minCorner(static_cast<PointCoordinateType>(min_corner(0)),
			static_cast<PointCoordinateType>(min_corner(1)));		
		CCVector2 maxCorner(static_cast<PointCoordinateType>(max_corner(0)),
			static_cast<PointCoordinateType>(max_corner(1)));
		PointCoordinateType eq[6];
		for (size_t i = 0;  i < 6; ++i)
		{
			eq[i] = static_cast<PointCoordinateType>(equation(i));
		}
		
		Tuple3ub dims(static_cast<unsigned char>(dimensions(0)),
			static_cast<unsigned char>(dimensions(1)),
			static_cast<unsigned char>(dimensions(2)));

		auto prim = new ccQuadric(minCorner, maxCorner, eq, &dims, &matrix, name.c_str(), precision);
		prim->clearTriNormals();
		return prim;
	}), "-Quadric orthogonal dimension is 'Z' by default: "
		"\n"
		"-param min_corner min corner of the 'representation' base area; "
		"\n"
		"-param max_corner max corner of the 'representation' base area; "
		"\n"
		"-param equation equation coefficients ( Z = a + b.X + c.Y + d.X^2 + e.X.Y + f.Y^2); "
		"\n"
		"-param dimensions optional dimension indexes; "
		"\n"
		"-param trans_matrix optional 3D transformation (can be set afterwards with ccDrawableObject::setGLTransformation); "
		"\n"
		"-param precision drawing precision (angular step = 360/precision).",
		"\n"
		"-param name name.",
		"min_corner"_a,
		"max_corner"_a,
		"equation"_a,
		"dimensions"_a = Eigen::Vector3i::Ones(),
		"trans_matrix"_a = Eigen::Matrix4d::Identity(),
		"precision"_a = 24,
		"name"_a = "Quadric")
	.def("__repr__", [](const ccQuadric &quadric) {
		std::string info = fmt::format(
			"ccQuadric with faces {} and equations: {}",
			quadric.size(), quadric.getEquationString().toStdString());
		return info;
	})
	.def("get_min_corner", [](const ccQuadric& quadric) {
		return Eigen::Vector2d(quadric.getMinCorner().x, quadric.getMinCorner().y);
	}, "Returns the quadric min corner.")
	.def("get_max_corner", [](const ccQuadric& quadric) {
		return Eigen::Vector2d(quadric.getMaxCorner().x, quadric.getMaxCorner().y);
	}, "Returns the quadric max corner.")
	.def("get_equation_coefficient", [](const ccQuadric& quadric) {
		const PointCoordinateType* equation = quadric.getEquationCoefs();
		Eigen::Vector6d coefficient(6);
		for (size_t i = 0; i < 6; ++i)
		{
			coefficient(i) = static_cast<double>(equation[i]);
		}
		return coefficient;
	}, "Returns the quadric equation coefficients.")
	.def("get_equation_dims", [](const ccQuadric& quadric) {
		const Tuple3ub& dims = quadric.getEquationDims();
		return Eigen::Vector3i(
			static_cast<int>(dims.x),
			static_cast<int>(dims.y),
			static_cast<int>(dims.z));
	}, "Returns the quadric equation 'coordinate system' (X,Y,Z dimensions indexes).")
	.def("get_equation_string", [](const ccQuadric& quadric) {
		return quadric.getEquationString().toStdString();
	}, "Returns the quadric equation coefficients as a string.")
	.def("project_on_quadric", [](const ccQuadric& quadric, const Eigen::Vector3d& point) {
		const CCVector3 P = CCVector3::fromArray(point);
		CCVector3 Q;
		PointCoordinateType elevation = quadric.projectOnQuadric(P, Q);
		return std::make_tuple(elevation, CCVector3d::fromArray(point));
	}, "Returns the quadric equation coefficients as a string.")
	.def_static("fit", [](cloudViewer::GenericIndexedCloudPersist& cloud) {
		cloudViewer::GenericIndexedCloudPersist* persistCloud = static_cast<cloudViewer::GenericIndexedCloudPersist*>(&cloud);
		if (!persistCloud)
		{
			cloudViewer::utility::LogWarning(
				"[ccQuadric::Fit] Illegal input parameters, only support point cloud!");
			return std::make_tuple(std::make_shared<ccQuadric>("Quadric"), 0.0);
		}
		double rms = 0.0;
		ccQuadric* quadric = ccQuadric::Fit(persistCloud, &rms);
		return std::make_tuple(std::shared_ptr<ccQuadric>(quadric), rms);
	}, "Fits a quadric primitive on a cloud", "cloud"_a);

	docstring::ClassMethodDocInject(m, "ccQuadric", "get_min_corner");
	docstring::ClassMethodDocInject(m, "ccQuadric", "get_max_corner");
	docstring::ClassMethodDocInject(m, "ccQuadric", "get_equation_coefficient");
	docstring::ClassMethodDocInject(m, "ccQuadric", "get_equation_dims");
	docstring::ClassMethodDocInject(m, "ccQuadric", "get_equation_string");
	docstring::ClassMethodDocInject(m, "ccQuadric", "project_on_quadric");

	// cloudViewer.geometry.ccCone
	py::class_<ccCone, PyGenericPrimitive<ccCone>,
		std::shared_ptr<ccCone>, ccGenericPrimitive>
		pycone(m, "ccCone", "The 3D cone primitive.");
	py::detail::bind_default_constructor<ccCone>(pycone);
	py::detail::bind_copy_functions<ccCone>(pycone);
	pycone.def(py::init([](const std::string& name) {
		return new ccCone(name.c_str());
	}), "Simplified constructor", "name"_a = "Cone")
	.def(py::init([](
			PointCoordinateType bottom_radius,
			PointCoordinateType top_radius,
			PointCoordinateType height,
			PointCoordinateType x_off,
			PointCoordinateType y_off,
			const Eigen::Matrix4d& trans_matrix,
			unsigned precision, const std::string& name) {
			const ccGLMatrix matrix = ccGLMatrix::FromEigenMatrix(trans_matrix);
			auto prim = new ccCone(bottom_radius, top_radius, height,
				x_off, y_off, &matrix, name.c_str(), precision);
			prim->clearTriNormals();
			return prim;
		}), "Cone axis corresponds to the 'Z' dimension by default: "
		"\n"
		"-param bottom_radius cone bottom radius; "
		"\n"
		"-param top_radius cone top radius; "
		"\n"
		"-param height cone height (transformation should point to the axis center); "
		"\n"
		"-param x_off displacement of axes along X-axis (Snout mode); "
		"\n"
		"-param y_off displacement of axes along Y-axis (Snout mode); "
		"\n"
		"-param trans_matrix optional 3D transformation (can be set afterwards with ccDrawableObject::setGLTransformation); "
		"\n"
		"-param precision drawing precision (angular step = 360/precision); "
		"\n"
		"-param name primitive name.", 
		"bottom_radius"_a, "top_radius"_a, "height"_a,
		"x_off"_a = 0, "y_off"_a = 0, "trans_matrix"_a = Eigen::Matrix4d::Identity(),
		"precision"_a = 24, "name"_a = "Cone")
	.def("__repr__", [](const ccCone &cone) {
		std::string info = fmt::format(
			"ccCone with faces {}, bottom radius {}, top radius {} and height {}",
			cone.size(), cone.getBottomRadius(), cone.getTopRadius(), cone.getHeight());
		return info;
	})
	.def("get_height", &ccCone::getHeight, "Returns cone height.")
	.def("set_height", &ccCone::setHeight, "Sets cone height", "height"_a)
	.def("get_bottom_radius", &ccCone::getBottomRadius, "Returns cone bottom radius.")
	.def("set_bottom_radius", &ccCone::setBottomRadius, "Sets cone bottom radius.", "radius"_a)
	.def("get_top_radius", &ccCone::getTopRadius, "Returns cone top radius.")
	.def("set_top_radius", &ccCone::setTopRadius, "Sets cone top radius.", "radius"_a)
	.def("get_bottom_center", [](const ccCone& cone) {
		CCVector3d::fromArray(cone.getBottomCenter());
	}, "Returns cone axis bottom end point after applying transformation")
	.def("get_top_center", [](const ccCone& cone) {
		CCVector3d::fromArray(cone.getTopCenter());
	}, "Returns cone axis top end point after applying transformation")
	.def("get_small_center", [](const ccCone& cone) {
		CCVector3d::fromArray(cone.getSmallCenter());
	}, "Returns cone axis end point associated with whichever radii is smaller")
	.def("get_large_center", [](const ccCone& cone) {
		CCVector3d::fromArray(cone.getLargeCenter());
	}, "Returns cone axis end point associated with whichever radii is larger")
	.def("get_small_radius", &ccCone::getSmallRadius, "Returns whichever cone radii is smaller")
	.def("get_large_radius", &ccCone::getLargeRadius, "Returns whichever cone radii is larger")
	.def("is_snout_mode", &ccCone::isSnoutMode, "Returns true if the Cone was created in snout mode.");

	docstring::ClassMethodDocInject(m, "ccCone", "get_height");
	docstring::ClassMethodDocInject(m, "ccCone", "set_height",
		{ {"height", "changes primitive content (calls ccGenericPrimitive::updateRepresentation)."} });
	docstring::ClassMethodDocInject(m, "ccCone", "get_bottom_radius");
	docstring::ClassMethodDocInject(m, "ccCone", "set_bottom_radius",
		{ {"radius", "changes primitive content (calls ccGenericPrimitive::updateRepresentation)."} });
	docstring::ClassMethodDocInject(m, "ccCone", "get_top_radius");
	docstring::ClassMethodDocInject(m, "ccCone", "set_top_radius",
		{ {"radius", "changes primitive content (calls ccGenericPrimitive::updateRepresentation)."} });
	docstring::ClassMethodDocInject(m, "ccCone", "get_bottom_center");
	docstring::ClassMethodDocInject(m, "ccCone", "get_top_center");
	docstring::ClassMethodDocInject(m, "ccCone", "get_small_center");
	docstring::ClassMethodDocInject(m, "ccCone", "get_large_center");
	docstring::ClassMethodDocInject(m, "ccCone", "get_small_radius");
	docstring::ClassMethodDocInject(m, "ccCone", "get_large_radius");
	docstring::ClassMethodDocInject(m, "ccCone", "is_snout_mode");

	// cloudViewer.geometry.ccCylinder
	py::class_<ccCylinder, PyGenericPrimitive<ccCylinder>,
		std::shared_ptr<ccCylinder>, ccCone>
		pycylinder(m, "ccCylinder", "The 3D Box primitive.");
	py::detail::bind_default_constructor<ccCylinder>(pycylinder);
	py::detail::bind_copy_functions<ccCylinder>(pycylinder);
	pycylinder.def(py::init([](const std::string& name) {
		return new ccCylinder(name.c_str());
	}), "Simplified constructor", "name"_a = "Cylinder")
	.def(py::init([](PointCoordinateType radius, PointCoordinateType height,
			const Eigen::Matrix4d& trans_matrix, unsigned precision, 
			const std::string& name) {
		const ccGLMatrix matrix = ccGLMatrix::FromEigenMatrix(trans_matrix);
		auto prim = new ccCylinder(radius, height, &matrix, name.c_str(), precision);
		prim->clearTriNormals();
		return prim;
	}), "Cylinder axis corresponds to the 'Z' dimension: "
		"Internally represented by a cone with the same top and bottom radius. "
		"\n"
		"-param radius cylinder radius"
		"\n"
		"-param height cylinder height(transformation should point to the axis center)"
		"\n"
		"-param trans_matrix optional 3D transformation (can be set afterwards with ccDrawableObject::setGLTransformation); "
		"\n"
		"-param precision drawing precision (angular step = 360/precision); "
		"\n"
		"-param name name.", 
		"radius"_a,
		"height"_a,
		"trans_matrix"_a = Eigen::Matrix4d::Identity(), 
		"precision"_a = 24,
		"name"_a = "Cylinder")
	.def("__repr__", [](const ccCylinder &cylinder) {
		std::string info = fmt::format(
			"ccCylinder with faces {}, radius {} and height {}",
			cylinder.size(), cylinder.getBottomRadius(), cylinder.getHeight());
		return info;
	});

	// cloudViewer.geometry.ccDish
	py::class_<ccDish, PyGenericPrimitive<ccDish>,
		std::shared_ptr<ccDish>, ccGenericPrimitive>
		pydish(m, "ccDish", "The 3D dish primitive.");
	py::detail::bind_default_constructor<ccDish>(pydish);
	py::detail::bind_copy_functions<ccDish>(pydish);
	pydish.def(py::init([](const std::string& name) {
		return new ccDish(name.c_str());
	}), "Simplified constructor", "name"_a = "Dish")
		.def(py::init([](
			PointCoordinateType base_radius,
			PointCoordinateType height,
			PointCoordinateType second_radius,
			const Eigen::Matrix4d& trans_matrix,
			unsigned precision,
			const std::string& name) {
		const ccGLMatrix matrix = ccGLMatrix::FromEigenMatrix(trans_matrix);
		auto prim = new ccDish(base_radius, height, second_radius, &matrix, name.c_str(), precision);
		prim->clearTriNormals();
		return prim;
	}), "dish dimensions axis along each dimension are defined in a single 3D vector: "
			"A dish is in fact composed of 6 planes (ccPlane)."
		"\n"
		"-param radius base radius; "
		"\n"
		"-param height maximum height of dished surface above base; "
		"\n"
		"-param radius2 If radius2 is zero, dish is drawn as a section of sphere. If radius2 is >0, dish is defined as half of an ellipsoid.; "
		"\n"
		"-param trans_matrix optional 3D transformation (can be set afterwards with ccDrawableObject::setGLTransformation); "
		"\n"
		"-param precision drawing precision (angular step = 360/precision); "
		"\n"
		"-param name name.", 
		"base_radius"_a,
		"height"_a,
		"second_radius"_a = 0,
		"trans_matrix"_a = Eigen::Matrix4d::Identity(), 
		"precision"_a = 24,
		"name"_a = "Dish")
	.def("__repr__", [](const ccDish &dish) {
		std::string info = fmt::format(
			"ccDish with faces {}, R1 {}, R2 {} and heigth {}",
			dish.size(), dish.getBaseRadius(), dish.getSecondRadius(), dish.getHeight());
		return info;
	})
	.def("get_base_radius", &ccDish::getBaseRadius, "Returns the dish base radius.")
	.def("get_second_radius", &ccDish::getSecondRadius, "Returns the dish second radius.")
	.def("get_height", &ccDish::getHeight, "Returns the dish height.");

	docstring::ClassMethodDocInject(m, "ccDish", "get_base_radius");
	docstring::ClassMethodDocInject(m, "ccDish", "get_second_radius");
	docstring::ClassMethodDocInject(m, "ccDish", "get_height");

	// cloudViewer.geometry.ccExtru
	py::class_<ccExtru, PyGenericPrimitive<ccExtru>,
		std::shared_ptr<ccExtru>, ccGenericPrimitive>
		pyextru(m, "ccExtru", "The 3D extru primitive.");
	py::detail::bind_default_constructor<ccExtru>(pyextru);
	py::detail::bind_copy_functions<ccExtru>(pyextru);
	pyextru.def(py::init([](const std::string& name) {
		return new ccExtru(name.c_str());
	}), "Simplified constructor", "name"_a = "Extrusion")
	.def(py::init([](const std::vector<Eigen::Vector2d>& profile,
			PointCoordinateType height,
			const Eigen::Matrix4d& trans_matrix,
			const std::string& name) {
			const ccGLMatrix matrix = ccGLMatrix::FromEigenMatrix(trans_matrix);
			std::vector<CCVector2> tempProfile(profile.size());
			for (const auto& pro : profile)
			{
				tempProfile.emplace_back(CCVector2(
					static_cast<PointCoordinateType>(pro(0)),
					static_cast<PointCoordinateType>(pro(1))));
			}
			auto prim = new ccExtru(tempProfile, height, &matrix, name.c_str());
			prim->clearTriNormals();
			return prim;
		}), "extru dimensions axis along each dimension are defined in a single 3D vector: "
			"A extru is in fact composed of 6 planes (ccPlane)."
		"\n"
		"-param profile 2D profile to extrude; "
		"\n"
		"-param height extrusion thickness; "
		"\n"
		"-param trans_matrix optional 3D transformation (can be set afterwards with ccDrawableObject::setGLTransformation); "
		"\n"
		"-param name name.", 
		"profile"_a,
		"height"_a,
		"trans_matrix"_a = Eigen::Matrix4d::Identity(), 
		"name"_a = "Extrusion")
	.def("__repr__", [](const ccExtru &extru) {
		std::string info = fmt::format(
			"ccExtru with faces {} and height {}",
			extru.size(), extru.getThickness());
		return info;
	})
	.def("get_thickness", &ccExtru::getThickness, "Returns extrusion thickness.")
	.def("get_profile", [](const ccExtru& extru) {
		const std::vector<CCVector2>& profile = extru.getProfile();
		std::vector<Eigen::Vector2d> outProfile;
		for (const auto& pro : profile)
		{
			outProfile.emplace_back(Eigen::Vector2d(pro.x, pro.y));
		}
		return outProfile;
	}, "Returns extrusion profile.");

	docstring::ClassMethodDocInject(m, "ccExtru", "get_thickness");
	docstring::ClassMethodDocInject(m, "ccExtru", "get_profile");
}

void pybind_primitives_methods(py::module &m) {}

}  // namespace geometry
}  // namespace cloudViewer
