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
#include <Logging.h>

// ECV_DB_LIB
#include <ecvMesh.h>
#include <ecvFacet.h>
#include <ecvPolyline.h>
#include <ecvPointCloud.h>
#include <ecvPlanarEntityInterface.h>

// LOCAL
#include "pybind/docstring.h"
#include "pybind/geometry/geometry.h"
#include "pybind/geometry/geometry_trampoline.h"

#pragma warning(disable:4715)

namespace cloudViewer {
namespace geometry {

void pybind_facet(py::module &m) {
	// cloudViewer.geometry.ccFacet
	py::class_<ccFacet, PyGeometry<ccFacet>, 
		std::shared_ptr<ccFacet>, ccPlanarEntityInterface, ccHObject>
		pyfacet(m, "ccFacet", py::multiple_inheritance(), "The Composite object: "
			"point cloud + 2D1/2 contour polyline + 2D1/2 surface mesh.");
	py::detail::bind_default_constructor<ccFacet>(pyfacet);
	py::detail::bind_copy_functions<ccFacet>(pyfacet);
	pyfacet.def(py::init([](PointCoordinateType max_edge_length, const std::string& name) {
			return new ccFacet(max_edge_length, name.c_str());
	}), "-param max_edge_length max edge length (if possible - ignored if 0); "
		"\n"
		"-param name name", 
		"max_edge_length"_a = 0, "name"_a = "Facet")
	.def("__repr__", [](const ccFacet &facet) {
			double area = facet.getSurface();
			double rms = facet.getRMS();
			unsigned facesNumber = 0;
			unsigned verticeNumber = 0;
			PointCoordinateType perimeter = 0.0f;
			if (facet.getPolygon())
			{
				facesNumber = facet.getPolygon()->size();
			}
			if (facet.getContour())
			{
				verticeNumber = facet.getContour()->size();
				perimeter = facet.getContour()->computeLength();
			}

			std::string info = fmt::format(
				"ccFacet (with {} faces, {} points, area {}, perimeter {}, rms {})",
				facesNumber, verticeNumber, area, perimeter, rms);
			return info;
		})
	.def(py::self + py::self)
	.def(py::self += py::self)
	.def("get_rms", &ccFacet::getRMS, "Returns associated RMS.")
	.def("get_area", &ccFacet::getSurface, "Returns associated surface area.")
	.def("get_normal_mesh", &ccFacet::getNormalVectorMesh, "Gets normal vector mesh.", "update"_a = false)
	.def("set_color", [](ccFacet& facet, const Eigen::Vector3d& color) {
			facet.setColor(ecvColor::Rgb::FromEigen(color));
		}, "Sets the facet unique color.", "color"_a)
	.def("get_center", [](const ccFacet& facet) {
			return CCVector3d::fromArray(facet.getCenter());
		}, "Returns the facet center.")
	.def("get_plane_equation", [](const ccFacet& facet) {
			const PointCoordinateType* eq = facet.getPlaneEquation();
			Eigen::Vector4d equation(4);
			for (size_t i = 0; i < 4; ++i)
			{
				equation(4) = eq[i];
			}
			return equation;
		}, "Returns Plane equation - as usual in CC plane equation is ax + by + cz = d")
	.def("invert_normal", &ccFacet::invertNormal, "Inverts the facet normal.")
	.def("get_polygon", [](ccFacet& facet) {
			if (facet.getPolygon()) {
				return std::ref(*facet.getPolygon());
			} else {
				cloudViewer::utility::LogWarning("[ccFacet] ccFacet do not have polygons!");
			}
		}, "Returns polygon mesh (if any)")
	.def("set_polygon", [](ccFacet& facet, ccMesh& mesh) {
			facet.setPolygon(&mesh);
		}, "Sets polygon mesh", "mesh"_a)
	.def("get_contour", [](ccFacet& facet) {
			if (facet.getContour()) {
				return std::ref(*facet.getContour());
			} else {
				cloudViewer::utility::LogWarning("[ccFacet] ccFacet do not have contours!");
			}
		}, "Returns contour polyline (if any)")
	.def("set_contour", [](ccFacet& facet, ccPolyline& poly) {
			facet.setContour(&poly);
		}, "Sets contour polyline", "poly"_a)
	.def("get_contour_vertices", [](ccFacet& facet) {
			if (facet.getContourVertices()) {
				return std::ref(*facet.getContourVertices());
			} else {
				cloudViewer::utility::LogWarning("[ccFacet] ccFacet do not have origin points!");
			}
		}, "Returns contour vertices (if any)")
	.def("set_contour_vertices", [](ccFacet& facet, ccPointCloud& vertices) {
			facet.setContourVertices(&vertices);
		}, "Sets contour vertices", "vertices"_a)
	.def("get_origin_points", [](ccFacet& facet) {
			if (facet.getOriginPoints()) {
				return std::ref(*facet.getOriginPoints());
			} else {
				cloudViewer::utility::LogWarning("[ccFacet] ccFacet do not have origin points!");
			}
		}, "Returns origin points (if any)")
	.def("set_origin_points", [](ccFacet& facet, ccPointCloud& cloud) {
			facet.setOriginPoints(&cloud);
		}, "Sets origin points", "cloud"_a)
	.def("clone", [](const ccFacet& facet) {
			return std::shared_ptr<ccFacet>(facet.clone());
		}, "Clones this facet.")
	.def("paint_uniform_color", &ccFacet::paintUniformColor,
		" Assigns facet the same color.", "color"_a)
	.def_static("Create", [](std::shared_ptr<ccPointCloud> cloud, 
				PointCoordinateType max_edge_length, bool transfer_ownership,
				const Eigen::Vector4d& plane_equation) {
			cloudViewer::GenericIndexedCloudPersist* persistCloud = cloud.get();
			if (!persistCloud)
			{
				cloudViewer::utility::LogWarning(
					"[ccFacet::Create] Illegal input parameters, only support point cloud!");
				return cloudViewer::make_shared<ccFacet>();
			}
			
			PointCoordinateType eq[4];
			for (size_t i = 0; i < 4; ++i)
			{
				eq[i] = static_cast<PointCoordinateType>(plane_equation(i));
			}
			ccFacet* facet = ccFacet::Create(persistCloud,
				max_edge_length, transfer_ownership, 
				plane_equation.isZero() ? nullptr : eq);
			return std::shared_ptr<ccFacet>(facet);
		}, "Creates a facet from a set of points", "cloud"_a, 
			"max_edge_length"_a = 0, "transfer_ownership"_a = false, 
			"plane_equation"_a = Eigen::Vector4d::Zero());
	
	docstring::ClassMethodDocInject(m, "ccFacet", "clone");
	docstring::ClassMethodDocInject(m, "ccFacet", "get_rms");
	docstring::ClassMethodDocInject(m, "ccFacet", "get_area");
	docstring::ClassMethodDocInject(m, "ccFacet", "set_color",
		{ {"color", "facet rgb color."} });
	docstring::ClassMethodDocInject(m, "ccFacet", "get_center");
	docstring::ClassMethodDocInject(m, "ccFacet", "get_normal_mesh");
	docstring::ClassMethodDocInject(m, "ccFacet", "get_plane_equation");
	docstring::ClassMethodDocInject(m, "ccFacet", "paint_uniform_color");
	docstring::ClassMethodDocInject(m, "ccFacet", "invert_normal");
	docstring::ClassMethodDocInject(m, "ccFacet", "get_polygon");
	docstring::ClassMethodDocInject(m, "ccFacet", "set_polygon");
	docstring::ClassMethodDocInject(m, "ccFacet", "get_contour");
	docstring::ClassMethodDocInject(m, "ccFacet", "set_contour");
	docstring::ClassMethodDocInject(m, "ccFacet", "get_contour_vertices");
	docstring::ClassMethodDocInject(m, "ccFacet", "set_contour_vertices");
	docstring::ClassMethodDocInject(m, "ccFacet", "get_origin_points");
	docstring::ClassMethodDocInject(m, "ccFacet", "set_origin_points");
	
}

void pybind_facet_methods(py::module &m) {}

}  // namespace geometry
}  // namespace cloudViewer