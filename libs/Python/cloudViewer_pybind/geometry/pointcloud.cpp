// ----------------------------------------------------------------------------
// -                        cloudViewer: www.cloudViewer.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.cloudViewer.org
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

#include <vector>

#include <Image.h>
#include <RGBDImage.h>
#include <ecvPointCloud.h>
#include <Camera/PinholeCameraIntrinsic.h>

#include "cloudViewer_pybind/docstring.h"
#include "cloudViewer_pybind/geometry/geometry.h"
#include "cloudViewer_pybind/geometry/geometry_trampoline.h"

using namespace cloudViewer;

void pybind_pointcloud(py::module &m) {

	py::class_<geometry::RansacResult> ransacResult(
		m, "RansacResult", "Ransac result.");
	py::detail::bind_default_constructor<geometry::RansacResult>(
		ransacResult);
	py::detail::bind_copy_functions<geometry::RansacResult>(
		ransacResult);
	ransacResult.def(py::init<>())
		.def("__repr__",
			[](const geometry::RansacResult &result) {
		return std::string(
			"geometry::RansacResult with "
			"points indices size = ") + std::to_string(result.indices.size()) +
			" and drawing precision = " + std::to_string(result.getDrawingPrecision()) +
			" and primitive type = " + result.getTypeName();
			})
		.def("get_type_name",
			&geometry::RansacResult::getTypeName,
			"Returns type name (sphere, cylinder, etc.).")
		.def("get_drawing_Precision",
			&geometry::RansacResult::getDrawingPrecision,
			"Returns drawing precision (or 0 if feature is not supported).")
		.def("set_drawing_Precision",
			&geometry::RansacResult::setDrawingPrecision,
			"Sets drawing precision." 
			"Warning: steps should always be >= "
			"ccGenericPrimitive::MIN_DRAWING_PRECISION = 4",
			"steps"_a)
		.def_readwrite("indices",
			&geometry::RansacResult::indices,
			"points indices.")
		.def_readwrite("primitive",
			&geometry::RansacResult::primitive,
			"The ransac primitive mesh shape.");
		docstring::ClassMethodDocInject(m, "RansacResult", "get_type_name");
		docstring::ClassMethodDocInject(m, "RansacResult", "get_drawing_Precision");
		docstring::ClassMethodDocInject(m, "RansacResult", "set_drawing_Precision",
			{ {"steps", "The steps drawing precision."} });

	// ccPointCloud::RansacParams
	py::class_<geometry::RansacParams> ransacParam(
		m, "RansacParams", "Ransac SD Parameters.");
	py::detail::bind_default_constructor<geometry::RansacParams>(ransacParam);
	py::detail::bind_copy_functions<geometry::RansacParams>(ransacParam);
	ransacParam.def(py::init<float>(), "scale"_a)
		.def("__repr__", [](const geometry::RansacParams &param) {
			return std::string(
			"geometry::RansacParams with "
			"epsilon = ") + std::to_string(param.epsilon) +
			" and bitmapEpsilon = " + std::to_string(param.bitmapEpsilon) +
			" and supportPoints = " + std::to_string(param.supportPoints) +
			" and maxNormalDev_deg = " + std::to_string(param.maxNormalDev_deg) +
			" and probability = " + std::to_string(param.probability) +
			" and randomColor = " + std::to_string(param.randomColor) +
			" and minRadius = " + std::to_string(param.minRadius) +
			" and maxRadius = " + std::to_string(param.maxRadius);
		})
		.def_readwrite("epsilon",
			&geometry::RansacParams::epsilon,
			"Distance threshold.")
		.def_readwrite("bit_map_epsilon",
			&geometry::RansacParams::bitmapEpsilon,
			"Bitmap resolution.")
		.def_readwrite("support_points",
			&geometry::RansacParams::supportPoints,
			"This is the minimal numer of points required for a primitive.")
		.def_readwrite("max_normal_deviation_deg",
			&geometry::RansacParams::maxNormalDev_deg,
			"Maximal normal deviation from ideal shape (in degrees).")
		.def_readwrite("probability",
			&geometry::RansacParams::probability,
			"Probability that no better candidate was overlooked during sampling.")
		.def_readwrite("random_color",
			&geometry::RansacParams::randomColor,
			"Should the resulting detected shapes sub point cloud be colored randomly.")
		.def_readwrite("prim_enabled_list",
			&geometry::RansacParams::primEnabled,
			"RANSAC PRIMITIVE TYPES.")
		.def_readwrite("min_radius",
			&geometry::RansacParams::minRadius,
			"Minimum radius threshold.")
		.def_readwrite(
			"max_radius", &geometry::RansacParams::maxRadius,
			"Maximum radius threshold.");

	// ccPointCloud::RansacParams::RANSAC_PRIMITIVE_TYPES
	py::enum_<geometry::RansacParams::RANSAC_PRIMITIVE_TYPES>
		ransac_primitive_type(ransacParam, "RANSAC_PRIMITIVE_TYPES", py::arithmetic());
	ransac_primitive_type
		.value("Plane", geometry::RansacParams::RPT_PLANE)
		.value("Sphere", geometry::RansacParams::RPT_SPHERE)
		.value("Cylinder", geometry::RansacParams::RPT_CYLINDER)
		.value("Cone", geometry::RansacParams::RPT_CONE)
		.value("Torus", geometry::RansacParams::RPT_TORUS)
		.export_values();
	ransac_primitive_type.attr("__doc__") = docstring::static_property(
		py::cpp_function([](py::handle arg) -> std::string {
			return "Enum class for Ransac Primitive types.";
		}), py::none(), py::none(), "");

	py::class_<ccPointCloud, PyGeometry<ccPointCloud>,
		std::shared_ptr<ccPointCloud>, ccHObject>
		pointcloud(m, "ccPointCloud", py::multiple_inheritance(),
			"ccPointCloud class. A point cloud consists of point "
			"coordinates, and optionally point colors and point "
			"normals.");
	py::detail::bind_default_constructor<ccPointCloud>(pointcloud);
	py::detail::bind_copy_functions<ccPointCloud>(pointcloud);
	pointcloud
		.def(py::init([](const std::string& name) {
				return new ccPointCloud(name.c_str());
			}), "name"_a = "cloud")
		.def(py::init<const std::vector<Eigen::Vector3d>&, const std::string&>(),
			"Create a PointCloud from points", "points"_a, "name"_a = "cloud")
		.def("__repr__", [](const ccPointCloud &pcd) {
		return std::string("ccPointCloud with ") +
			std::to_string(pcd.size()) + " points.";
		})
		.def(py::self + py::self)
		.def(py::self += py::self)
		.def("size", &ccPointCloud::size,
			"Returns points number.")
		.def("has_points", &ccPointCloud::hasPoints,
			"Returns ``True`` if the point cloud contains points.")
		.def("has_normals", &ccPointCloud::hasNormals,
			"Returns ``True`` if the point cloud contains point normals.")
		.def("has_colors", &ccPointCloud::hasColors,
			"Returns ``True`` if the point cloud contains point colors.")
		.def("normalize_normals", &ccPointCloud::normalizeNormals,
			"Normalize point normals to length 1.")
		.def("paint_uniform_color",
			&ccPointCloud::paintUniformColor, "color"_a,
			"Assigns each point in the PointCloud the same color.")
		.def("select_by_index", &ccPointCloud::selectByIndex,
			"Function to select points from input pointcloud into output "
			"pointcloud.",
			"indices"_a, "invert"_a = false)
		.def("voxel_down_sample", &ccPointCloud::voxelDownSample,
			"Function to downsample input pointcloud into output "
			"pointcloud with "
			"a voxel. Normals and colors are averaged if they exist.",
			"voxel_size"_a)
		.def("voxel_down_sample_and_trace",
			&ccPointCloud::voxelDownSampleAndTrace,
			"Function to downsample using "
			"ccPointCloud::VoxelDownSample. Also records point "
			"cloud index before downsampling",
			"voxel_size"_a, "min_bound"_a, "max_bound"_a,
			"approximate_class"_a = false)
		.def("uniform_down_sample",
			&ccPointCloud::uniformDownSample,
			"Function to downsample input pointcloud into output "
			"pointcloud "
			"uniformly. The sample is performed in the order of the "
			"points with "
			"the 0-th point always chosen, not at random.",
			"every_k_points"_a)
		.def("crop", (std::shared_ptr<ccPointCloud>(ccPointCloud::*)(
			const ccBBox &) const) &
			ccPointCloud::Crop,
			"Function to crop input pointcloud into output pointcloud",
			"bounding_box"_a)
		.def("crop", (std::shared_ptr<ccPointCloud>(ccPointCloud::*)(
			const ecvOrientedBBox &) const) &
			ccPointCloud::Crop,
			"Function to crop input pointcloud into output pointcloud",
			"bounding_box"_a)
		.def("remove_non_finite_points",
			&ccPointCloud::removeNonFinitePoints,
			"Function to remove non-finite points from the PointCloud",
			"remove_nan"_a = true, "remove_infinite"_a = true)
		.def("remove_radius_outlier",
			&ccPointCloud::removeRadiusOutliers,
			"Function to remove points that have less than nb_points"
			" in a given sphere of a given radius",
			"nb_points"_a, "radius"_a)
		.def("remove_statistical_outlier",
			&ccPointCloud::removeStatisticalOutliers,
			"Function to remove points that are further away from their "
			"neighbors in average",
			"nb_neighbors"_a, "std_ratio"_a)
		.def("estimate_normals", &ccPointCloud::estimateNormals,
			"Function to compute the normals of a point cloud. Normals "
			"are oriented with respect to the input point cloud if "
			"normals exist",
			"search_param"_a = geometry::KDTreeSearchParamKNN(),
			"fast_normal_computation"_a = true)
		.def("orient_normals_to_align_with_direction",
			&ccPointCloud::orientNormalsToAlignWithDirection,
			"Function to orient the normals of a point cloud",
			"orientation_reference"_a = Eigen::Vector3d(0.0, 0.0, 1.0))
		.def("orient_normals_towards_camera_location",
			&ccPointCloud::orientNormalsTowardsCameraLocation,
			"Function to orient the normals of a point cloud",
			"camera_location"_a = Eigen::Vector3d(0.0, 0.0, 0.0))
		.def("compute_point_cloud_distance",
			&ccPointCloud::computePointCloudDistance,
			"For each point in the source point cloud, compute the "
			"distance to "
			"the target point cloud.",
			"target"_a)
		.def("compute_mean_and_covariance",
			&ccPointCloud::computeMeanAndCovariance,
			"Function to compute the mean and covariance matrix of a "
			"point "
			"cloud.")
		.def("compute_mahalanobis_distance",
			&ccPointCloud::computeMahalanobisDistance,
			"Function to compute the Mahalanobis distance for points in a "
			"point "
			"cloud. See: "
			"https://en.wikipedia.org/wiki/Mahalanobis_distance.")
		.def("compute_nearest_neighbor_distance",
			&ccPointCloud::computeNearestNeighborDistance,
			"Function to compute the distance from a point to its nearest "
			"neighbor in the point cloud")
		.def("compute_resolution",
			&ccPointCloud::computeResolution,
			"Function to compute the point cloud resolution.")
		.def("compute_convex_hull",
			&ccPointCloud::computeConvexHull,
			"Computes the convex hull of the point cloud.")
		.def("hidden_point_removal",
			&ccPointCloud::hiddenPointRemoval,
			"Removes hidden points from a point cloud and returns a mesh "
			"of the remaining points. Based on Katz et al. 'Direct "
			"Visibility of Point Sets', 2007. Additional information "
			"about the choice of radius for noisy point clouds can be "
			"found in Mehra et. al. 'Visibility of Noisy Point Cloud "
			"Data', 2010.",
			"camera_location"_a, "radius"_a)
		.def("cluster_dbscan", &ccPointCloud::clusterDBSCAN,
			"Cluster PointCloud using the DBSCAN algorithm  Ester et al., "
			"'A Density-Based Algorithm for Discovering Clusters in Large "
			"Spatial Databases with Noise', 1996. Returns a list of point "
			"labels, -1 indicates noise according to the algorithm.",
			"eps"_a, "min_points"_a, "print_progress"_a = false)
		.def("execute_ransac", &ccPointCloud::executeRANSAC,
			"Cluster ccPointCloud using the RANSAC algorithm, "
			"Wrapper to Schnabel et al. library for automatic"
			" shape detection in point cloud ,Efficient RANSAC"
			" for Point - Cloud Shape Detection, Ruwen Schnabel, "
			"Roland Wahl, Returns a list of ransac point labels"
			" and shape entity(ccGenericPrimitive)",
			"params"_a = geometry::RansacParams(), "print_progress"_a = false)
		.def("segment_plane", &ccPointCloud::segmentPlane,
			"Segments a plane in the point cloud using the RANSAC "
			"algorithm.",
			"distance_threshold"_a, "ransac_n"_a, "num_iterations"_a)
		.def("set_point", &ccPointCloud::setEigenPoint,
			"set point coordinate by given index.",
			"index"_a, "point"_a)
		.def("get_point", &ccPointCloud::getEigenPoint,
			"get point coordinate by given index.",
			"index"_a)
		.def("set_points", py::overload_cast<
			const std::vector<Eigen::Vector3d>&>(&ccPointCloud::addPoints),
			"``float64`` array of shape ``(num_points, 3)``, "
			"use ``numpy.asarray()`` to access data: Points "
			"coordinates.",
			"points"_a)
		.def("get_points", &ccPointCloud::getEigenPoints,
			"``float64`` array of shape ``(num_points, 3)``, "
			"use ``numpy.asarray()`` to access data: Points "
			"coordinates.")
		.def("set_color", &ccPointCloud::setEigenColor,
			"set point color by given index.",
			"index"_a, "color"_a)
		.def("get_color", &ccPointCloud::getEigenColor,
			"get point color by given index.",
			"index"_a)
		.def("set_colors", &ccPointCloud::addEigenColors,
			"``float64`` array of shape ``(num_points, 3)``, "
			"range ``[0, 1]`` , use ``numpy.asarray()`` to access "
			"data: RGB colors of points.",
			"colors"_a)
		.def("get_colors", &ccPointCloud::getEigenColors,
			"``float64`` array of shape ``(num_points, 3)``, "
			"range ``[0, 1]`` , use ``numpy.asarray()`` to access "
			"data: RGB colors of points.")
		.def("set_normal", &ccPointCloud::setEigenNormal,
			"set point normal by given index.",
			"index"_a, "normal"_a)
		.def("get_normal", &ccPointCloud::getEigenNormal,
			"get point normal by given index.",
			"index"_a)
		.def("set_normals", &ccPointCloud::addEigenNorms,
			"``float64`` array of shape ``(num_points, 3)``, "
			"use ``numpy.asarray()`` to access data: Points "
			"normals.",
			"normals"_a)
		.def("get_normals", &ccPointCloud::getEigenNormals,
			"``float64`` array of shape ``(num_points, 3)``, "
			"use ``numpy.asarray()`` to access data: Points "
			"normals.")
		.def_static(
			"create_from_depth_image",
			&ccPointCloud::CreateFromDepthImage,
			R"(Factory function to create a pointcloud from a depth image and a
			camera. Given depth value d at (u, v) image coordinate, the corresponding 3d
			point is:

				  - z = d / depth_scale
				  - x = (u - cx) * z / fx
				  - y = (v - cy) * z / fy
			)",
			"depth"_a, "intrinsic"_a,
			"extrinsic"_a = Eigen::Matrix4d::Identity(),
			"depth_scale"_a = 1000.0, "depth_trunc"_a = 1000.0,
			"stride"_a = 1, "project_valid_depth_only"_a = true)
		.def_static("create_from_rgbd_image",
			&ccPointCloud::CreateFromRGBDImage,
			"Factory function to create a pointcloud from an RGB-D "
			"image and a camera. Given depth value d at (u, "
			"v) image coordinate, the corresponding 3d point is: "
			R"(- z = d / depth_scale
						  - x = (u - cx) * z / fx
						  - y = (v - cy) * z / fy)",
			"image"_a, "intrinsic"_a,
			"extrinsic"_a = Eigen::Matrix4d::Identity(),
			"project_valid_depth_only"_a = true);
	docstring::ClassMethodDocInject(m, "ccPointCloud", "size");
	docstring::ClassMethodDocInject(m, "ccPointCloud", "has_colors");
	docstring::ClassMethodDocInject(m, "ccPointCloud", "has_normals");
	docstring::ClassMethodDocInject(m, "ccPointCloud", "has_points");
	docstring::ClassMethodDocInject(m, "ccPointCloud", "set_point");
	docstring::ClassMethodDocInject(m, "ccPointCloud", "set_points");
	docstring::ClassMethodDocInject(m, "ccPointCloud", "get_point");
	docstring::ClassMethodDocInject(m, "ccPointCloud", "get_points");
	docstring::ClassMethodDocInject(m, "ccPointCloud", "set_color");
	docstring::ClassMethodDocInject(m, "ccPointCloud", "set_colors");
	docstring::ClassMethodDocInject(m, "ccPointCloud", "get_color");
	docstring::ClassMethodDocInject(m, "ccPointCloud", "get_colors");
	docstring::ClassMethodDocInject(m, "ccPointCloud", "set_normal");
	docstring::ClassMethodDocInject(m, "ccPointCloud", "set_normals");
	docstring::ClassMethodDocInject(m, "ccPointCloud", "get_normal");
	docstring::ClassMethodDocInject(m, "ccPointCloud", "get_normals");
	docstring::ClassMethodDocInject(m, "ccPointCloud", "normalize_normals");
	docstring::ClassMethodDocInject(
		m, "ccPointCloud", "paint_uniform_color",
		{ {"color", "RGB color for the PointCloud."} });
	docstring::ClassMethodDocInject(
		m, "ccPointCloud", "select_by_index",
		{ {"indices", "Indices of points to be selected."},
		 {"invert",
		  "Set to ``True`` to invert the selection of indices."} });
	docstring::ClassMethodDocInject(
		m, "ccPointCloud", "voxel_down_sample",
		{ {"voxel_size", "Voxel size to downsample into."},
		 {"invert", "set to ``True`` to invert the selection of indices"} });
	docstring::ClassMethodDocInject(
		m, "ccPointCloud", "voxel_down_sample_and_trace",
		{ {"voxel_size", "Voxel size to downsample into."},
		 {"min_bound", "Minimum coordinate of voxel boundaries"},
		 {"max_bound", "Maximum coordinate of voxel boundaries"} });
	docstring::ClassMethodDocInject(
		m, "ccPointCloud", "uniform_down_sample",
		{ {"every_k_points",
		  "Sample rate, the selected point indices are [0, k, 2k, ...]"} });
	docstring::ClassMethodDocInject(
		m, "ccPointCloud", "crop",
		{ {"bounding_box", "AxisAlignedBoundingBox to crop points"} });
	docstring::ClassMethodDocInject(
		m, "ccPointCloud", "remove_non_finite_points",
		{ {"remove_nan", "Remove NaN values from the PointCloud"},
		 {"remove_infinite",
		  "Remove infinite values from the PointCloud"} });
	docstring::ClassMethodDocInject(
		m, "ccPointCloud", "remove_radius_outlier",
		{ {"nb_points", "Number of points within the radius."},
		 {"radius", "Radius of the sphere."} });
	docstring::ClassMethodDocInject(
		m, "ccPointCloud", "remove_statistical_outlier",
		{ {"nb_neighbors", "Number of neighbors around the target point."},
		 {"std_ratio", "Standard deviation ratio."} });
	docstring::ClassMethodDocInject(
		m, "ccPointCloud", "estimate_normals",
		{ {"search_param",
		  "The KDTree search parameters for neighborhood search."},
		 {"fast_normal_computation",
		  "If true, the normal estiamtion uses a non-iterative method to "
		  "extract the eigenvector from the covariance matrix. This is "
		  "faster, but is not as numerical stable."} });
	docstring::ClassMethodDocInject(
		m, "ccPointCloud", "orient_normals_to_align_with_direction",
		{ {"orientation_reference",
		  "Normals are oriented with respect to orientation_reference."} });
	docstring::ClassMethodDocInject(
		m, "ccPointCloud", "orient_normals_towards_camera_location",
		{ {"camera_location",
		  "Normals are oriented with towards the camera_location."} });
	docstring::ClassMethodDocInject(m, "ccPointCloud",
		"compute_point_cloud_distance",
		{ {"target", "The target point cloud."} });
	docstring::ClassMethodDocInject(m, "ccPointCloud",
		"compute_mean_and_covariance");
	docstring::ClassMethodDocInject(m, "ccPointCloud",
		"compute_mahalanobis_distance");
	docstring::ClassMethodDocInject(m, "ccPointCloud",
		"compute_nearest_neighbor_distance");
	docstring::ClassMethodDocInject(m, "ccPointCloud", "compute_resolution");
	docstring::ClassMethodDocInject(m, "ccPointCloud", "compute_convex_hull",
		{ {"input", "The input point cloud."} });
	docstring::ClassMethodDocInject(
		m, "ccPointCloud", "hidden_point_removal",
		{ {"input", "The input point cloud."},
		 {"camera_location",
		  "All points not visible from that location will be reomved"},
		 {"radius", "The radius of the sperical projection"} });
	docstring::ClassMethodDocInject(
		m, "ccPointCloud", "cluster_dbscan",
		{ {"eps",
		  "Density parameter that is used to find neighbouring points."},
		 {"min_points", "Minimum number of points to form a cluster."},
		 {"print_progress",
		  "If true the progress is visualized in the console."} });
	docstring::ClassMethodDocInject(
		m, "ccPointCloud", "segment_plane",
		{ {"distance_threshold",
		  "Max distance a point can be from the plane model, and still be "
		  "considered an inlier."},
		 {"ransac_n",
		  "Number of initial points to be considered inliers in each "
		  "iteration."},
		 {"num_iterations", "Number of iterations."} });
	docstring::ClassMethodDocInject(
		m, "ccPointCloud", "create_from_depth_image",
		{ {"depth",
		  "The input depth image can be either a float image, or a "
		  "uint16_t image."},
		 {"intrinsic", "Intrinsic parameters of the camera."},
		 {"extrnsic", "Extrinsic parameters of the camera."},
		 {"depth_scale", "The depth is scaled by 1 / depth_scale."},
		 {"depth_trunc", "Truncated at depth_trunc distance."},
		 {"stride",
		  "Sampling factor to support coarse point cloud extraction."} });
	docstring::ClassMethodDocInject(
		m, "ccPointCloud", "create_from_rgbd_image",
		{ {"image", "The input image."},
		 {"intrinsic", "Intrinsic parameters of the camera."},
		 {"extrnsic", "Extrinsic parameters of the camera."} });
}

void pybind_pointcloud_methods(py::module &m) {}
