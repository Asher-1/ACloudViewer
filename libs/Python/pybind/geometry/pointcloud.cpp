// ----------------------------------------------------------------------------
// -                        CloudViewer: asher-1.github.io                    -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 asher-1.github.io
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

#include <Image.h>
#include <RGBDImage.h>
#include <ecvPolyline.h>
#include <ecvPointCloud.h>
#include <camera/PinholeCameraIntrinsic.h>

#include <vector>

#include "pybind/docstring.h"
#include "pybind/geometry/cloudbase.h"
#include "pybind/geometry/geometry.h"
#include "pybind/geometry/geometry_trampoline.h"

#ifdef _MSC_VER
#pragma warning(disable : 4715)
#endif

namespace cloudViewer {
namespace geometry {

void pybind_pointcloud(py::module& m) {
#ifdef CV_RANSAC_SUPPORT
    py::class_<geometry::RansacResult> ransacResult(m, "RansacResult",
                                                    "Ransac result.");
    py::detail::bind_default_constructor<geometry::RansacResult>(ransacResult);
    py::detail::bind_copy_functions<geometry::RansacResult>(ransacResult);
    ransacResult.def(py::init<>())
            .def("__repr__",
                 [](const geometry::RansacResult& result) {
                     return std::string(
                                    "geometry::RansacResult with "
                                    "points indices size = ") +
                            std::to_string(result.indices.size()) +
                            " , drawing precision = " +
                            std::to_string(result.getDrawingPrecision()) +
                            " and primitive type = " + result.getTypeName();
                 })
            .def("get_type_name", &geometry::RansacResult::getTypeName,
                 "Returns type name (sphere, cylinder, etc.).")
            .def("get_drawing_Precision",
                 &geometry::RansacResult::getDrawingPrecision,
                 "Returns drawing precision (or 0 if feature is not "
                 "supported).")
            .def("set_drawing_Precision",
                 &geometry::RansacResult::setDrawingPrecision,
                 "Sets drawing precision."
                 "Warning: steps should always be >= "
                 "ccGenericPrimitive::MIN_DRAWING_PRECISION = 4",
                 "steps"_a)
            .def_readwrite("indices", &geometry::RansacResult::indices,
                           "points indices.")
            .def_readwrite("primitive", &geometry::RansacResult::primitive,
                           "The ransac primitive mesh shape.");
    docstring::ClassMethodDocInject(m, "RansacResult", "get_type_name");
    docstring::ClassMethodDocInject(m, "RansacResult", "get_drawing_Precision");
    docstring::ClassMethodDocInject(
            m, "RansacResult", "set_drawing_Precision",
            {{"steps", "The steps drawing precision."}});

    // ccPointCloud::RansacParams
    py::class_<geometry::RansacParams> ransacParam(m, "RansacParams",
                                                   "Ransac SD Parameters.");
    py::detail::bind_default_constructor<geometry::RansacParams>(ransacParam);
    py::detail::bind_copy_functions<geometry::RansacParams>(ransacParam);
    ransacParam.def(py::init<float>(), "scale"_a)
            .def("__repr__",
                 [](const geometry::RansacParams& param) {
                     return std::string(
                                    "geometry::RansacParams with "
                                    "epsilon = ") +
                            std::to_string(param.epsilon) +
                            " and bitmapEpsilon = " +
                            std::to_string(param.bitmapEpsilon) +
                            " and supportPoints = " +
                            std::to_string(param.supportPoints) +
                            " and maxNormalDev_deg = " +
                            std::to_string(param.maxNormalDev_deg) +
                            " and probability = " +
                            std::to_string(param.probability) +
                            " and randomColor = " +
                            std::to_string(param.randomColor) +
                            " and minRadius = " +
                            std::to_string(param.minRadius) +
                            " and maxRadius = " +
                            std::to_string(param.maxRadius);
                 })
            .def_readwrite("epsilon", &geometry::RansacParams::epsilon,
                           "Distance threshold.")
            .def_readwrite("bit_map_epsilon",
                           &geometry::RansacParams::bitmapEpsilon,
                           "Bitmap resolution.")
            .def_readwrite("support_points",
                           &geometry::RansacParams::supportPoints,
                           "This is the minimal numer of points required for a "
                           "primitive.")
            .def_readwrite(
                    "max_normal_deviation_deg",
                    &geometry::RansacParams::maxNormalDev_deg,
                    "Maximal normal deviation from ideal shape (in degrees).")
            .def_readwrite("probability", &geometry::RansacParams::probability,
                           "Probability that no better candidate was "
                           "overlooked during sampling.")
            .def_readwrite("random_color", &geometry::RansacParams::randomColor,
                           "Should the resulting detected shapes sub point "
                           "cloud be colored randomly.")
            .def_readwrite("prim_enabled_list",
                           &geometry::RansacParams::primEnabled,
                           "RANSAC PRIMITIVE TYPES.")
            .def_readwrite("min_radius", &geometry::RansacParams::minRadius,
                           "Minimum radius threshold.")
            .def_readwrite("max_radius", &geometry::RansacParams::maxRadius,
                           "Maximum radius threshold.");

    // ccPointCloud::RansacParams::RANSAC_PRIMITIVE_TYPES
    py::enum_<geometry::RansacParams::RANSAC_PRIMITIVE_TYPES>
            ransac_primitive_type(ransacParam, "RANSAC_PRIMITIVE_TYPES",
                                  py::arithmetic());
    ransac_primitive_type.value("Plane", geometry::RansacParams::RPT_PLANE)
            .value("Sphere", geometry::RansacParams::RPT_SPHERE)
            .value("Cylinder", geometry::RansacParams::RPT_CYLINDER)
            .value("Cone", geometry::RansacParams::RPT_CONE)
            .value("Torus", geometry::RansacParams::RPT_TORUS)
            .export_values();
    ransac_primitive_type.attr("__doc__") = docstring::static_property(
            py::cpp_function([](py::handle arg) -> std::string {
                return "Enum class for Ransac Primitive types.";
            }),
            py::none(), py::none(), "");
#endif

    py::class_<ccPointCloud, PyGeometry<ccPointCloud>,
               std::shared_ptr<ccPointCloud>, ccGenericPointCloud, ccHObject>
            pointcloud(m, "ccPointCloud", py::multiple_inheritance(),
                       "ccPointCloud class. A point cloud consists of point "
                       "coordinates, and optionally point colors and point "
                       "normals.");
    py::detail::bind_default_constructor<ccPointCloud>(pointcloud);
    py::detail::bind_copy_functions<ccPointCloud>(pointcloud);
    pointcloud
            .def(py::init([](const std::string& name) {
                     return new ccPointCloud(name.c_str());
                 }),
                 "name"_a = "cloud")
            .def(py::init<const std::vector<Eigen::Vector3d>&,
                          const std::string&>(),
                 "Create a PointCloud from points", "points"_a,
                 "name"_a = "cloud")
            .def("__repr__",
                 [](const ccPointCloud& pcd) {
                     return std::string("ccPointCloud with ") +
                            std::to_string(pcd.size()) + " points.";
                 })
            .def(py::self + py::self)
            .def(py::self += py::self)
            .def("has_covariances", &ccPointCloud::hasCovariances,
                 "Returns ``True`` if the point cloud contains covariances.")
            .def("normalize_normals", &ccPointCloud::normalizeNormals,
                 "Normalize point normals to length 1.")
            .def("paint_uniform_color", &ccPointCloud::paintUniformColor,
                 "color"_a,
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
                 "cloud index before down sampling",
                 "voxel_size"_a, "min_bound"_a, "max_bound"_a,
                 "approximate_class"_a = false)
            .def("uniform_down_sample", &ccPointCloud::uniformDownSample,
                 "Function to downsample input pointcloud into output "
                 "pointcloud "
                 "uniformly. The sample is performed in the order of the "
                 "points with "
                 "the 0-th point always chosen, not at random.",
                 "every_k_points"_a)
            .def("random_down_sample", &ccPointCloud::randomDownSample,
                 "Function to downsample input pointcloud into output "
                 "pointcloud "
                 "randomly. The sample is generated by randomly sampling "
                 "the indexes from the point cloud.",
                 "sampling_ratio"_a)
            .def("crop",
                 (std::shared_ptr<ccPointCloud>(ccPointCloud::*)(const ccBBox&)
                          const) &
                         ccPointCloud::Crop,
                 "Function to crop input pointcloud into output pointcloud",
                 "bounding_box"_a)
            .def("crop",
                 (std::shared_ptr<ccPointCloud>(ccPointCloud::*)(
                         const ecvOrientedBBox&) const) &
                         ccPointCloud::Crop,
                 "Function to crop input pointcloud into output pointcloud",
                 "bounding_box"_a)
            .def("remove_non_finite_points",
                 &ccPointCloud::removeNonFinitePoints,
                 "Function to remove non-finite points from the PointCloud",
                 "remove_nan"_a = true, "remove_infinite"_a = true)
            .def("remove_radius_outlier", &ccPointCloud::removeRadiusOutliers,
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
            .def("orient_normals_consistent_tangent_plane",
                 &ccPointCloud::orientNormalsConsistentTangentPlane,
                 "Function to orient the normals with respect to consistent "
                 "tangent planes",
                 "k"_a)
            .def("compute_point_cloud_distance",
                 &ccPointCloud::computePointCloudDistance,
                 "For each point in the source point cloud, compute the "
                 "distance to "
                 "the target point cloud.",
                 "target"_a)
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
            .def("compute_resolution", &ccPointCloud::computeResolution,
                 "Function to compute the point cloud resolution.")
            .def("compute_convex_hull", &ccPointCloud::computeConvexHull,
                 "Computes the convex hull of the point cloud.")
            .def("hidden_point_removal", &ccPointCloud::hiddenPointRemoval,
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
#ifdef CV_RANSAC_SUPPORT
            .def("execute_ransac", &ccPointCloud::executeRANSAC,
                 "Cluster ccPointCloud using the RANSAC algorithm, "
                 "Wrapper to Schnabel et al. library for automatic"
                 " shape detection in point cloud ,Efficient RANSAC"
                 " for Point - Cloud Shape Detection, Ruwen Schnabel, "
                 "Roland Wahl, Returns a list of ransac point labels"
                 " and shape entity(ccGenericPrimitive)",
                 "params"_a = geometry::RansacParams(),
                 "print_progress"_a = false)
#endif
            .def("segment_plane", &ccPointCloud::segmentPlane,
                 "Segments a plane in the point cloud using the RANSAC "
                 "algorithm.",
                 "distance_threshold"_a, "ransac_n"_a, "num_iterations"_a)
            .def("set_point", &ccPointCloud::setEigenPoint,
                 "set point coordinate by given index.", "index"_a, "point"_a)
            .def("get_point", &ccPointCloud::getEigenPoint,
                 "get point coordinate by given index.", "index"_a)
            .def("set_points",
                 py::overload_cast<const std::vector<Eigen::Vector3d>&>(
                         &ccPointCloud::addPoints),
                 "``float64`` array of shape ``(num_points, 3)``, "
                 "use ``numpy.asarray()`` to access data: Points "
                 "coordinates.",
                 "points"_a)
            .def("get_points", &ccPointCloud::getEigenPoints,
                 "``float64`` array of shape ``(num_points, 3)``, "
                 "use ``numpy.asarray()`` to access data: Points "
                 "coordinates.")
            .def("set_color", &ccPointCloud::setEigenColor,
                 "set point color by given index.", "index"_a, "color"_a)
            .def("get_color", &ccPointCloud::getEigenColor,
                 "get point color by given index.", "index"_a)
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
                 "set point normal by given index.", "index"_a, "normal"_a)
            .def("get_normal", &ccPointCloud::getEigenNormal,
                 "get point normal by given index.", "index"_a)
            .def("set_normals", &ccPointCloud::addEigenNorms,
                 "``float64`` array of shape ``(num_points, 3)``, "
                 "use ``numpy.asarray()`` to access data: Points normals.",
                 "normals"_a)
            .def("get_normals", &ccPointCloud::getEigenNormals,
                 "``float64`` array of shape ``(num_points, 3)``, "
                 "use ``numpy.asarray()`` to access data: Points normals.")

            .def("unalloacte_points", &ccPointCloud::unalloactePoints,
                 "Erases the cloud points.")
            .def("unalloacte_colors", &ccPointCloud::unallocateColors,
                 "Erases the cloud colors.")
            .def("unalloacte_norms", &ccPointCloud::unallocateNorms,
                 "Erases the cloud normals.")
            .def("colors_have_changed", &ccPointCloud::colorsHaveChanged,
                 R"(Notify a modification of color/scalar field display parameters or contents.)")
            .def("normals_have_changed", &ccPointCloud::normalsHaveChanged,
                 "Notify a modification of normals display parameters or "
                 "contents.")
            .def("points_have_changed", &ccPointCloud::pointsHaveChanged,
                 "Notify a modification of points display parameters or "
                 "contents.")
            .def("reserve", &ccPointCloud::reserve,
                 "Reserves memory for all the active features.",
                 "points_number"_a)
            .def("reserve_points", &ccPointCloud::reserveThePointsTable,
                 "Reserves memory to store the points coordinates.",
                 "points_number"_a)
            .def("reserve_colors", &ccPointCloud::reserveTheRGBTable,
                 "Reserves memory to store the RGB colors.")
            .def("reserve_norms", &ccPointCloud::reserveTheNormsTable,
                 "Reserves memory to store the compressed normals.")
            .def("resize", &ccPointCloud::resize,
                 "Resizes all the active features arrays.", "points_number"_a)
            .def("resize_colors", &ccPointCloud::resizeTheRGBTable,
                 "Resizes the RGB colors array.", "fill_with_white"_a = false)
            .def("resize_norms", &ccPointCloud::reserveTheNormsTable,
                 "Resizes the compressed normals array.")
            .def("shrink", &ccPointCloud::shrinkToFit,
                 "Removes unused capacity.")
            .def("reset", &ccPointCloud::reset, " Clears the cloud database.")
            .def("capacity", &ccPointCloud::capacity,
                 "Returns cloud capacity (i.e. reserved size).")
            .def("invalidate_bbox", &ccPointCloud::invalidateBoundingBox,
                 "Invalidates bounding box.")
            .def("set_current_input_sf_index",
                 &ccPointCloud::setCurrentInScalarField,
                 " Sets the INPUT scalar field index (or -1 if none).",
                 "index"_a)
            .def("get_current_input_sf_index",
                 &ccPointCloud::getCurrentInScalarFieldIndex,
                 "Returns current INPUT scalar field index (or -1 if none).")
            .def("set_current_output_sf_index",
                 &ccPointCloud::setCurrentOutScalarField,
                 " Sets the OUTPUT scalar field index (or -1 if none).",
                 "index"_a)
            .def("get_current_output_sf_index",
                 &ccPointCloud::getCurrentOutScalarFieldIndex,
                 "Returns current OUTPUT scalar field index (or -1 if none).")
            .def("set_current_sf", &ccPointCloud::setCurrentScalarField,
                 " Sets both the INPUT & OUTPUT scalar field.", "index"_a)
            .def("sfs_count", &ccPointCloud::getNumberOfScalarFields,
                 "Returns the number of associated (and active) scalar fields.")
            .def(
                    "get_sf_by_index",
                    [](const ccPointCloud& cloud, std::size_t index) {
                        if (cloud.getScalarField(static_cast<int>(index))) {
                            return std::ref(*cloud.getScalarField(
                                    static_cast<int>(index)));
                        } else {
                            cloudViewer::utility::LogWarning(
                                    "[ccPointCloud] Do not have any scalar "
                                    "field!");
                        }
                    },
                    "Returns a pointer to a specific scalar field.", "index"_a)
            .def(
                    "get_sf_name",
                    [](const ccPointCloud& cloud, std::size_t index) {
                        return std::string(cloud.getScalarFieldName(
                                static_cast<int>(index)));
                    },
                    "Returns the name of a specific scalar field.", "index"_a)
            .def(
                    "get_sf_index_by_name",
                    [](const ccPointCloud& cloud, const std::string& name) {
                        return cloud.getScalarFieldIndexByName(name.c_str());
                    },
                    "Returns the index of a scalar field represented by its "
                    "name.",
                    "name"_a)
            .def(
                    "get_current_input_sf",
                    [](const ccPointCloud& cloud) {
                        if (cloud.getCurrentInScalarField()) {
                            return std::ref(*cloud.getCurrentInScalarField());
                        } else {
                            cloudViewer::utility::LogWarning(
                                    "[ccPointCloud] cloud input does not have "
                                    "any scalar field!");
                        }
                    },
                    "Returns the scalar field currently associated to the "
                    "cloud input.")
            .def(
                    "get_current_output_sf",
                    [](const ccPointCloud& cloud) {
                        if (cloud.getCurrentOutScalarField()) {
                            return std::ref(*cloud.getCurrentOutScalarField());
                        } else {
                            cloudViewer::utility::LogWarning(
                                    "[ccPointCloud] cloud output does not have "
                                    "any scalar field!");
                        }
                    },
                    "Returns the scalar field currently associated to the "
                    "cloud output.")
            .def(
                    "rename_sf",
                    [](ccPointCloud& cloud, std::size_t index,
                       const std::string& name) {
                        return cloud.renameScalarField(static_cast<int>(index),
                                                       name.c_str());
                    },
                    "Renames a specific scalar field.", "index"_a, "name"_a)
            .def(
                    "get_current_displayed_sf",
                    [](const ccPointCloud& cloud) {
                        if (cloud.getCurrentDisplayedScalarField()) {
                            return std::ref(
                                    *cloud.getCurrentDisplayedScalarField());
                        } else {
                            cloudViewer::utility::LogWarning(
                                    "[ccPointCloud] Do not have scalar "
                                    "fields!");
                        }
                    },
                    "Returns the currently displayed scalar (or 0 if none).")
            .def("set_current_displayed_sf",
                 &ccPointCloud::setCurrentDisplayedScalarField,
                 "Sets the currently displayed scalar field.", "index"_a)
            .def("get_current_sf_index",
                 &ccPointCloud::getCurrentDisplayedScalarFieldIndex,
                 "Returns the currently displayed scalar field index (or -1 if "
                 "none).")
            .def("delete_sf", &ccPointCloud::deleteScalarField,
                 "Deletes a specific scalar field.", "index"_a)
            .def("delete_all_sfs", &ccPointCloud::deleteAllScalarFields,
                 "Deletes all scalar fields associated to this cloud.")
            .def(
                    "add_sf",
                    [](ccPointCloud& cloud, const std::string& unique_name) {
                        return cloud.addScalarField(unique_name.c_str());
                    },
                    "Creates a new scalar field and registers it.",
                    "unique_name"_a)
            .def(
                    "add_sf",
                    [](ccPointCloud& cloud, ccScalarField& sf) {
                        return cloud.addScalarField(&sf);
                    },
                    "Creates a new scalar field and registers it.", "sf"_a)
            .def("sf_color_scale_shown", &ccPointCloud::sfColorScaleShown,
                 "Returns whether color scale should be displayed or not.")
            .def("show_sf_color_scale", &ccPointCloud::showSFColorsScale,
                 "Sets whether color scale should be displayed or not.",
                 "state"_a)
            .def(
                    "compute_gravity_center",
                    [](ccPointCloud& cloud) {
                        return CCVector3d::fromArray(
                                cloud.computeGravityCenter());
                    },
                    "Returns the cloud gravity center")
            .def(
                    "crop_2d",
                    [](ccPointCloud& cloud, const ccPolyline& polyline,
                       unsigned char ortho_dim, bool inside) {
                        cloudViewer::ReferenceCloud* ref =
                                cloud.crop2D(&polyline, ortho_dim, inside);
                        if (!ref || ref->size() == 0) {
                            if (ref) {
                                delete ref;
                            }
                            ref = nullptr;
                            if (polyline.isEmpty()) {
                                cloudViewer::utility::LogWarning(
                                        "[ccPointCloud::crop2D] Invalid input "
                                        "polyline");
                            }
                            if (ortho_dim > 2) {
                                cloudViewer::utility::LogWarning(
                                        "[ccPointCloud::crop2D] Invalid input "
                                        "ortho_dim");
                            }
                            if (cloud.isEmpty()) {
                                cloudViewer::utility::LogWarning(
                                        "[ccPointCloud::crop2D] Cloud is "
                                        "empty!");
                            }

                            return std::ref(cloud);
                        }

                        ccPointCloud* croppedCloud = cloud.partialClone(ref);
                        {
                            delete ref;
                            ref = nullptr;
                        }
                        if (!croppedCloud) {
                            cloudViewer::utility::LogWarning(
                                    "[ccPointCloud::crop2D] Not enough "
                                    "memory!");
                            return std::ref(cloud);
                        }

                        return std::ref(*croppedCloud);
                    },
                    "Crops the cloud inside (or outside) a 2D polyline",
                    "polyline"_a, "ortho_dim"_a, "inside"_a = true)
            .def(
                    "compute_closest_points",
                    [](ccPointCloud& cloud, ccGenericPointCloud& other_cloud,
                       unsigned char octree_level) {
                        return std::ref(*cloud.computeCPSet(
                                other_cloud, nullptr, octree_level));
                    },
                    "Computes the closest point of this cloud relatively to "
                    "another cloud",
                    "other_cloud"_a, "octree_level"_a = 0)
            .def(
                    "append",
                    [](ccPointCloud& cloud, ccPointCloud& input_cloud,
                       unsigned start_count, bool ignore_children) {
                        return cloud.append(&input_cloud, start_count,
                                            ignore_children);
                    },
                    "Appends a cloud to this one", "input_cloud"_a,
                    "start_count"_a, "ignore_children"_a = false)
            .def(
                    "interpolate_colors_from",
                    [](ccPointCloud& pc, ccPointCloud& other_cloud,
                       unsigned char octree_level) {
                        return pc.interpolateColorsFrom(&other_cloud, nullptr,
                                                        octree_level);
                    },
                    "Interpolate colors from another cloud (nearest neighbor "
                    "only)",
                    "other_cloud"_a, "octree_level"_a = 0)
            .def("convert_normal_to_rgb", &ccPointCloud::convertNormalToRGB,
                 "Converts normals to RGB colors.")
            .def("convert_rgb_to_grey_scale",
                 &ccPointCloud::convertRGBToGreyScale,
                 "Converts RGB to grey scale colors.")
            .def("add_grey_color", &ccPointCloud::addGreyColor,
                 "Pushes a grey color on stack (shortcut).", "grey"_a)
            .def("colorize", &ccPointCloud::colorize,
                 "Multiplies all color components of all points by "
                 "coefficients.",
                 "r"_a, "g"_a, "b"_a)
            .def("set_color_by_banding", &ccPointCloud::setRGBColorByBanding,
                 "Assigns color to points by 'banding'.", "dim"_a,
                 "frequency"_a)
            .def("has_sensor", &ccPointCloud::hasSensor,
                 "Returns whether the mesh as an associated sensor or not.")
            .def("invert_normals", &ccPointCloud::invertNormals,
                 "Inverts normals (if any).")
            .def("convert_current_sf_to_colors",
                 &ccPointCloud::convertCurrentScalarFieldToColors,
                 "Converts current scalar field (values & display parameters) "
                 "to RGB colors.",
                 "mix_with_existing_color"_a = false)
            .def("set_color_with_current_sf",
                 &ccPointCloud::setRGBColorWithCurrentScalarField,
                 "Sets RGB colors with current scalar field (values & "
                 "parameters).",
                 "mix_with_existing_color"_a = false)
            .def(
                    "hide_points_by_sf",
                    [](ccPointCloud& cloud, ScalarType min_val,
                       ScalarType max_val) {
                        cloud.hidePointsByScalarValue(min_val, max_val);
                    },
                    "Hides points whose scalar values falls into an interval.",
                    "min_val"_a, "max_val"_a)
            .def(
                    "hide_points_by_sf",
                    [](ccPointCloud& cloud, std::vector<ScalarType> values) {
                        cloud.hidePointsByScalarValue(values);
                    },
                    "Hides points whose scalar values falls into an interval.",
                    "values"_a)
            .def(
                    "filter_points_by_sf",
                    [](ccPointCloud& cloud, ScalarType min_val,
                       ScalarType max_val, bool outside) {
                        return std::shared_ptr<ccPointCloud>(
                                cloud.filterPointsByScalarValue(
                                        min_val, max_val, outside));
                    },
                    "Filters out points whose scalar values falls into an "
                    "interval.",
                    "min_val"_a, "max_val"_a, "outside"_a = false)
            .def(
                    "filter_points_by_sf",
                    [](ccPointCloud& cloud, std::vector<ScalarType> values,
                       bool outside) {
                        return std::shared_ptr<ccPointCloud>(
                                cloud.filterPointsByScalarValue(values,
                                                                outside));
                    },
                    "Filters out points whose scalar values falls into an "
                    "interval.",
                    "values"_a, "outside"_a = false)
            .def(
                    "convert_normal_to_dipdir_sfs",
                    [](ccPointCloud& cloud) {
                        auto dipSF = new ccScalarField("Dip");
                        auto dipDirSF = new ccScalarField("DipDir");
                        if (!cloud.convertNormalToDipDirSFs(dipSF, dipDirSF)) {
                            cloudViewer::utility::LogWarning(
                                    "[ccPointCloud] Failed to convert normal "
                                    "to Dip and DipDir scalar fields!");
                        }
                        return std::make_tuple(
                                std::unique_ptr<ccScalarField, py::nodelete>(
                                        dipSF),
                                std::unique_ptr<ccScalarField, py::nodelete>(
                                        dipDirSF));
                    },
                    "Converts normals to two scalar fields: 'dip' and 'dip "
                    "direction'.")
            .def(
                    "clone_this",
                    [](ccPointCloud& cloud, bool ignore_children) {
                        return std::shared_ptr<ccPointCloud>(
                                cloud.cloneThis(nullptr, ignore_children));
                    },
                    "All the main features of the entity are cloned, except "
                    "from the octree and"
                    " the points visibility information.",
                    "ignore_children"_a = true)
            .def(
                    "partial_clone",
                    [](const ccPointCloud& cloud,
                       std::shared_ptr<cloudViewer::ReferenceCloud> selection) {
                        return std::shared_ptr<ccPointCloud>(
                                cloud.partialClone(selection.get(), nullptr));
                    },
                    "Creates a new point cloud object from a ReferenceCloud "
                    "(selection).",
                    "selection"_a)
            .def(
                    "enhance_rgb_with_intensity_sf",
                    [](ccPointCloud& cloud, int sf_index,
                       bool use_intensity_range, double min_intensity,
                       double max_intensity) {
                        return cloud.enhanceRGBWithIntensitySF(
                                sf_index, use_intensity_range, min_intensity,
                                max_intensity);
                    },
                    "Enhances the RGB colors with the current scalar field "
                    "(assuming it's intensities).",
                    "sf_index"_a, "use_intensity_range"_a = false,
                    "min_intensity"_a = 0.0, "max_intensity"_a = 1.0)
            .def(
                    "export_coord_to_sf",
                    [](ccPointCloud& cloud, bool export_x, bool export_y,
                       bool export_z) {
                        bool exportDims[3];
                        exportDims[0] = export_x;
                        exportDims[1] = export_y;
                        exportDims[2] = export_z;
                        return cloud.exportCoordToSF(exportDims);
                    },
                    "Exports the specified coordinate dimension(s) to scalar "
                    "field(s).",
                    "export_x"_a = false, "export_y"_a = false,
                    "export_z"_a = false)
            .def(
                    "export_normal_to_sf",
                    [](ccPointCloud& cloud, bool export_x, bool export_y,
                       bool export_z) {
                        bool exportDims[3];
                        exportDims[0] = export_x;
                        exportDims[1] = export_y;
                        exportDims[2] = export_z;
                        return cloud.exportCoordToSF(exportDims);
                    },
                    "Exports the specified normal dimension(s) to scalar "
                    "field(s).",
                    "export_x"_a = false, "export_y"_a = false,
                    "export_z"_a = false)
            .def("estimate_covariances", &ccPointCloud::estimateCovariances,
                 "Function to compute the covariance matrix for each point "
                 "in the point cloud",
                 "search_param"_a = KDTreeSearchParamKNN())
            .def_readwrite("covariances", &ccPointCloud::covariances_,
                   "``float64`` array of shape ``(num_points, 3, 3)``, "
                   "use ``numpy.asarray()`` to access data: Points "
                   "covariances.")
            .def_static(
                    "estimate_point_covariances",
                    &ccPointCloud::EstimatePerPointCovariances,
                    "Static function to compute the covariance matrix for "
                    "each "
                    "point in the given point cloud, doesn't change the input",
                    "input"_a, "search_param"_a = KDTreeSearchParamKNN())
            .def_static(
                    "from",
                    [](const cloudViewer::GenericIndexedCloud& cloud,
                       std::shared_ptr<const ccGenericPointCloud>
                               source_cloud) {
                        return std::shared_ptr<ccPointCloud>(
                                ccPointCloud::From(&cloud, source_cloud.get()));
                    },
                    "'GenericCloud' is a very simple and light interface from "
                    "cloudViewer. It is"
                    "meant to give access to points coordinates of any "
                    "cloud(on the "
                    "condition it implements the GenericCloud interface of "
                    "course). "
                    "See cloudViewer documentation for more information about "
                    "GenericClouds."
                    "As the GenericCloud interface is very simple, only points "
                    "are imported."
                    "Note : throws an 'int' exception in case of error(see "
                    "CTOR_ERRORS)"
                    "-param cloud a GenericCloud structure"
                    "-param source_cloud cloud from which main parameters will "
                    "be imported(optional)",
                    "cloud"_a, "source_cloud"_a = nullptr)
            .def_static(
                    "from",
                    [](const ccPointCloud& source_cloud,
                       const std::vector<size_t>& indices) {
                        return std::shared_ptr<ccPointCloud>(
                                ccPointCloud::From(&source_cloud, indices));
                    },
                    "Function to select points from input ccPointCloud into "
                    "output ccPointCloud."
                    "Points with indices in param indices are selected",
                    "source_cloud"_a, "indices"_a)
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
            m, "ccPointCloud", "estimate_point_covariances",
            {{"input", "The input point cloud."},
             {"search_param",
              "The KDTree search parameters for neighborhood search."}});
    docstring::ClassMethodDocInject(
            m, "ccPointCloud", "estimate_covariances",
            {{"search_param",
              "The KDTree search parameters for neighborhood search."}});
    docstring::ClassMethodDocInject(
            m, "ccPointCloud", "paint_uniform_color",
            {{"color", "RGB color for the PointCloud."}});
    docstring::ClassMethodDocInject(
            m, "ccPointCloud", "select_by_index",
            {{"indices", "Indices of points to be selected."},
             {"invert",
              "Set to ``True`` to invert the selection of indices."}});
    docstring::ClassMethodDocInject(
            m, "ccPointCloud", "voxel_down_sample",
            {{"voxel_size", "Voxel size to downsample into."},
             {"invert", "set to ``True`` to invert the selection of indices"}});
    docstring::ClassMethodDocInject(
            m, "ccPointCloud", "voxel_down_sample_and_trace",
            {{"voxel_size", "Voxel size to downsample into."},
             {"min_bound", "Minimum coordinate of voxel boundaries"},
             {"max_bound", "Maximum coordinate of voxel boundaries"}});
    docstring::ClassMethodDocInject(
            m, "ccPointCloud", "uniform_down_sample",
            {{"every_k_points",
              "Sample rate, the selected point indices are [0, k, 2k, ...]"}});
    docstring::ClassMethodDocInject(
            m, "ccPointCloud", "random_down_sample",
            {{"sampling_ratio",
              "Sampling ratio, the ratio of number of selected points to total "
              "number of points[0-1]"}});
    docstring::ClassMethodDocInject(
            m, "ccPointCloud", "crop",
            {{"bounding_box", "ccBBox to crop points"}});
    docstring::ClassMethodDocInject(
            m, "ccPointCloud", "remove_non_finite_points",
            {{"remove_nan", "Remove NaN values from the PointCloud"},
             {"remove_infinite",
              "Remove infinite values from the PointCloud"}});
    docstring::ClassMethodDocInject(
            m, "ccPointCloud", "remove_radius_outlier",
            {{"nb_points", "Number of points within the radius."},
             {"radius", "Radius of the sphere."}});
    docstring::ClassMethodDocInject(
            m, "ccPointCloud", "remove_statistical_outlier",
            {{"nb_neighbors", "Number of neighbors around the target point."},
             {"std_ratio", "Standard deviation ratio."}});
    docstring::ClassMethodDocInject(
            m, "ccPointCloud", "estimate_normals",
            {{"search_param",
              "The KDTree search parameters for neighborhood search."},
             {"fast_normal_computation",
              "If true, the normal estiamtion uses a non-iterative method to "
              "extract the eigenvector from the covariance matrix. This is "
              "faster, but is not as numerical stable."}});
    docstring::ClassMethodDocInject(
            m, "ccPointCloud", "orient_normals_to_align_with_direction",
            {{"orientation_reference",
              "Normals are oriented with respect to orientation_reference."}});
    docstring::ClassMethodDocInject(
            m, "ccPointCloud", "orient_normals_towards_camera_location",
            {{"camera_location",
              "Normals are oriented with towards the camera_location."}});
    docstring::ClassMethodDocInject(
            m, "ccPointCloud", "orient_normals_consistent_tangent_plane",
            {{"k",
              "Number of k nearest neighbors used in constructing the "
              "Riemannian graph used to propogate normal orientation."}});
    docstring::ClassMethodDocInject(m, "ccPointCloud",
                                    "compute_point_cloud_distance",
                                    {{"target", "The target point cloud."}});
    docstring::ClassMethodDocInject(m, "ccPointCloud",
                                    "compute_mahalanobis_distance");
    docstring::ClassMethodDocInject(m, "ccPointCloud",
                                    "compute_nearest_neighbor_distance");
    docstring::ClassMethodDocInject(m, "ccPointCloud", "compute_resolution");
    docstring::ClassMethodDocInject(m, "ccPointCloud", "compute_convex_hull",
                                    {{"input", "The input point cloud."}});
    docstring::ClassMethodDocInject(
            m, "ccPointCloud", "hidden_point_removal",
            {{"input", "The input point cloud."},
             {"camera_location",
              "All points not visible from that location will be removed"},
             {"radius", "The radius of the sperical projection"}});
    docstring::ClassMethodDocInject(
            m, "ccPointCloud", "cluster_dbscan",
            {{"eps",
              "Density parameter that is used to find neighboring points."},
             {"min_points", "Minimum number of points to form a cluster."},
             {"print_progress",
              "If true the progress is visualized in the console."}});
    docstring::ClassMethodDocInject(
            m, "ccPointCloud", "segment_plane",
            {{"distance_threshold",
              "Max distance a point can be from the plane model, and still be "
              "considered an inlier."},
             {"ransac_n",
              "Number of initial points to be considered inliers in each "
              "iteration."},
             {"num_iterations", "Number of iterations."}});
    docstring::ClassMethodDocInject(
            m, "ccPointCloud", "create_from_depth_image",
            {{"depth",
              "The input depth image can be either a float image, or a "
              "uint16_t image."},
             {"intrinsic", "Intrinsic parameters of the camera."},
             {"extrnsic", "Extrinsic parameters of the camera."},
             {"depth_scale", "The depth is scaled by 1 / depth_scale."},
             {"depth_trunc", "Truncated at depth_trunc distance."},
             {"stride",
              "Sampling factor to support coarse point cloud extraction."}});
    docstring::ClassMethodDocInject(
            m, "ccPointCloud", "create_from_rgbd_image",
            {{"image", "The input image."},
             {"intrinsic", "Intrinsic parameters of the camera."},
             {"extrnsic", "Extrinsic parameters of the camera."}});
    docstring::ClassMethodDocInject(m, "ccPointCloud", "unalloacte_points");
    docstring::ClassMethodDocInject(m, "ccPointCloud", "unalloacte_colors");
    docstring::ClassMethodDocInject(m, "ccPointCloud", "unalloacte_norms");
    docstring::ClassMethodDocInject(m, "ccPointCloud", "colors_have_changed");
    docstring::ClassMethodDocInject(m, "ccPointCloud", "normals_have_changed");
    docstring::ClassMethodDocInject(m, "ccPointCloud", "points_have_changed");
    docstring::ClassMethodDocInject(m, "ccPointCloud", "reserve");
    docstring::ClassMethodDocInject(m, "ccPointCloud", "reserve_points");
    docstring::ClassMethodDocInject(m, "ccPointCloud", "reserve_colors");
    docstring::ClassMethodDocInject(m, "ccPointCloud", "reserve_norms");
    docstring::ClassMethodDocInject(m, "ccPointCloud", "resize");
    docstring::ClassMethodDocInject(m, "ccPointCloud", "resize_colors");
    docstring::ClassMethodDocInject(m, "ccPointCloud", "resize_norms");
    docstring::ClassMethodDocInject(m, "ccPointCloud", "shrink");
    docstring::ClassMethodDocInject(m, "ccPointCloud", "reset");
    docstring::ClassMethodDocInject(m, "ccPointCloud", "capacity");
    docstring::ClassMethodDocInject(m, "ccPointCloud", "invalidate_bbox");
    docstring::ClassMethodDocInject(m, "ccPointCloud",
                                    "set_current_input_sf_index");
    docstring::ClassMethodDocInject(m, "ccPointCloud",
                                    "get_current_input_sf_index");
    docstring::ClassMethodDocInject(m, "ccPointCloud",
                                    "set_current_output_sf_index");
    docstring::ClassMethodDocInject(m, "ccPointCloud",
                                    "get_current_output_sf_index");
    docstring::ClassMethodDocInject(m, "ccPointCloud", "sfs_count");
    docstring::ClassMethodDocInject(m, "ccPointCloud", "get_sf_by_index");
    docstring::ClassMethodDocInject(m, "ccPointCloud", "get_sf_name");
    docstring::ClassMethodDocInject(m, "ccPointCloud", "get_sf_index_by_name");
    docstring::ClassMethodDocInject(m, "ccPointCloud", "get_current_input_sf");
    docstring::ClassMethodDocInject(m, "ccPointCloud", "get_current_output_sf");
    docstring::ClassMethodDocInject(m, "ccPointCloud", "rename_sf");
    docstring::ClassMethodDocInject(m, "ccPointCloud", "set_current_sf");
    docstring::ClassMethodDocInject(m, "ccPointCloud",
                                    "set_current_displayed_sf");
    docstring::ClassMethodDocInject(m, "ccPointCloud",
                                    "get_current_displayed_sf");
    docstring::ClassMethodDocInject(m, "ccPointCloud", "get_current_sf_index");
    docstring::ClassMethodDocInject(m, "ccPointCloud", "delete_sf");
    docstring::ClassMethodDocInject(m, "ccPointCloud", "delete_all_sfs");
    docstring::ClassMethodDocInject(m, "ccPointCloud", "add_sf");
    docstring::ClassMethodDocInject(m, "ccPointCloud", "sf_color_scale_shown");
    docstring::ClassMethodDocInject(m, "ccPointCloud", "show_sf_color_scale");
    docstring::ClassMethodDocInject(m, "ccPointCloud",
                                    "compute_gravity_center");
    docstring::ClassMethodDocInject(m, "ccPointCloud", "crop_2d");
    docstring::ClassMethodDocInject(m, "ccPointCloud",
                                    "compute_closest_points");
    docstring::ClassMethodDocInject(m, "ccPointCloud", "append");
    docstring::ClassMethodDocInject(m, "ccPointCloud",
                                    "interpolate_colors_from");
    docstring::ClassMethodDocInject(m, "ccPointCloud", "convert_normal_to_rgb");
    docstring::ClassMethodDocInject(m, "ccPointCloud",
                                    "convert_rgb_to_grey_scale");
    docstring::ClassMethodDocInject(m, "ccPointCloud", "add_grey_color");
    docstring::ClassMethodDocInject(m, "ccPointCloud", "colorize");
    docstring::ClassMethodDocInject(m, "ccPointCloud", "set_color_by_banding");
    docstring::ClassMethodDocInject(m, "ccPointCloud", "has_sensor");
    docstring::ClassMethodDocInject(m, "ccPointCloud", "invert_normals");
    docstring::ClassMethodDocInject(m, "ccPointCloud",
                                    "convert_current_sf_to_colors");
    docstring::ClassMethodDocInject(m, "ccPointCloud",
                                    "set_color_with_current_sf");
    docstring::ClassMethodDocInject(m, "ccPointCloud", "hide_points_by_sf");
    docstring::ClassMethodDocInject(m, "ccPointCloud", "filter_points_by_sf");
    docstring::ClassMethodDocInject(m, "ccPointCloud",
                                    "convert_normal_to_dipdir_sfs");
    docstring::ClassMethodDocInject(m, "ccPointCloud", "clone_this");
    docstring::ClassMethodDocInject(m, "ccPointCloud", "partial_clone");
    docstring::ClassMethodDocInject(m, "ccPointCloud",
                                    "enhance_rgb_with_intensity_sf");
    docstring::ClassMethodDocInject(m, "ccPointCloud", "export_coord_to_sf");
    docstring::ClassMethodDocInject(m, "ccPointCloud", "export_normal_to_sf");
}

void pybind_pointcloud_methods(py::module& m) {}

}  // namespace geometry
}  // namespace cloudViewer
