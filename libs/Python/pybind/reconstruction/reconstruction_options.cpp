// ----------------------------------------------------------------------------
// -                        cloudViewer: asher-1.github.io                    -
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

#include "pybind/reconstruction/reconstruction_options.h"

#include <Logging.h>

#include "pipelines/option_utils.h"
#include "pybind/docstring.h"

namespace cloudViewer {
namespace reconstruction {
namespace options {

void pybind_image_reader_options(py::module& m) {
    // cloudViewer.reconstruction.options.ImageReaderOptions
    py::class_<colmap::ImageReaderOptions,
               std::shared_ptr<colmap::ImageReaderOptions>>
            image_reader_options(m, "ImageReaderOptions",
                                 "Image Reader option class.");
    image_reader_options.def(py::init<>())
            .def("check", &colmap::ImageReaderOptions::Check,
                 "Check parameters validation.")
            .def_readwrite(
                    "database_path", &colmap::ImageReaderOptions::database_path,
                    "str: (Default ``''``) Path to database in which to store "
                    "the extracted data.")
            .def_readwrite(
                    "image_path", &colmap::ImageReaderOptions::image_path,
                    "str: (Default ``''``) Root path to folder which contains "
                    "the images.")
            .def_readwrite(
                    "mask_path", &colmap::ImageReaderOptions::mask_path,
                    "str: (Default ``''``) Optional root path to folder which "
                    "contains image masks. For a given image, the "
                    "corresponding mask must have the same sub-path below "
                    "this root as the image has below image_path. The filename "
                    "must be equal, "
                    "aside from the added extension .png. For example, for an "
                    "image "
                    "image_path/abc/012.jpg, the mask would be "
                    "mask_path/abc/012.jpg.png. No features "
                    "will be extracted in regions where the mask image is "
                    "black (pixel intensity "
                    "value 0 in grayscale).")
            .def_readwrite("image_list",
                           &colmap::ImageReaderOptions::image_list,
                           "List(str, ...): Optional list of images to read. "
                           "The list must contain the relative path "
                           "of the images with respect to the image_path.")
            .def_readwrite(
                    "camera_model", &colmap::ImageReaderOptions::camera_model,
                    "str: (Default ``SIMPLE_RADIAL``) Name of the camera "
                    "model. Supported camera model:"
                    "(``SIMPLE_PINHOLE``, ``PINHOLE``, ``SIMPLE_RADIAL``, "
                    "``RADIAL``, ``OPENCV``, ``OPENCV_FISHEYE``, "
                    "``FULL_OPENCV``, ``FOV``, ``SIMPLE_RADIAL_FISHEYE``, "
                    "``RADIAL_FISHEYE``, ``THIN_PRISM_FISHEYE``, "
                    "``SIMPLE_RADIAL_FISHEYE``)")
            .def_readwrite(
                    "single_camera", &colmap::ImageReaderOptions::single_camera,
                    "bool: (Default ``False``) Set to ``True`` to enable the "
                    "same camera for all images.")
            .def_readwrite(
                    "single_camera_per_folder",
                    &colmap::ImageReaderOptions::single_camera_per_folder,
                    "bool: (Default ``False``) Set to ``True`` to enable the "
                    "same camera for all images in the same sub-folder.")
            .def_readwrite(
                    "single_camera_per_image",
                    &colmap::ImageReaderOptions::single_camera_per_image,
                    "bool: (Default ``False``) Set to ``True`` to enable a "
                    "different camera for each image.")
            .def_readwrite(
                    "existing_camera_id",
                    &colmap::ImageReaderOptions::existing_camera_id,
                    "int: (Default "
                    "``kInvalidCameraId=std::numeric_limits<camera_t>::max()``)"
                    " Whether to "
                    "explicitly use an existing camera for all images. "
                    "Note that in this case the specified camera model and "
                    "parameters are ignored.")
            .def_readwrite(
                    "camera_params", &colmap::ImageReaderOptions::camera_params,
                    "str: (Default ``''``) Manual specification of "
                    "camera parameters. If empty, camera parameters will "
                    "be extracted from EXIF, i.e. principal point and focal "
                    "length.")
            .def_readwrite(
                    "default_focal_length_factor",
                    &colmap::ImageReaderOptions::default_focal_length_factor,
                    "float: (Default ``1.2``) If camera parameters are not "
                    "specified manually and the image does not have focal "
                    "length EXIF information, the focal length is set to "
                    "the value ``default_focal_length_factor * max(width, "
                    "height)``.")
            .def_readwrite("camera_mask_path",
                           &colmap::ImageReaderOptions::camera_mask_path,
                           "str: (Default ``''``) Optional path to an image "
                           "file specifying a mask for all images. No features "
                           "will be extracted in regions where the mask is "
                           "black (pixel intensity value 0 in grayscale).");
}

void pybind_sift_extraction_options(py::module& m) {
    // cloudViewer.reconstruction.options.SiftExtractionOptions
    py::class_<colmap::SiftExtractionOptions,
               std::shared_ptr<colmap::SiftExtractionOptions>>
            sift_extraction_options(m, "SiftExtractionOptions",
                                    "Sift Extraction option class.");
    sift_extraction_options.def(py::init<>())
            .def("check", &colmap::SiftExtractionOptions::Check,
                 "Check parameters validation.")
            .def_readwrite("num_threads",
                           &colmap::SiftExtractionOptions::num_threads,
                           "int (Default ``-1``) Number of threads for feature "
                           "extraction.")
            .def_readwrite("use_gpu", &colmap::SiftExtractionOptions::use_gpu,
                           "bool: (Default ``True``) Whether to use the GPU "
                           "for feature "
                           "extraction.")
            .def_readwrite(
                    "gpu_index", &colmap::SiftExtractionOptions::gpu_index,
                    "str: (Default ``-1``) Index of the GPU used for feature "
                    "extraction. For multi-GPU extraction, you should separate "
                    "multiple GPU indices by comma, e.g. ``0,1,2,3``.")
            .def_readwrite(
                    "max_image_size",
                    &colmap::SiftExtractionOptions::max_image_size,
                    "int: (Default ``3200``) Maximum image size, otherwise "
                    "image will be down-scaled.")
            .def_readwrite(
                    "max_num_features",
                    &colmap::SiftExtractionOptions::max_num_features,
                    "int: (Default ``8192``) Maximum number of features to "
                    "detect, keeping larger-scale features.")
            .def_readwrite(
                    "first_octave",
                    &colmap::SiftExtractionOptions::first_octave,
                    "int: (Default ``-1``) First octave in the pyramid, i.e. "
                    "-1 upsamples the image by one level.")
            .def_readwrite("octave_resolution",
                           &colmap::SiftExtractionOptions::octave_resolution,
                           "int: (Default ``3``) Number of levels per octave.")
            .def_readwrite("peak_threshold",
                           &colmap::SiftExtractionOptions::peak_threshold,
                           "float: (Default ``0.02 / octave_resolution``) Peak "
                           "threshold for detection.")
            .def_readwrite(
                    "edge_threshold",
                    &colmap::SiftExtractionOptions::edge_threshold,
                    "float: (Default ``10.0``) Edge threshold for detection.")
            .def_readwrite(
                    "estimate_affine_shape",
                    &colmap::SiftExtractionOptions::estimate_affine_shape,
                    "bool: (Default ``False``) Estimate affine shape of SIFT "
                    "features "
                    "in the form of oriented ellipses as opposed to original "
                    "SIFT "
                    "which estimates oriented disks.")
            .def_readwrite(
                    "max_num_orientations",
                    &colmap::SiftExtractionOptions::max_num_orientations,
                    "int: (Default ``2``) Maximum number of orientations per "
                    "keypoint if not estimate_affine_shape.")
            .def_readwrite(
                    "upright", &colmap::SiftExtractionOptions::upright,
                    "bool: (Default ``False``) Fix the orientation to 0 for "
                    "upright features.")
            .def_readwrite(
                    "darkness_adaptivity",
                    &colmap::SiftExtractionOptions::darkness_adaptivity,
                    "bool: (Default ``False``) Whether to adapt the feature "
                    "detection depending on the image darkness. "
                    "Note that this feature is only available in the "
                    "OpenGL SiftGPU version.")
            .def_readwrite(
                    "domain_size_pooling",
                    &colmap::SiftExtractionOptions::domain_size_pooling,
                    "bool: (Default ``False``) Domain-size pooling parameters."
                    " Domain-size pooling computes an average SIFT descriptor "
                    "across multiple scales around the "
                    "detected scale. This was proposed in Domain-Size Pooling "
                    "in Local Descriptors "
                    "and Network Architectures, J. Dong and S. Soatto, CVPR "
                    "2015. This "
                    "has been shown to outperform other SIFT variants and "
                    "learned descriptors "
                    "in Comparative Evaluation of Hand-Crafted and Learned "
                    "Local Features, "
                    "Schönberger, Hardmeier, Sattler, Pollefeys, CVPR 2016.")
            .def_readwrite("dsp_min_scale",
                           &colmap::SiftExtractionOptions::dsp_min_scale,
                           "float: (Default ``1.0 / 6.0``) min scale.")
            .def_readwrite("dsp_max_scale",
                           &colmap::SiftExtractionOptions::dsp_max_scale,
                           "float: (Default ``3.0``) max scale.")
            .def_readwrite("dsp_num_scales",
                           &colmap::SiftExtractionOptions::dsp_num_scales,
                           "int: (Default ``10``) The number of scales.")
            .def_readwrite(
                    "normalization",
                    &colmap::SiftExtractionOptions::normalization,
                    "Normalization: L1-normalizes each descriptor followed by "
                    "element-wise square rooting. This normalization is "
                    "usually "
                    "better than standard L2-normalization. "
                    "See ``Three things everyone should know to improve object "
                    "retrieval``, "
                    "Relja Arandjelovic and Andrew Zisserman, CVPR 2012.");

    // cloudViewer.reconstruction.options.NormalizationType
    py::enum_<colmap::SiftExtractionOptions::Normalization> normalization_type(
            sift_extraction_options, "NormalizationType", py::arithmetic());
    normalization_type
            .value("L1_ROOT",
                   colmap::SiftExtractionOptions::Normalization::L1_ROOT)
            .value("L2", colmap::SiftExtractionOptions::Normalization::L2)
            .export_values();
    normalization_type.attr("__doc__") = docstring::static_property(
            py::cpp_function([](py::handle arg) -> std::string {
                return "Enum class for Reconstruction Normalization types.";
            }),
            py::none(), py::none(), "");
}

void pybind_sift_matching_options(py::module& m) {
    // cloudViewer.reconstruction.options.SiftMatchingOptions
    py::class_<colmap::SiftMatchingOptions> sift_matching_options(
            m, "SiftMatchingOptions", "Sift Matching option class.");
    sift_matching_options.def(py::init<>())
            .def("check", &colmap::SiftMatchingOptions::Check,
                 "Check parameters validation.")
            .def_readwrite("num_threads",
                           &colmap::SiftMatchingOptions::num_threads,
                           "(Default ``-1``) Number of threads for feature "
                           "extraction.")
            .def_readwrite(
                    "use_gpu", &colmap::SiftMatchingOptions::use_gpu,
                    "(Default ``True``) Whether to use the GPU for feature "
                    "extraction.")
            .def_readwrite(
                    "gpu_index", &colmap::SiftMatchingOptions::gpu_index,
                    "(Default ``-1``) Index of the GPU used for feature "
                    "extraction. For multi-GPU extraction, you should separate "
                    "multiple GPU indices by comma, e.g. ``0,1,2,3``.")
            .def_readwrite("max_ratio", &colmap::SiftMatchingOptions::max_ratio,
                           "float: (Default ``0.8``) Maximum distance ratio "
                           "between first and second best match.")
            .def_readwrite(
                    "max_distance", &colmap::SiftMatchingOptions::max_distance,
                    "float: (Default ``0.7``) Maximum distance to best match.")
            .def_readwrite("cross_check",
                           &colmap::SiftMatchingOptions::cross_check,
                           "bool: (Default ``True``) Whether to enable cross "
                           "checking in matching.")
            .def_readwrite(
                    "max_num_matches",
                    &colmap::SiftMatchingOptions::max_num_matches,
                    "int: (Default ``32768``) Maximum number of matches.")
            .def_readwrite("max_error", &colmap::SiftMatchingOptions::max_error,
                           "float: (Default ``4.0``) Maximum epipolar error in "
                           "pixels for geometric verification.")
            .def_readwrite("confidence",
                           &colmap::SiftMatchingOptions::confidence,
                           "float: (Default ``0.999``) Confidence threshold "
                           "for geometric verification.")

            .def_readwrite("min_num_trials",
                           &colmap::SiftMatchingOptions::min_num_trials,
                           "int: (Default ``100``) Minimum number of RANSAC "
                           "iterations. Note that this option "
                           "overrules the min_inlier_ratio option.")
            .def_readwrite("max_num_trials",
                           &colmap::SiftMatchingOptions::max_num_trials,
                           "int: (Default ``10000``) Maximum number of RANSAC "
                           "iterations.")
            .def_readwrite(
                    "min_inlier_ratio",
                    &colmap::SiftMatchingOptions::min_inlier_ratio,
                    "float: (Default ``0.25``) A priori assumed minimum inlier "
                    "ratio, which determines the maximum number of iterations.")
            .def_readwrite(
                    "min_num_inliers",
                    &colmap::SiftMatchingOptions::min_num_inliers,
                    "int: (Default ``15``) Minimum number of inliers for an "
                    "image pair to be considered as geometrically verified.")
            .def_readwrite("multiple_models",
                           &colmap::SiftMatchingOptions::multiple_models,
                           "bool: (Default ``False``) Whether to attempt to "
                           "estimate multiple geometric models per image pair.")
            .def_readwrite(
                    "guided_matching",
                    &colmap::SiftMatchingOptions::guided_matching,
                    "bool: (Default ``False``) Whether to perform guided "
                    "matching, if geometric verification succeeds.");
}

void pybind_exhaustive_matching_options(py::module& m) {
    // cloudViewer.reconstruction.options.ExhaustiveMatchingOptions
    py::class_<colmap::ExhaustiveMatchingOptions> exhaustive_matching_options(
            m, "ExhaustiveMatchingOptions",
            "Exhaustive Matching option class.");
    exhaustive_matching_options.def(py::init<>())
            .def("check", &colmap::ExhaustiveMatchingOptions::Check,
                 "Check parameters validation.")
            .def_readwrite("block_size",
                           &colmap::ExhaustiveMatchingOptions::block_size,
                           "int: (Default ``50``) Block size, i.e. number of "
                           "images to simultaneously load into memory.");
}

void pybind_sequential_matching_options(py::module& m) {
    // cloudViewer.reconstruction.options.SequentialMatchingOptions
    py::class_<colmap::SequentialMatchingOptions> sequential_matching_options(
            m, "SequentialMatchingOptions",
            "Sequential Matching option class.");
    sequential_matching_options.def(py::init<>())
            .def("check", &colmap::SequentialMatchingOptions::Check,
                 "Check parameters validation.")
            .def_readwrite(
                    "overlap", &colmap::SequentialMatchingOptions::overlap,
                    "int: (Default ``10``) Number of overlapping image pairs.")
            .def_readwrite(
                    "quadratic_overlap",
                    &colmap::SequentialMatchingOptions::quadratic_overlap,
                    "bool: (Default ``True``) Whether to match images against "
                    "their quadratic neighbors.")
            .def_readwrite("loop_detection",
                           &colmap::SequentialMatchingOptions::loop_detection,
                           "bool: (Default ``False``) Whether to enable "
                           "vocabulary tree based loop detection.")
            .def_readwrite(
                    "loop_detection_period",
                    &colmap::SequentialMatchingOptions::loop_detection_period,
                    "int: (Default ``10``) Loop detection is invoked every "
                    "`loop_detection_period` images.")
            .def_readwrite("loop_detection_num_images",
                           &colmap::SequentialMatchingOptions::
                                   loop_detection_num_images,
                           "int: (Default ``50``) The number of images to "
                           "retrieve in loop detection. "
                           "This number should be significantly bigger than "
                           "the sequential matching overlap.")
            .def_readwrite("loop_detection_num_nearest_neighbors",
                           &colmap::SequentialMatchingOptions::
                                   loop_detection_num_nearest_neighbors,
                           "int: (Default ``1``) Number of nearest neighbors "
                           "to retrieve per query feature.")
            .def_readwrite("loop_detection_num_checks",
                           &colmap::SequentialMatchingOptions::
                                   loop_detection_num_checks,
                           "int: (Default ``256``) Number of nearest-neighbor "
                           "checks to use in retrieval.")
            .def_readwrite("loop_detection_num_images_after_verification",
                           &colmap::SequentialMatchingOptions::
                                   loop_detection_num_images_after_verification,
                           "int: (Default ``0``) How many images to return "
                           "after spatial verification. "
                           "Set to 0 to turn off spatial verification.")
            .def_readwrite("loop_detection_max_num_features",
                           &colmap::SequentialMatchingOptions::
                                   loop_detection_max_num_features,
                           "int: (Default ``-1``) The maximum number of "
                           "features to use for indexing an image. "
                           "If an image has more features, only the "
                           "largest-scale features will be indexed.")
            .def_readwrite("vocab_tree_path",
                           &colmap::SequentialMatchingOptions::vocab_tree_path,
                           "str: (Default ``"
                           "``) Path to the vocabulary tree.");
}

void pybind_vocabtree_matching_options(py::module& m) {
    // cloudViewer.reconstruction.options.VocabTreeMatchingOptions
    py::class_<colmap::VocabTreeMatchingOptions> vocabtree_matching_options(
            m, "VocabTreeMatchingOptions", "VocabTree Matching option class.");
    vocabtree_matching_options.def(py::init<>())
            .def("check", &colmap::VocabTreeMatchingOptions::Check,
                 "Check parameters validation.")
            .def_readwrite("num_images",
                           &colmap::VocabTreeMatchingOptions::num_images,
                           "int: (Default ``100``) Number of images to "
                           "retrieve for each query image.")
            .def_readwrite(
                    "num_nearest_neighbors",
                    &colmap::VocabTreeMatchingOptions::num_nearest_neighbors,
                    "int: (Default ``5``) Number of nearest neighbors to "
                    "retrieve per query feature.")
            .def_readwrite("num_checks",
                           &colmap::VocabTreeMatchingOptions::num_checks,
                           "int: (Default ``256``) Number of nearest-neighbor "
                           "checks to use in retrieval.")
            .def_readwrite("num_images_after_verification",
                           &colmap::VocabTreeMatchingOptions::
                                   num_images_after_verification,
                           "int: (Default ``0``) How many images to return "
                           "after spatial verification. Set to 0 to turn off "
                           "spatial verification.")
            .def_readwrite(
                    "max_num_features",
                    &colmap::VocabTreeMatchingOptions::max_num_features,
                    "int: (Default ``-1``) The maximum number of features to "
                    "use for indexing an image. If an image has more features, "
                    "only the largest-scale features will be indexed.")
            .def_readwrite("vocab_tree_path",
                           &colmap::VocabTreeMatchingOptions::vocab_tree_path,
                           "str: (Default ``''``) Path to the vocabulary tree.")
            .def_readwrite("match_list_path",
                           &colmap::VocabTreeMatchingOptions::match_list_path,
                           "str: (Default ``''``) Optional path to file with "
                           "specific image "
                           "names to match.");
}

void pybind_spatial_matching_options(py::module& m) {
    // cloudViewer.reconstruction.options.SpatialMatchingOptions
    py::class_<colmap::SpatialMatchingOptions> spatial_matching_options(
            m, "SpatialMatchingOptions", "Spatial Matching option class.");
    spatial_matching_options.def(py::init<>())
            .def("check", &colmap::SpatialMatchingOptions::Check,
                 "Check parameters validation.")
            .def_readwrite(
                    "is_gps", &colmap::SpatialMatchingOptions::is_gps,
                    "bool: (Default ``True``) Whether the location priors in "
                    "the database are GPS coordinates in the form of longitude "
                    "and latitude coordinates in degrees.")
            .def_readwrite("ignore_z",
                           &colmap::SpatialMatchingOptions::ignore_z,
                           "bool: (Default ``True``) Whether to ignore the "
                           "Z-component of the location prior.")
            .def_readwrite("max_num_neighbors",
                           &colmap::SpatialMatchingOptions::max_num_neighbors,
                           "int: (Default ``50``) The maximum number of "
                           "nearest neighbors to match.")
            .def_readwrite(
                    "max_distance",
                    &colmap::SpatialMatchingOptions::max_distance,
                    "int: (Default ``100``) The maximum distance between the "
                    "query and nearest neighbor. For GPS coordinates the unit "
                    "is Euclidean distance in meters.");
}

void pybind_transitive_matching_options(py::module& m) {
    // cloudViewer.reconstruction.options.TransitiveMatchingOptions
    py::class_<colmap::TransitiveMatchingOptions> transitive_matching_options(
            m, "TransitiveMatchingOptions",
            "Transitive Matching option class.");
    transitive_matching_options.def(py::init<>())
            .def("check", &colmap::TransitiveMatchingOptions::Check,
                 "Check parameters validation.")
            .def_readwrite("batch_size",
                           &colmap::TransitiveMatchingOptions::batch_size,
                           "int: (Default ``1000``) The maximum number of "
                           "image pairs to process in one batch.")
            .def_readwrite("num_iterations",
                           &colmap::TransitiveMatchingOptions::num_iterations,
                           "int: (Default ``3``) The number of transitive "
                           "closure iterations.");
}

void pybind_imagepairs_matching_options(py::module& m) {
    // cloudViewer.reconstruction.options.ImagePairsMatchingOptions
    py::class_<colmap::ImagePairsMatchingOptions> imagepairs_matching_options(
            m, "ImagePairsMatchingOptions",
            "ImagePairs Matching option class.");
    imagepairs_matching_options.def(py::init<>())
            .def("check", &colmap::ImagePairsMatchingOptions::Check,
                 "Check parameters validation.")
            .def_readwrite("block_size",
                           &colmap::ImagePairsMatchingOptions::block_size,
                           "int: (Default ``1225``) Number of image pairs to "
                           "match in one batch.")
            .def_readwrite("match_list_path",
                           &colmap::ImagePairsMatchingOptions::match_list_path,
                           "str: (Default ``''``) Optional path to file with "
                           "specific image names to match.");
}

void pybind_featurepairs_matching_options(py::module& m) {
    // cloudViewer.reconstruction.options.FeaturePairsMatchingOptions
    py::class_<colmap::FeaturePairsMatchingOptions>
            featurepairs_matching_options(
                    m, "FeaturePairsMatchingOptions",
                    "FeaturePairs Matching option class.");
    featurepairs_matching_options.def(py::init<>())
            .def("check", &colmap::FeaturePairsMatchingOptions::Check,
                 "Check parameters validation.")
            .def_readwrite("verify_matches",
                           &colmap::FeaturePairsMatchingOptions::verify_matches,
                           "bool: (Default ``True``) Whether to geometrically "
                           "verify the given matches.")
            .def_readwrite(
                    "match_list_path",
                    &colmap::FeaturePairsMatchingOptions::match_list_path,
                    "str: (Default ``''``) Path to the file with the matches.");
}

void pybind_bundle_adjustment_options(py::module& m) {
    // cloudViewer.reconstruction.options.BundleAdjustmentOptions
    py::class_<colmap::BundleAdjustmentOptions> bundle_adjustment_options(
            m, "BundleAdjustmentOptions", "Bundle Adjustment option class.");
    bundle_adjustment_options.def(py::init<>())
            .def("check", &colmap::BundleAdjustmentOptions::Check,
                 "Check parameters validation.")
            .def_readwrite(
                    "loss_function_type",
                    &colmap::BundleAdjustmentOptions::loss_function_type,
                    "LossFunctionType: (Default ``LossFunctionType::TRIVIAL``) "
                    "Loss function types: Trivial (non-robust) and Cauchy "
                    "(robust) loss.")
            .def_readwrite(
                    "loss_function_scale",
                    &colmap::BundleAdjustmentOptions::loss_function_scale,
                    "float: (Default ``1.0``) Scaling factor determines "
                    "residual at which robustification takes place.")
            .def_readwrite(
                    "refine_focal_length",
                    &colmap::BundleAdjustmentOptions::refine_focal_length,
                    "bool: (Default ``True``) Whether to refine the focal "
                    "length parameter group.")
            .def_readwrite(
                    "refine_principal_point",
                    &colmap::BundleAdjustmentOptions::refine_principal_point,
                    "bool: (Default ``False``) Whether to refine the principal "
                    "point parameter group.")
            .def_readwrite(
                    "refine_extra_params",
                    &colmap::BundleAdjustmentOptions::refine_extra_params,
                    "bool: (Default ``True``) Whether to refine the extra "
                    "parameter group.")
            .def_readwrite("refine_extrinsics",
                           &colmap::BundleAdjustmentOptions::refine_extrinsics,
                           "bool: (Default ``True``) Whether to refine the "
                           "extrinsic parameter group.")
            .def_readwrite("print_summary",
                           &colmap::BundleAdjustmentOptions::print_summary,
                           "bool: (Default ``True``) Whether to print a final "
                           "summary.")
            .def_readwrite(
                    "min_num_residuals_for_multi_threading",
                    &colmap::BundleAdjustmentOptions::
                            min_num_residuals_for_multi_threading,
                    "int: (Default ``50000``) Minimum number of residuals to "
                    "enable multi-threading. Note that single-threaded is "
                    "typically better for small bundle adjustment problems due "
                    "to the overhead of threading.");

    // cloudViewer.reconstruction.options.LossFunctionType
    py::enum_<colmap::BundleAdjustmentOptions::LossFunctionType>
            loss_function_type(bundle_adjustment_options, "LossFunctionType",
                               py::arithmetic());
    loss_function_type
            .value("TRIVIAL",
                   colmap::BundleAdjustmentOptions::LossFunctionType::TRIVIAL)
            .value("SOFT_L1",
                   colmap::BundleAdjustmentOptions::LossFunctionType::SOFT_L1)
            .value("CAUCHY",
                   colmap::BundleAdjustmentOptions::LossFunctionType::CAUCHY)
            .export_values();
    loss_function_type.attr("__doc__") = docstring::static_property(
            py::cpp_function([](py::handle arg) -> std::string {
                return "Enum class for Reconstruction LossFunction types.";
            }),
            py::none(), py::none(), "");
}

void pybind_incremental_triangulator_options(py::module& m) {
    // cloudViewer.reconstruction.options.IncrementalTriangulatorOptions
    py::class_<colmap::IncrementalTriangulator::Options>
            incremental_triangulator_options(
                    m, "IncrementalTriangulatorOptions",
                    "Incremental Triangulator option class.");
    incremental_triangulator_options.def(py::init<>())
            .def("check", &colmap::IncrementalTriangulator::Options::Check,
                 "Check parameters validation.")
            .def_readwrite(
                    "max_transitivity",
                    &colmap::IncrementalTriangulator::Options::max_transitivity,
                    "int: (Default ``1``) Maximum transitivity to search for "
                    "correspondences.")
            .def_readwrite("create_max_angle_error",
                           &colmap::IncrementalTriangulator::Options::
                                   create_max_angle_error,
                           "float: (Default ``2.0``) Maximum angular error to "
                           "create new triangulations.")
            .def_readwrite("continue_max_angle_error",
                           &colmap::IncrementalTriangulator::Options::
                                   continue_max_angle_error,
                           "float: (Default ``2.0``) Maximum angular error to "
                           "continue existing triangulations.")
            .def_readwrite("merge_max_reproj_error",
                           &colmap::IncrementalTriangulator::Options::
                                   merge_max_reproj_error,
                           "float: (Default ``4.0``) Maximum reprojection "
                           "error in pixels to merge triangulations.")
            .def_readwrite("complete_max_reproj_error",
                           &colmap::IncrementalTriangulator::Options::
                                   complete_max_reproj_error,
                           "float: (Default ``4.0``) Maximum reprojection "
                           "error to complete an existing triangulation.")
            .def_readwrite("complete_max_transitivity",
                           &colmap::IncrementalTriangulator::Options::
                                   complete_max_transitivity,
                           "int: (Default ``5``) Maximum transitivity for "
                           "track completion.")
            .def_readwrite("re_max_angle_error",
                           &colmap::IncrementalTriangulator::Options::
                                   re_max_angle_error,
                           "float: (Default ``5.0``) Maximum angular error to "
                           "re-triangulate under-reconstructed image pairs.")
            .def_readwrite(
                    "re_min_ratio",
                    &colmap::IncrementalTriangulator::Options::re_min_ratio,
                    "float: (Default ``0.2``) Minimum ratio of common "
                    "triangulations between an image pair over the number of "
                    "correspondences between that image pair to be considered "
                    "as under-reconstructed.")
            .def_readwrite(
                    "re_max_trials",
                    &colmap::IncrementalTriangulator::Options::re_max_trials,
                    "int: (Default ``1``) Maximum number of trials to "
                    "re-triangulate an image pair.")
            .def_readwrite("min_angle",
                           &colmap::IncrementalTriangulator::Options::min_angle,
                           "float: (Default ``1.5``) Minimum pairwise "
                           "triangulation angle for a stable triangulation.")
            .def_readwrite("ignore_two_view_tracks",
                           &colmap::IncrementalTriangulator::Options::
                                   ignore_two_view_tracks,
                           "bool: (Default ``True``) Whether to ignore "
                           "two-view tracks.")
            .def_readwrite("min_focal_length_ratio",
                           &colmap::IncrementalTriangulator::Options::
                                   min_focal_length_ratio,
                           "float: (Default ``0.1``) Thresholds for bogus "
                           "camera parameters. Images with bogus camera "
                           "parameters are ignored in triangulation.")
            .def_readwrite("max_focal_length_ratio",
                           &colmap::IncrementalTriangulator::Options::
                                   max_focal_length_ratio,
                           "float: (Default ``10.0``) Thresholds for bogus "
                           "camera parameters. Images with bogus camera "
                           "parameters are ignored in triangulation.")
            .def_readwrite(
                    "max_extra_param",
                    &colmap::IncrementalTriangulator::Options::max_extra_param,
                    "float: (Default ``1.0``) Thresholds for bogus camera "
                    "parameters. Images with bogus camera parameters are "
                    "ignored in triangulation.");
}

void pybind_parallel_bundle_adjustment_options(py::module& m) {
    // cloudViewer.reconstruction.options.ParallelBundleAdjustmentOptions
    py::class_<colmap::ParallelBundleAdjuster::Options>
            parallel_bundle_adjustment_options(
                    m, "ParallelBundleAdjustmentOptions",
                    "Parallel Bundle-Adjustment option class.");
    parallel_bundle_adjustment_options.def(py::init<>())
            .def("check", &colmap::ParallelBundleAdjuster::Options::Check,
                 "Check parameters validation.")
            .def_readwrite(
                    "print_summary",
                    &colmap::ParallelBundleAdjuster::Options::print_summary,
                    "bool: (Default ``True``) Whether to print a final "
                    "summary.")
            .def_readwrite(
                    "max_num_iterations",
                    &colmap::ParallelBundleAdjuster::Options::
                            max_num_iterations,
                    "int: (Default ``50``) Maximum number of iterations.")
            .def_readwrite("gpu_index",
                           &colmap::ParallelBundleAdjuster::Options::gpu_index,
                           "int: (Default ``-1``) Index of the GPU used for "
                           "bundle adjustment.")
            .def_readwrite(
                    "num_threads",
                    &colmap::ParallelBundleAdjuster::Options::num_threads,
                    "int: (Default ``-1``) Number of threads for CPU based "
                    "bundle adjustment.")
            .def_readwrite("min_num_residuals_for_multi_threading",
                           &colmap::ParallelBundleAdjuster::Options::
                                   min_num_residuals_for_multi_threading,
                           "int: (Default ``50000``) Minimum number of "
                           "residuals to enable multi-threading."
                           " Note that single-threaded is typically better for "
                           "small bundle adjustment problems due to the "
                           "overhead of threading.");
}

void pybind_incremental_mapper_options(py::module& m) {
    // cloudViewer.reconstruction.options.IncrementalMapperSubOptions
    py::class_<colmap::IncrementalMapper::Options>
            incremental_mapper_sub_options(
                    m, "IncrementalMapperSubOptions",
                    "Incremental Mapper sub option class.");
    incremental_mapper_sub_options.def(py::init<>())
            .def("check", &colmap::IncrementalMapper::Options::Check,
                 "Check parameters validation.")
            .def_readwrite(
                    "init_min_num_inliers",
                    &colmap::IncrementalMapper::Options::init_min_num_inliers,
                    "int: (Default ``100``) Minimum number of inliers for "
                    "initial image pair.")
            .def_readwrite(
                    "init_max_error",
                    &colmap::IncrementalMapper::Options::init_max_error,
                    "float: (Default ``4.0``) Maximum error in pixels for "
                    "two-view geometry estimation for initial image pair.")
            .def_readwrite("init_max_forward_motion",
                           &colmap::IncrementalMapper::Options::
                                   init_max_forward_motion,
                           "float: (Default ``0.95``) Maximum forward motion "
                           "for initial image pair.")
            .def_readwrite(
                    "init_min_tri_angle",
                    &colmap::IncrementalMapper::Options::init_min_tri_angle,
                    "float: (Default ``16.0``) Minimum triangulation angle for "
                    "initial image pair.")
            .def_readwrite(
                    "init_max_reg_trials",
                    &colmap::IncrementalMapper::Options::init_max_reg_trials,
                    "int: (Default ``2``) Maximum number of trials to use an "
                    "image for initialization.")
            .def_readwrite(
                    "abs_pose_max_error",
                    &colmap::IncrementalMapper::Options::abs_pose_max_error,
                    "float: (Default ``12.0``) Maximum reprojection error in "
                    "absolute pose estimation.")
            .def_readwrite("abs_pose_min_num_inliers",
                           &colmap::IncrementalMapper::Options::
                                   abs_pose_min_num_inliers,
                           "int: (Default ``30``) Minimum number of inliers in "
                           "absolute pose estimation.")
            .def_readwrite("abs_pose_min_inlier_ratio",
                           &colmap::IncrementalMapper::Options::
                                   abs_pose_min_inlier_ratio,
                           "float: (Default ``0.25``) Minimum inlier ratio in "
                           "absolute pose estimation.")
            .def_readwrite("abs_pose_refine_focal_length",
                           &colmap::IncrementalMapper::Options::
                                   abs_pose_refine_focal_length,
                           "bool: (Default ``True``) Whether to estimate the "
                           "focal length in absolute pose estimation.")
            .def_readwrite("abs_pose_refine_extra_params",
                           &colmap::IncrementalMapper::Options::
                                   abs_pose_refine_extra_params,
                           "bool: (Default ``True``) Whether to estimate the "
                           "extra parameters in absolute pose estimation.")
            .def_readwrite(
                    "local_ba_num_images",
                    &colmap::IncrementalMapper::Options::local_ba_num_images,
                    "int: (Default ``6``) Number of images to optimize in "
                    "local bundle adjustment.")
            .def_readwrite(
                    "local_ba_min_tri_angle",
                    &colmap::IncrementalMapper::Options::local_ba_min_tri_angle,
                    "float: (Default ``6.0``) Minimum triangulation for images "
                    "to be chosen in local bundle adjustment.")
            .def_readwrite(
                    "min_focal_length_ratio",
                    &colmap::IncrementalMapper::Options::min_focal_length_ratio,
                    "float: (Default ``0.1``) Opening angle of ~130deg. "
                    "Thresholds for bogus camera parameters. "
                    "Images with bogus camera parameters are filtered and "
                    "ignored in triangulation.")
            .def_readwrite(
                    "max_focal_length_ratio",
                    &colmap::IncrementalMapper::Options::max_focal_length_ratio,
                    "float: (Default ``10.0``) Opening angle of ~5deg. "
                    "Thresholds for bogus camera parameters. "
                    "Images with bogus camera parameters are filtered and "
                    "ignored in triangulation.")
            .def_readwrite("max_extra_param",
                           &colmap::IncrementalMapper::Options::max_extra_param,
                           "float: (Default ``1.0``) Thresholds for bogus "
                           "camera parameters. "
                           "Images with bogus camera parameters are filtered "
                           "and ignored in triangulation.")
            .def_readwrite("filter_max_reproj_error",
                           &colmap::IncrementalMapper::Options::
                                   filter_max_reproj_error,
                           "float: (Default ``4.0``) Maximum reprojection "
                           "error in pixels for observations.")
            .def_readwrite(
                    "filter_min_tri_angle",
                    &colmap::IncrementalMapper::Options::filter_min_tri_angle,
                    "float: (Default ``1.5``) Minimum triangulation angle in "
                    "degrees for stable 3D points.")
            .def_readwrite("max_reg_trials",
                           &colmap::IncrementalMapper::Options::max_reg_trials,
                           "int: (Default ``3``) Maximum number of trials to "
                           "register an image.")
            .def_readwrite(
                    "fix_existing_images",
                    &colmap::IncrementalMapper::Options::fix_existing_images,
                    "bool: (Default ``False``) If reconstruction is provided "
                    "as input, fix the existing image poses.")
            .def_readwrite("num_threads",
                           &colmap::IncrementalMapper::Options::num_threads,
                           "int: (Default ``-1``) Number of threads.")
            .def_readwrite(
                    "image_selection_method",
                    &colmap::IncrementalMapper::Options::image_selection_method,
                    "ImageSelectionMethod: (Default "
                    "``ImageSelectionMethod::MIN_UNCERTAINTY``) Method to find "
                    "and select next best image to register.");

    // cloudViewer.reconstruction.options.ImageSelectionMethod
    py::enum_<colmap::IncrementalMapper::Options::ImageSelectionMethod>
            image_selection_method(incremental_mapper_sub_options,
                                   "ImageSelectionMethod", py::arithmetic());
    image_selection_method
            .value("MAX_VISIBLE_POINTS_NUM",
                   colmap::IncrementalMapper::Options::ImageSelectionMethod::
                           MAX_VISIBLE_POINTS_NUM)
            .value("MAX_VISIBLE_POINTS_RATIO",
                   colmap::IncrementalMapper::Options::ImageSelectionMethod::
                           MAX_VISIBLE_POINTS_RATIO)
            .value("MIN_UNCERTAINTY",
                   colmap::IncrementalMapper::Options::ImageSelectionMethod::
                           MIN_UNCERTAINTY)
            .export_values();
    image_selection_method.attr("__doc__") = docstring::static_property(
            py::cpp_function([](py::handle arg) -> std::string {
                return "Enum class for Reconstruction ImageSelection Method.";
            }),
            py::none(), py::none(), "");

    // cloudViewer.reconstruction.options.IncrementalMapperOptions
    py::class_<colmap::IncrementalMapperOptions> incremental_mapper_options(
            m, "IncrementalMapperOptions", "Incremental Mapper option class.");
    incremental_mapper_options.def(py::init<>())
            .def("check", &colmap::IncrementalMapperOptions::Check,
                 "Check parameters validation.")
            .def("get_mapper", &colmap::IncrementalMapperOptions::Mapper,
                 "Get Incremental Mapper sub options.")
            .def("get_incremental_triangulator_options",
                 &colmap::IncrementalMapperOptions::Triangulation,
                 "Get Incremental triangulator options.")
            .def("get_local_ba_options",
                 &colmap::IncrementalMapperOptions::LocalBundleAdjustment,
                 "Get local BundleAdjustment options.")
            .def("get_global_ba_options",
                 &colmap::IncrementalMapperOptions::GlobalBundleAdjustment,
                 "Get global BundleAdjustment options.")
            .def("get_parallel_global_ba_options",
                 &colmap::IncrementalMapperOptions::
                         ParallelGlobalBundleAdjustment,
                 "Get parallel global BundleAdjustment options.")
            .def_readwrite("min_num_matches",
                           &colmap::IncrementalMapperOptions::min_num_matches,
                           "int: (Default ``15``) The minimum number of "
                           "matches for inlier matches to be considered.")
            .def_readwrite("ignore_watermarks",
                           &colmap::IncrementalMapperOptions::ignore_watermarks,
                           "bool: (Default ``False``) Whether to ignore the "
                           "inlier matches of watermark image pairs.")
            .def_readwrite("multiple_models",
                           &colmap::IncrementalMapperOptions::multiple_models,
                           "bool: (Default ``True``) Whether to reconstruct "
                           "multiple sub-models.")
            .def_readwrite("max_num_models",
                           &colmap::IncrementalMapperOptions::max_num_models,
                           "int: (Default ``50``) The number of sub-models to "
                           "reconstruct.")
            .def_readwrite(
                    "max_model_overlap",
                    &colmap::IncrementalMapperOptions::max_model_overlap,
                    "int: (Default ``20``) The maximum number of overlapping "
                    "images between sub-models. If the current sub-models "
                    "shares more than this number of images with another "
                    "model, then the reconstruction is stopped.")
            .def_readwrite("min_model_size",
                           &colmap::IncrementalMapperOptions::min_model_size,
                           "int: (Default ``10``) The minimum number of "
                           "registered images of a sub-model, otherwise the "
                           "sub-model is discarded.")
            .def_readwrite("init_image_id1",
                           &colmap::IncrementalMapperOptions::init_image_id1,
                           "int: (Default ``-1``) The image identifiers used "
                           "to initialize the reconstruction. "
                           "Note that only one or both image identifiers can "
                           "be specified. In the former case, the second image "
                           "is automatically determined.")
            .def_readwrite("init_image_id2",
                           &colmap::IncrementalMapperOptions::init_image_id2,
                           "int: (Default ``-1``) The image identifiers used "
                           "to initialize the reconstruction. "
                           "Note that only one or both image identifiers can "
                           "be specified. In the former case, the second image "
                           "is automatically determined.")
            .def_readwrite("init_num_trials",
                           &colmap::IncrementalMapperOptions::init_num_trials,
                           "int: (Default ``200``) The number of trials to "
                           "initialize the reconstruction.")
            .def_readwrite("extract_colors",
                           &colmap::IncrementalMapperOptions::extract_colors,
                           "bool: (Default ``True``) Whether to extract colors "
                           "for reconstructed points.")
            .def_readwrite("num_threads",
                           &colmap::IncrementalMapperOptions::num_threads,
                           "int: (Default ``-1``) The number of threads to use "
                           "during reconstruction.")
            .def_readwrite(
                    "min_focal_length_ratio",
                    &colmap::IncrementalMapperOptions::min_focal_length_ratio,
                    "float: (Default ``0.1``) Thresholds for filtering images "
                    "with degenerate intrinsics.")
            .def_readwrite(
                    "max_focal_length_ratio",
                    &colmap::IncrementalMapperOptions::max_focal_length_ratio,
                    "float: (Default ``10.0``) Thresholds for filtering images "
                    "with degenerate intrinsics.")
            .def_readwrite("max_extra_param",
                           &colmap::IncrementalMapperOptions::max_extra_param,
                           "float: (Default ``1.0``) Thresholds for filtering "
                           "images with degenerate intrinsics.")
            .def_readwrite(
                    "ba_refine_focal_length",
                    &colmap::IncrementalMapperOptions::ba_refine_focal_length,
                    "bool: (Default ``True``) Which intrinsic parameters to "
                    "optimize during the reconstruction.")
            .def_readwrite("ba_refine_principal_point",
                           &colmap::IncrementalMapperOptions::
                                   ba_refine_principal_point,
                           "bool: (Default ``False``) Which intrinsic "
                           "parameters to optimize during the reconstruction.")
            .def_readwrite(
                    "ba_refine_extra_params",
                    &colmap::IncrementalMapperOptions::ba_refine_extra_params,
                    "bool: (Default ``True``) Which intrinsic parameters to "
                    "optimize during the reconstruction.")
            .def_readwrite("ba_min_num_residuals_for_multi_threading",
                           &colmap::IncrementalMapperOptions::
                                   ba_min_num_residuals_for_multi_threading,
                           "int: (Default ``50000``) The minimum number of "
                           "residuals per bundle adjustment problem to enable "
                           "multi-threading solving of the problems.")
            .def_readwrite(
                    "ba_local_num_images",
                    &colmap::IncrementalMapperOptions::ba_local_num_images,
                    "int: (Default ``6``) The number of images to optimize in "
                    "local bundle adjustment.")
            .def_readwrite("ba_local_function_tolerance",
                           &colmap::IncrementalMapperOptions::
                                   ba_local_function_tolerance,
                           "float: (Default ``0.0``) Ceres solver function "
                           "tolerance for local bundle adjustment.")
            .def_readwrite("ba_local_max_num_iterations",
                           &colmap::IncrementalMapperOptions::
                                   ba_local_max_num_iterations,
                           "int: (Default ``25``) The maximum number of local "
                           "bundle adjustment iterations.")
            .def_readwrite("ba_global_use_pba",
                           &colmap::IncrementalMapperOptions::ba_global_use_pba,
                           "bool: (Default ``False``) Whether to use PBA in "
                           "global bundle adjustment.")
            .def_readwrite(
                    "ba_global_pba_gpu_index",
                    &colmap::IncrementalMapperOptions::ba_global_pba_gpu_index,
                    "int: (Default ``-1``) The GPU index for PBA bundle "
                    "adjustment.")
            .def_readwrite(
                    "ba_global_images_ratio",
                    &colmap::IncrementalMapperOptions::ba_global_images_ratio,
                    "float: (Default ``1.1``) The growth rates after which to "
                    "perform global bundle adjustment.")
            .def_readwrite(
                    "ba_global_points_ratio",
                    &colmap::IncrementalMapperOptions::ba_global_points_ratio,
                    "float: (Default ``1.1``) The growth rates after which to "
                    "perform global bundle adjustment.")
            .def_readwrite(
                    "ba_global_images_freq",
                    &colmap::IncrementalMapperOptions::ba_global_images_freq,
                    "int: (Default ``500``) The growth rates after which to "
                    "perform global bundle adjustment.")
            .def_readwrite(
                    "ba_global_points_freq",
                    &colmap::IncrementalMapperOptions::ba_global_points_freq,
                    "int: (Default ``250000``) The growth rates after which to "
                    "perform global bundle adjustment.")
            .def_readwrite("ba_global_function_tolerance",
                           &colmap::IncrementalMapperOptions::
                                   ba_global_function_tolerance,
                           "float: (Default ``0.0``) Ceres solver function "
                           "tolerance for global bundle adjustment")
            .def_readwrite("ba_global_max_num_iterations",
                           &colmap::IncrementalMapperOptions::
                                   ba_global_max_num_iterations,
                           "int: (Default ``50``) The maximum number of global "
                           "bundle adjustment iterations.")
            .def_readwrite(
                    "ba_local_max_refinements",
                    &colmap::IncrementalMapperOptions::ba_local_max_refinements,
                    "int: (Default ``2``) The thresholds for iterative bundle "
                    "adjustment refinements.")
            .def_readwrite("ba_local_max_refinement_change",
                           &colmap::IncrementalMapperOptions::
                                   ba_local_max_refinement_change,
                           "float: (Default ``0.001``) The thresholds for "
                           "iterative bundle adjustment refinements.")
            .def_readwrite("ba_global_max_refinements",
                           &colmap::IncrementalMapperOptions::
                                   ba_global_max_refinements,
                           "int: (Default ``5``) The thresholds for iterative "
                           "bundle adjustment refinements.")
            .def_readwrite("ba_global_max_refinement_change",
                           &colmap::IncrementalMapperOptions::
                                   ba_global_max_refinement_change,
                           "float: (Default ``0.0005``) The thresholds for "
                           "iterative bundle adjustment refinements.")
            .def_readwrite("snapshot_path",
                           &colmap::IncrementalMapperOptions::snapshot_path,
                           "str: (Default ``''``) Path to a folder with "
                           "reconstruction snapshots during incremental "
                           "reconstruction. Snapshots will be saved according "
                           "to the specified frequency of registered images.")
            .def_readwrite(
                    "snapshot_images_freq",
                    &colmap::IncrementalMapperOptions::snapshot_images_freq,
                    "int: (Default ``0``) The frequency of registered images.")
            .def_readwrite("image_names",
                           &colmap::IncrementalMapperOptions::image_names,
                           "set(str, ...): Which images to reconstruct. If no "
                           "images are specified, all images will be "
                           "reconstructed by default.")
            .def_readwrite(
                    "fix_existing_images",
                    &colmap::IncrementalMapperOptions::fix_existing_images,
                    "bool: (Default ``False``) If reconstruction is provided "
                    "as input, fix the existing image poses.")
            .def_readwrite("mapper", &colmap::IncrementalMapperOptions::mapper,
                           "IncrementalMapperSubOptions: Incremental mapper "
                           "sub options.")
            .def_readwrite("triangulation",
                           &colmap::IncrementalMapperOptions::triangulation,
                           "IncrementalTriangulatorOptions: Incremental "
                           "triangulator options.");
}

void pybind_patch_match_options(py::module& m) {
    // cloudViewer.reconstruction.options.PatchMatchOptions
    py::class_<colmap::mvs::PatchMatchOptions> patch_match_options(
            m, "PatchMatchOptions", "PatchMatch option class.");
    patch_match_options.def(py::init<>())
            .def("check", &colmap::mvs::PatchMatchOptions::Check,
                 "Check parameters validation.")
            .def_readwrite("max_image_size",
                           &colmap::mvs::PatchMatchOptions::max_image_size,
                           "int: (Default ``-1``) Maximum image size in either "
                           "dimension.")
            .def_readwrite(
                    "gpu_index", &colmap::mvs::PatchMatchOptions::gpu_index,
                    "str: (Default ``'-1'``) Index of the GPU used for patch "
                    "match. For multi-GPU usage, you should separate multiple "
                    "GPU indices by comma, e.g., ``0,1,2,3``.");
}

void pybind_stereo_fusion_options(py::module& m) {
    // cloudViewer.reconstruction.options.StereoFusionOptions
    py::class_<colmap::mvs::StereoFusionOptions> stereo_fusion_options(
            m, "StereoFusionOptions", "Stereo fusion option class.");
    stereo_fusion_options.def(py::init<>())
            .def("check", &colmap::mvs::StereoFusionOptions::Check,
                 "Check parameters validation.")
            .def("print", &colmap::mvs::StereoFusionOptions::Print,
                 "Print the options to stdout..")
            .def_readwrite("mask_path",
                           &colmap::mvs::StereoFusionOptions::mask_path,
                           "str: (Default ``''``) Path for PNG masks. Same "
                           "format expected as ImageReaderOptions.")
            .def_readwrite("num_threads",
                           &colmap::mvs::StereoFusionOptions::num_threads,
                           "int: (Default ``-1``) The number of threads to use "
                           "during fusion.")
            .def_readwrite("max_image_size",
                           &colmap::mvs::StereoFusionOptions::max_image_size,
                           "int: (Default ``-1``) Maximum image size in either "
                           "dimension.")
            .def_readwrite("min_num_pixels",
                           &colmap::mvs::StereoFusionOptions::min_num_pixels,
                           "int: (Default ``5``) Minimum number of fused "
                           "pixels to produce a point.")
            .def_readwrite("max_num_pixels",
                           &colmap::mvs::StereoFusionOptions::max_num_pixels,
                           "int: (Default ``10000``) Maximum number of pixels "
                           "to fuse into a single point.")
            .def_readwrite(
                    "max_traversal_depth",
                    &colmap::mvs::StereoFusionOptions::max_traversal_depth,
                    "int: (Default ``100``) Maximum depth in consistency graph "
                    "traversal.")
            .def_readwrite("max_reproj_error",
                           &colmap::mvs::StereoFusionOptions::max_reproj_error,
                           "float: (Default ``2.0``) Maximum relative "
                           "difference between measured and projected pixel.")
            .def_readwrite("max_depth_error",
                           &colmap::mvs::StereoFusionOptions::max_depth_error,
                           "float: (Default ``0.01``) Maximum relative "
                           "difference between measured and projected depth.")
            .def_readwrite(
                    "max_normal_error",
                    &colmap::mvs::StereoFusionOptions::max_normal_error,
                    "float: (Default ``10.0``) Maximum angular difference in "
                    "degrees of normals of pixels to be fused.")
            .def_readwrite("check_num_images",
                           &colmap::mvs::StereoFusionOptions::check_num_images,
                           "int: (Default ``50``) Number of overlapping "
                           "images to transitively check for fusing points.")
            .def_readwrite("use_cache",
                           &colmap::mvs::StereoFusionOptions::use_cache,
                           "bool: (Default ``False``) Flag indicating whether "
                           "to use LRU cache or pre-load all data.")
            .def_readwrite(
                    "cache_size", &colmap::mvs::StereoFusionOptions::cache_size,
                    "float: (Default ``32.0``) Cache size in gigabytes for "
                    "fusion. The fusion keeps the bitmaps, depth maps, normal "
                    "maps, and consistency graphs of this number of images in "
                    "memory. A higher value leads to less disk access and "
                    "faster fusion, while a lower value leads to reduced "
                    "memory usage. Note that a single image can consume a lot "
                    "of memory, if the consistency graph is dense.")
            .def_readwrite("bounding_box",
                           &colmap::mvs::StereoFusionOptions::bounding_box,
                           "List((), ()): The bounding box size.");
}

void pybind_poisson_meshing_options(py::module& m) {
    // cloudViewer.reconstruction.options.PoissonMeshingOptions
    py::class_<colmap::mvs::PoissonMeshingOptions> poisson_meshing_options(
            m, "PoissonMeshingOptions", "Poisson meshing option class.");
    poisson_meshing_options.def(py::init<>())
            .def("check", &colmap::mvs::PoissonMeshingOptions::Check,
                 "Check parameters validation.")
            .def_readwrite("point_weight",
                           &colmap::mvs::PoissonMeshingOptions::point_weight,
                           "float: (Default ``1.0``) This floating point value "
                           "specifies the importance that interpolation of the "
                           "point samples is given in the formulation of the "
                           "screened Poisson equation. The results of the "
                           "original (unscreened) Poisson Reconstruction can "
                           "be obtained by setting this value to 0.")
            .def_readwrite(
                    "depth", &colmap::mvs::PoissonMeshingOptions::depth,
                    "int: (Default ``'13'``) This integer is the maximum "
                    "depth of the tree that will be used for surface "
                    "reconstruction. Running at depth d corresponds to solving "
                    "on a voxel grid whose resolution is no larger than 2^d x "
                    "2^d x 2^d. Note that since the reconstructor adapts the "
                    "octree to the sampling density, the specified "
                    "reconstruction depth is only an upper bound.")
            .def_readwrite(
                    "color", &colmap::mvs::PoissonMeshingOptions::color,
                    "float: (Default ``'32.0'``) If specified, the "
                    "reconstruction code assumes that the input is equipped "
                    "with colors and will extrapolate the color values to the "
                    "vertices of the reconstructed mesh. The floating point "
                    "value specifies the relative importance of finer color "
                    "estimates over lower ones.")
            .def_readwrite("trim", &colmap::mvs::PoissonMeshingOptions::trim,
                           "float: (Default ``'10.0'``) This floating point "
                           "values specifies the value for mesh trimming. The "
                           "subset of the mesh with signal value less than the "
                           "trim value is discarded.")
            .def_readwrite("num_threads",
                           &colmap::mvs::PoissonMeshingOptions::num_threads,
                           "int: (Default ``'-1'``) The number of threads used "
                           "for the Poisson reconstruction.");
}

void pybind_delaunay_meshing_options(py::module& m) {
    // cloudViewer.reconstruction.options.DelaunayMeshingOptions
    py::class_<colmap::mvs::DelaunayMeshingOptions> delaunay_meshing_options(
            m, "DelaunayMeshingOptions", "Delaunay meshing option class.");
    delaunay_meshing_options.def(py::init<>())
            .def("check", &colmap::mvs::DelaunayMeshingOptions::Check,
                 "Check parameters validation.")
            .def_readwrite("max_proj_dist",
                           &colmap::mvs::DelaunayMeshingOptions::max_proj_dist,
                           "float: (Default ``20.0``) Unify input points into "
                           "one cell in the Delaunay triangulation that fall "
                           "within a reprojected radius of the given pixels.")
            .def_readwrite(
                    "max_depth_dist",
                    &colmap::mvs::DelaunayMeshingOptions::max_depth_dist,
                    "float: (Default ``'0.05'``) Maximum relative depth "
                    "difference between input point and a vertex of an "
                    "existing cell in the Delaunay triangulation, otherwise a "
                    "new vertex is created in the triangulation.")
            .def_readwrite(
                    "visibility_sigma",
                    &colmap::mvs::DelaunayMeshingOptions::visibility_sigma,
                    "float: (Default ``'3.0'``) The standard deviation of wrt. "
                    "the number of images seen by each point. Increasing this "
                    "value decreases the influence of points seen in few "
                    "images.")
            .def_readwrite(
                    "distance_sigma_factor",
                    &colmap::mvs::DelaunayMeshingOptions::distance_sigma_factor,
                    "float: (Default ``'1.0'``) The factor that is applied to "
                    "the computed distance sigma, which is automatically "
                    "computed as the 25th percentile of edge lengths. A higher "
                    "value will increase the smoothness of the surface.")
            .def_readwrite("quality_regularization",
                           &colmap::mvs::DelaunayMeshingOptions::
                                   quality_regularization,
                           "float: (Default ``'1.0'``) A higher quality "
                           "regularization leads to a smoother surface.")
            .def_readwrite(
                    "max_side_length_factor",
                    &colmap::mvs::DelaunayMeshingOptions::
                            max_side_length_factor,
                    "float: (Default ``'25.0'``) Filtering thresholds for "
                    "outlier surface mesh faces. If the longest side of a mesh "
                    "face (longest out of 3) exceeds the side lengths of all "
                    "faces at a certain percentile by the given factor, then "
                    "it is considered an outlier mesh face and discarded.")
            .def_readwrite(
                    "max_side_length_percentile",
                    &colmap::mvs::DelaunayMeshingOptions::
                            max_side_length_percentile,
                    "float: (Default ``'95.0'``) Filtering thresholds for "
                    "outlier surface mesh faces. If the longest side of a mesh "
                    "face (longest out of 3) exceeds the side lengths of all "
                    "faces at a certain percentile by the given factor, then "
                    "it is considered an outlier mesh face and discarded.")
            .def_readwrite("num_threads",
                           &colmap::mvs::DelaunayMeshingOptions::num_threads,
                           "int: (Default ``'-1'``) The number of threads to "
                           "use for reconstruction. Default is all threads.");
}

void pybind_reconstruction_options(py::module& m) {
    py::module m_submodule =
            m.def_submodule("options", "Reconstruction options");
    pybind_image_reader_options(m_submodule);
    pybind_sift_extraction_options(m_submodule);
    pybind_sift_matching_options(m_submodule);
    pybind_exhaustive_matching_options(m_submodule);
    pybind_sequential_matching_options(m_submodule);
    pybind_vocabtree_matching_options(m_submodule);
    pybind_spatial_matching_options(m_submodule);
    pybind_transitive_matching_options(m_submodule);
    pybind_imagepairs_matching_options(m_submodule);
    pybind_featurepairs_matching_options(m_submodule);
    pybind_bundle_adjustment_options(m_submodule);
    pybind_incremental_triangulator_options(m_submodule);
    pybind_parallel_bundle_adjustment_options(m_submodule);
    pybind_incremental_mapper_options(m_submodule);
    pybind_patch_match_options(m_submodule);
    pybind_stereo_fusion_options(m_submodule);
    pybind_poisson_meshing_options(m_submodule);
    pybind_delaunay_meshing_options(m_submodule);
}

}  // namespace options
}  // namespace reconstruction
}  // namespace cloudViewer
