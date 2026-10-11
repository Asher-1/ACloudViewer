# Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
#       its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

set(FOLDER_NAME "controllers")

COLMAP_ADD_SOURCES(
    base_option_manager.h base_option_manager.cc
    automatic_reconstruction.h automatic_reconstruction.cc
    bundle_adjustment.h bundle_adjustment.cc
    da3_depth_controller.h da3_depth_controller.cc
    global_mapper.h global_mapper.cc
    global_pipeline.h global_pipeline.cc
    rotation_averaging.h rotation_averaging.cc
    hierarchical_pipeline.h hierarchical_pipeline.cc
    incremental_pipeline.h incremental_pipeline.cc
    reconstruction_clustering.h reconstruction_clustering.cc
    texturing_controller.h texturing_controller.cc
    image_reader.h image_reader.cc
    single_view_calibration.h single_view_calibration.cc
    option_manager.h option_manager.cc
    feature_extraction.h feature_extraction.cc
    matcher_cache.h matcher_cache.cc
    pairing.h pairing.cc
    feature_matching.h feature_matching.cc
    feature_matching_utils.h feature_matching_utils.cc
    undistorters.h undistorters.cc
)

COLMAP_ADD_TEST(da3_depth_controller_test da3_depth_controller_test.cc)

COLMAP_ADD_TEST(global_pipeline_test global_pipeline_test.cc)
COLMAP_ADD_TEST(single_view_calibration_test single_view_calibration_test.cc)
# TEMP-DIAG: PRNG seed scan for the non-trivial-rig filtering variance;
# DISABLED_ test, run manually with --gtest_also_run_disabled_tests.
COLMAP_ADD_TEST(pipeline_rng_scan_test pipeline_rng_scan_test.cc)
# TEMP-AUDIT: same-seed fork-vs-upstream pipeline scan (remove after audit).
COLMAP_ADD_TEST(pipeline_seed_scan_test pipeline_seed_scan_test.cc)
COLMAP_ADD_TEST(controllers_bundle_adjustment_test bundle_adjustment_test.cc)
COLMAP_ADD_TEST(image_reader_test image_reader_test.cc)
COLMAP_ADD_TEST(option_manager_test option_manager_test.cc)
COLMAP_ADD_TEST(controllers_reconstruction_clustering_test reconstruction_clustering_test.cc)
COLMAP_ADD_TEST(controllers_rotation_averaging_test rotation_averaging_test.cc)
COLMAP_ADD_TEST(matcher_cache_test matcher_cache_test.cc)
COLMAP_ADD_TEST(base_option_manager_test base_option_manager_test.cc)
COLMAP_ADD_TEST(incremental_pipeline_test incremental_pipeline_test.cc)
COLMAP_ADD_TEST(hierarchical_pipeline_test hierarchical_pipeline_test.cc)
COLMAP_ADD_TEST(feature_matching_utils_test feature_matching_utils_test.cc)
COLMAP_ADD_TEST(controllers_feature_matching_test feature_matching_test.cc)
COLMAP_ADD_TEST(pairing_test pairing_test.cc)
COLMAP_ADD_TEST(undistorters_test undistorters_test.cc)
