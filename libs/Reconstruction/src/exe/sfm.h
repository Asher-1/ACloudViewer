// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <filesystem>
#include <functional>
#include <memory>

#include "controllers/global_pipeline.h"
#include "controllers/hierarchical_pipeline.h"
#include "controllers/incremental_pipeline.h"
#include "scene/reconstruction.h"
#include "scene/reconstruction_manager.h"

namespace colmap {

// Upstream parity (d3ccaf35 exe/sfm.h): the programmatic SfM entry points.
// The fork equivalents drive IncrementalMapperController /
// HierarchicalMapperController (the fork's pipeline classes) with the same
// contracts: cancellation through the stop probe, sub-model flushing for
// fresh reconstructions, and direct output writes when continuing an
// existing one.

void RunPointTriangulatorImpl(
        const std::shared_ptr<Reconstruction>& reconstruction,
        const std::filesystem::path& database_path,
        const std::filesystem::path& image_path,
        const std::filesystem::path& output_path,
        const IncrementalMapperOptions& options,
        bool clear_points,
        bool refine_intrinsics,
        std::function<bool()> check_if_stopped = {});

bool RunIncrementalMapperImpl(
        const std::filesystem::path& database_path,
        const std::filesystem::path& image_path,
        const std::filesystem::path& output_path,
        const std::shared_ptr<IncrementalMapperOptions>& mapper_options,
        std::shared_ptr<ReconstructionManager>& reconstruction_manager,
        std::function<void()> initial_image_pair_callback = {},
        std::function<void()> next_image_callback = {},
        std::function<bool()> check_if_stopped = {});

bool RunGlobalMapperImpl(
        const std::filesystem::path& database_path,
        const std::filesystem::path& image_path,
        const std::filesystem::path& output_path,
        const std::shared_ptr<GlobalPipelineOptions>& mapper_options,
        std::shared_ptr<ReconstructionManager>& reconstruction_manager);

bool RunHierarchicalMapperImpl(
        const HierarchicalMapperController::Options& controller_options,
        const SceneClustering::Options& clustering_options,
        const IncrementalMapperOptions& mapper_options,
        const std::filesystem::path& output_path,
        ReconstructionManager* reconstruction_manager);

int RunAutomaticReconstructor(int argc, char** argv);
int RunBundleAdjuster(int argc, char** argv);
int RunColorExtractor(int argc, char** argv);
int RunMapper(int argc, char** argv);
int RunPosePriorMapper(int argc, char** argv);
int RunHierarchicalMapper(int argc, char** argv);

// Upstream parity (dbb41680 exe/sfm.h): the global SfM pipeline command.
int RunGlobalMapper(int argc, char** argv);

// Upstream parity (dbb41680 exe/sfm.h): the rotation averaging command.
int RunRotationAverager(int argc, char** argv);

// Upstream parity (dbb41680 exe/sfm.h): the view graph calibration command.
int RunViewGraphCalibrator(int argc, char** argv);
int RunPointFiltering(int argc, char** argv);
int RunPointTriangulator(int argc, char** argv);

}  // namespace colmap
