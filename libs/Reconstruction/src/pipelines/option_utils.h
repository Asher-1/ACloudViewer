// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <filesystem>
#include <limits>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "controllers/image_reader.h"
#include "controllers/incremental_pipeline.h"
#include "controllers/pairing.h"
#include "estimators/bundle_adjustment.h"
#include "estimators/bundle_adjustment_caspar.h"
#include "estimators/bundle_adjustment_ceres.h"
#include "estimators/two_view_geometry.h"
#include "feature/extractor.h"
#include "feature/matcher.h"
#include "mvs/delaunay_meshing.h"
#include "mvs/fusion.h"
#include "mvs/mesh_postprocessing.h"
#include "mvs/patch_match.h"
#include "mvs/poisson_meshing.h"

namespace cloudViewer {

// Serializes option structs into an argv vector consumed by the colmap exe
// entry points (exe/*.h). Option names must match the names registered by
// colmap::OptionManager exactly.
class OptionsParser {
public:
    OptionsParser();
    ~OptionsParser() { reset(); }

    void reset();

    inline bool parseOptions() {
        return parseOptions(this->argc_, this->argv_);
    }
    bool parseOptions(int& argc, char**& argv);

    inline int getArgc() const { return this->argc_; }
    inline char** getArgv() { return this->argv_; }

    void registerOption(const std::string& name, const bool* option) {
        options_.emplace_back(name, *option ? "true" : "false");
    }
    void registerOption(const std::string& name, const int* option) {
        options_.emplace_back(name, std::to_string(*option));
    }
    void registerOption(const std::string& name, const std::size_t* option) {
        options_.emplace_back(name, std::to_string(*option));
    }
    void registerOption(const std::string& name, const double* option) {
        std::ostringstream os;
        os.precision(std::numeric_limits<double>::max_digits10);
        os << *option;
        options_.emplace_back(name, os.str());
    }
    // Empty strings/paths are not forwarded so that the exe defaults apply.
    void registerOption(const std::string& name, const std::string* option) {
        if (!option->empty()) {
            options_.emplace_back(name, *option);
        }
    }
    void registerOption(const std::string& name,
                        const std::filesystem::path* option) {
        if (!option->empty()) {
            options_.emplace_back(name, option->string());
        }
    }
    // Enum classes with a MAKE_ENUM_CLASS_OVERLOAD_STREAM operator<<.
    template <typename Enum,
              typename = std::enable_if_t<std::is_enum<Enum>::value>>
    void registerOption(const std::string& name, const Enum* option) {
        std::ostringstream os;
        os << *option;
        options_.emplace_back(name, os.str());
    }

    // colmap
    void addFeatureExtractionOptions(
            const colmap::ImageReaderOptions& image_reader_options,
            const colmap::FeatureExtractionOptions& extraction_options);

    void addFeatureMatchingOptions(
            const colmap::FeatureMatchingOptions& matching_options);

    void addTwoViewGeometryOptions(
            const colmap::TwoViewGeometryOptions& geometry_options);

    void addExhaustivePairingOptions(
            const colmap::FeatureMatchingOptions& matching_options,
            const colmap::TwoViewGeometryOptions& geometry_options,
            const colmap::ExhaustivePairingOptions& pairing_options);

    void addSequentialPairingOptions(
            const colmap::FeatureMatchingOptions& matching_options,
            const colmap::TwoViewGeometryOptions& geometry_options,
            const colmap::SequentialPairingOptions& pairing_options);

    void addSpatialPairingOptions(
            const colmap::FeatureMatchingOptions& matching_options,
            const colmap::TwoViewGeometryOptions& geometry_options,
            const colmap::SpatialPairingOptions& pairing_options);

    void addTransitivePairingOptions(
            const colmap::FeatureMatchingOptions& matching_options,
            const colmap::TwoViewGeometryOptions& geometry_options,
            const colmap::TransitivePairingOptions& pairing_options);

    void addVocabTreePairingOptions(
            const colmap::FeatureMatchingOptions& matching_options,
            const colmap::TwoViewGeometryOptions& geometry_options,
            const colmap::VocabTreePairingOptions& pairing_options);

    void addImportedPairingOptions(
            const colmap::FeatureMatchingOptions& matching_options,
            const colmap::TwoViewGeometryOptions& geometry_options,
            const colmap::ImportedPairingOptions& pairing_options);

    void addMapperOptions(const colmap::IncrementalPipelineOptions&
                                  incremental_mapper_options);

    void addBundleAdjustmentOptions(
            const colmap::BundleAdjustmentOptions& bundle_adjustment_options);

    // colmap::mvs
    void addPatchMatchStereoOptions(
            const colmap::mvs::PatchMatchOptions& patch_match_options);
    void addStereoFusionOptions(
            const colmap::mvs::StereoFusionOptions& stereo_fusion_options);
    void addPoissonMeshingOptions(
            const colmap::mvs::PoissonMeshingOptions& poisson_meshing_options);
    void addDelaunayMeshingOptions(const colmap::mvs::DelaunayMeshingOptions&
                                           delaunay_meshing_options);
    void addMeshPostProcessingOptions(
            const colmap::mvs::MeshPostProcessingOptions& options);

public:
    static void ReleaseOptions(int argc, char** argv) {
        if (argv) {
            for (int i = 0; i < argc; ++i) {
                if (argv[i]) {
                    delete[] argv[i];
                }
            }
            argv = nullptr;
        }
    }

    static void SetValue(const std::string& value, int argc_, char** argv_) {
        argv_[argc_] = new char[value.size() + 1];
        std::copy(value.begin(), value.end(), argv_[argc_]);
        argv_[argc_][value.size()] = '\0';  // don't forget the terminating 0
    }

private:
    inline void releaseOptions() {
        if (this->argc_ > 0 && this->argv_) {
            ReleaseOptions(this->argc_, this->argv_);
        }
        this->argc_ = 0;
    }

private:
    int argc_;
    char** argv_;
    // Serialized (name, value) pairs in registration order.
    std::vector<std::pair<std::string, std::string>> options_;
};

}  // namespace cloudViewer
