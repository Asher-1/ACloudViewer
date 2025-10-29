// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "base/image_reader.h"
#include "controllers/incremental_mapper.h"
#include "feature/extraction.h"
#include "feature/matching.h"
#include "feature/sift.h"
#include "mvs/fusion.h"
#include "mvs/meshing.h"
#include "mvs/patch_match.h"
#include "optim/bundle_adjustment.h"

namespace cloudViewer {

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

    template <typename T>
    void registerOption(const std::string& name, const T* option) {
        if (std::is_same<T, bool>::value) {
            options_bool_.emplace_back(name,
                                       reinterpret_cast<const bool*>(option));
        } else if (std::is_same<T, int>::value) {
            options_int_.emplace_back(name,
                                      reinterpret_cast<const int*>(option));
        } else if (std::is_same<T, double>::value) {
            options_double_.emplace_back(
                    name, reinterpret_cast<const double*>(option));
        } else if (std::is_same<T, std::string>::value) {
            if (!reinterpret_cast<const std::string*>(option)->empty()) {
                options_string_.emplace_back(
                        name, reinterpret_cast<const std::string*>(option));
            }
        } else {
            std::cerr << "Unsupported option type" << std::endl;
        }
    }

    // colmap
    void addExtractionOptions(
            const colmap::ImageReaderOptions& image_reader_options,
            const colmap::SiftExtractionOptions& sift_extraction_options);

    void addMatchingOptions(
            const colmap::SiftMatchingOptions& sift_matching_options);

    void addExhaustiveMatchingOptions(
            const colmap::SiftMatchingOptions& sift_matching_options,
            const colmap::ExhaustiveMatchingOptions&
                    exhaustive_matching_options);

    void addSequentialMatchingOptions(
            const colmap::SiftMatchingOptions& sift_matching_options,
            const colmap::SequentialMatchingOptions&
                    sequential_matching_options);

    void addSpatialMatchingOptions(
            const colmap::SiftMatchingOptions& sift_matching_options,
            const colmap::SpatialMatchingOptions& spatial_matching_options);

    void addTransitiveMatchingOptions(
            const colmap::SiftMatchingOptions& sift_matching_options,
            const colmap::TransitiveMatchingOptions&
                    transitive_matching_options);

    void addVocabTreeMatchingOptions(
            const colmap::SiftMatchingOptions& sift_matching_options,
            const colmap::VocabTreeMatchingOptions&
                    vocab_tree_matching_options);

    void addImagePairsMatchingOptions(
            const colmap::SiftMatchingOptions& sift_matching_options,
            const colmap::ImagePairsMatchingOptions&
                    image_pairs_matching_options);

    void addMapperOptions(
            const colmap::IncrementalMapperOptions& incremental_mapper_options);

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

    inline unsigned long getParametersCount() {
        return static_cast<unsigned long>(
                options_int_.size() + options_bool_.size() +
                options_double_.size() + options_string_.size());
    }

private:
    int argc_;
    char** argv_;
    std::vector<std::pair<std::string, const bool*>> options_bool_;
    std::vector<std::pair<std::string, const int*>> options_int_;
    std::vector<std::pair<std::string, const double*>> options_double_;
    std::vector<std::pair<std::string, const std::string*>> options_string_;
};

}  // namespace cloudViewer
