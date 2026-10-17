// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <Eigen/Core>
#include <utility>
#include <vector>

#include "util/types.h"

namespace colmap {

// Feature extractor types persisted in the database `descriptors.type`
// column. Values must stay in sync with upstream COLMAP (SIFT = 0).
// Upstream parity (d3ccaf35 feature/types.h): the enum is declared through
// MAKE_ENUM_CLASS_OVERLOAD_STREAM, which also generates
// FeatureExtractorTypeToString (consumed by retrieval/resources.cc).
MAKE_ENUM_CLASS_OVERLOAD_STREAM(FeatureExtractorType,
                                -1,
                                UNDEFINED,
                                SIFT,
                                ALIKED_N16ROT,
                                ALIKED_N32,
                                LOMA_B,
                                LOMA_B128);

// Upstream parity (d3ccaf35 feature/types.h): the matcher-type dispatch
// enum for the new-style FeatureMatcher abstraction. The fork resolves
// SIFT_*/LOMA_* on the SIFT/LoMa engines; ALIKED_* land with the AICore
// ggml task family.
MAKE_ENUM_CLASS_OVERLOAD_STREAM(FeatureMatcherType,
                                -1,
                                UNDEFINED,
                                SIFT_BRUTEFORCE,
                                SIFT_LIGHTGLUE,
                                ALIKED_BRUTEFORCE,
                                ALIKED_LIGHTGLUE,
                                LOMA_BRUTEFORCE,
                                LOMA_B,
                                LOMA_B128,
                                LOMA_R,
                                LOMA_L,
                                LOMA_G);

struct FeatureKeypoint {
    FeatureKeypoint();
    FeatureKeypoint(const float x, const float y);
    FeatureKeypoint(const float x,
                    const float y,
                    const float scale,
                    const float orientation);
    FeatureKeypoint(const float x,
                    const float y,
                    const float a11,
                    const float a12,
                    const float a21,
                    const float a22);

    static FeatureKeypoint FromParameters(const float x,
                                          const float y,
                                          const float scale_x,
                                          const float scale_y,
                                          const float orientation,
                                          const float shear);

    // Rescale the feature location and shape size by the given scale factor.
    void Rescale(const float scale);
    void Rescale(const float scale_x, const float scale_y);

    // Rotate the feature location and shape by k * 90 degrees counter-clockwise
    // around the image center. The width and height are the dimensions of the
    // image the keypoint is currently defined on.
    void Rot90(int k, int width, int height);

    // Compute similarity shape parameters from affine shape.
    float ComputeScale() const;
    float ComputeScaleX() const;
    float ComputeScaleY() const;
    float ComputeOrientation() const;
    float ComputeShear() const;

    // Location of the feature, with the origin at the upper left image corner,
    // i.e. the upper left pixel has the coordinate (0.5, 0.5).
    float x;
    float y;

    // Affine shape of the feature.
    float a11;
    float a12;
    float a21;
    float a22;

    // Upstream parity (d3ccaf35): value comparison for cache/database
    // round-trip assertions. Keypoints are compared exactly - they are
    // serialized losslessly as float32.
    friend bool operator==(const FeatureKeypoint& lhs,
                           const FeatureKeypoint& rhs) {
        return lhs.x == rhs.x && lhs.y == rhs.y && lhs.a11 == rhs.a11 &&
               lhs.a12 == rhs.a12 && lhs.a21 == rhs.a21 && lhs.a22 == rhs.a22;
    }
    friend bool operator!=(const FeatureKeypoint& lhs,
                           const FeatureKeypoint& rhs) {
        return !(lhs == rhs);
    }
};

typedef Eigen::Matrix<uint8_t, 1, Eigen::Dynamic, Eigen::RowMajor>
        FeatureDescriptor;

struct FeatureMatch {
    FeatureMatch()
        : point2D_idx1(kInvalidPoint2DIdx), point2D_idx2(kInvalidPoint2DIdx) {}
    FeatureMatch(const point2D_t point2D_idx1, const point2D_t point2D_idx2)
        : point2D_idx1(point2D_idx1), point2D_idx2(point2D_idx2) {}

    // Feature index in first image.
    point2D_t point2D_idx1 = kInvalidPoint2DIdx;

    // Feature index in second image.
    point2D_t point2D_idx2 = kInvalidPoint2DIdx;

    // Upstream COLMAP dbb41680 parity (feature/types.h): equality so that
    // std::vector<FeatureMatch> can be compared directly in tests.
    inline bool operator==(const FeatureMatch& other) const {
        return point2D_idx1 == other.point2D_idx1 &&
               point2D_idx2 == other.point2D_idx2;
    }

    inline bool operator!=(const FeatureMatch& other) const {
        return !(*this == other);
    }
};

using FeatureKeypoints = std::vector<FeatureKeypoint>;
using FeatureDescriptorsData =
        Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using FeatureDescriptorsFloatData =
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

struct FeatureDescriptorsFloat;

// The descriptor type and its bytes are one value. Storing them together
// prevents an extractor's float32 byte representation from being consumed as
// SIFT uint8 values after it crosses a database or matcher boundary.
struct FeatureDescriptors {
    FeatureDescriptors() = default;
    FeatureDescriptors(const FeatureExtractorType type,
                       FeatureDescriptorsData data)
        : type(type), data(std::move(data)) {}

    static FeatureDescriptors FromFloat(
            const FeatureDescriptorsFloat& float_descriptors);
    FeatureDescriptorsFloat ToFloat() const;

    FeatureExtractorType type = FeatureExtractorType::UNDEFINED;
    FeatureDescriptorsData data;
};

struct FeatureDescriptorsFloat {
    FeatureDescriptorsFloat() = default;
    FeatureDescriptorsFloat(const FeatureExtractorType type,
                            FeatureDescriptorsFloatData data)
        : type(type), data(std::move(data)) {}

    static FeatureDescriptorsFloat FromBytes(
            const FeatureDescriptors& byte_descriptors);
    FeatureDescriptors ToBytes() const;

    FeatureExtractorType type = FeatureExtractorType::UNDEFINED;
    FeatureDescriptorsFloatData data;
};

typedef std::vector<FeatureMatch> FeatureMatches;

}  // namespace colmap
