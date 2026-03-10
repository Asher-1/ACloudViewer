// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/** @file my_point_types.h
 *  @brief Custom PCL point types for scalar fields, RGB, normals, and combined
 * attributes
 */

// CV_CORE_LIB
#include <Eigen.h>

// PCL
#include <pcl/point_types.h>
#include <pcl/register_point_struct.h>
#include <stdint.h>

//! PCL custom point type used for reading RGB data
struct OnlyRGB {
    union {
        union {
            struct {
                std::uint8_t b;
                std::uint8_t g;
                std::uint8_t r;
                std::uint8_t a;
            };
            float rgb;
        };
        std::uint32_t rgba;
    };
};

//! PCL custom point type used for reading intensity data
struct PointI {
    float intensity;
};

/** @struct FloatScalar
 *  @brief PCL point type with a single float scalar field
 */
struct FloatScalar {
    float S5c4laR;
};

/** @struct DoubleScalar
 *  @brief PCL point type with a single double scalar field
 */
struct DoubleScalar {
    double S5c4laR;
};

/** @struct IntScalar
 *  @brief PCL point type with a single int scalar field
 */
struct IntScalar {
    int S5c4laR;
};

/** @struct UIntScalar
 *  @brief PCL point type with a single unsigned int scalar field
 */
struct UIntScalar {
    unsigned S5c4laR;
};

struct ShortScalar {
    short S5c4laR;
};

struct UShortScalar {
    unsigned short S5c4laR;
};

/** @struct Int8Scalar
 *  @brief PCL point type with a single int8 scalar field
 */
struct Int8Scalar {
    std::int8_t S5c4laR;
};

/** @struct UInt8Scalar
 *  @brief PCL point type with a single uint8 scalar field
 */
struct UInt8Scalar {
    std::uint8_t S5c4laR;
};

//! PCL custom point type used for reading intensity data
struct OnlyNormals {
    float normal_x;
    float normal_y;
    float normal_z;
};

/** @struct OnlyNormalsCurvature
 *  @brief PCL point type with normals and curvature
 */
struct OnlyNormalsCurvature {
    PCL_ADD_NORMAL4D;

    union {
        struct {
            float curvature;
        };
        float data_c[4];
    };
};

/** @struct PointXYZScalar
 *  @brief PCL point type with XYZ coordinates and a scalar field
 */
struct PointXYZScalar {
    PCL_ADD_POINT4D;
    float scalar;
};

/** @struct PointXYZScalarRGB
 *  @brief PCL point type with XYZ, scalar, and RGB fields
 */
struct PointXYZScalarRGB {
    PCL_ADD_POINT4D;
    float scalar;
    union {
        union {
            struct {
                std::uint8_t b;
                std::uint8_t g;
                std::uint8_t r;
                std::uint8_t _unused;
            };
            float rgb;
        };
        std::uint32_t rgba;
    };
};

/** @struct PointXYZScalarRGBNormals
 *  @brief PCL point type with XYZ, scalar, RGB, and normals
 */
struct PointXYZScalarRGBNormals {
    PCL_ADD_NORMAL4D;
    // PCL_ADD_RGB;
    union {
        union {
            struct {
                std::uint8_t b;
                std::uint8_t g;
                std::uint8_t r;
                std::uint8_t a;
            };
            float rgb;
        };
        uint32_t rgba;
    };
    PCL_ADD_POINT4D;
    float curvature;
    float scalar;
};

POINT_CLOUD_REGISTER_POINT_STRUCT(OnlyRGB, (float, rgb, rgb))

POINT_CLOUD_REGISTER_POINT_STRUCT(PointI, (float, intensity, intensity))

POINT_CLOUD_REGISTER_POINT_STRUCT(FloatScalar, (float, S5c4laR, S5c4laR))

POINT_CLOUD_REGISTER_POINT_STRUCT(DoubleScalar, (double, S5c4laR, S5c4laR))

POINT_CLOUD_REGISTER_POINT_STRUCT(IntScalar, (int, S5c4laR, S5c4laR))

POINT_CLOUD_REGISTER_POINT_STRUCT(UIntScalar, (unsigned int, S5c4laR, S5c4laR))

POINT_CLOUD_REGISTER_POINT_STRUCT(ShortScalar, (short, S5c4laR, S5c4laR))

POINT_CLOUD_REGISTER_POINT_STRUCT(UShortScalar,
                                  (unsigned short, S5c4laR, S5c4laR))

POINT_CLOUD_REGISTER_POINT_STRUCT(Int8Scalar, (std::int8_t, S5c4laR, S5c4laR))

POINT_CLOUD_REGISTER_POINT_STRUCT(UInt8Scalar, (std::uint8_t, S5c4laR, S5c4laR))

POINT_CLOUD_REGISTER_POINT_STRUCT(
        OnlyNormals,
        (float, normal_x, normal_x)(float, normal_y, normal_y)(float,
                                                               normal_z,
                                                               normal_z))

POINT_CLOUD_REGISTER_POINT_STRUCT(
        OnlyNormalsCurvature,
        (float, normal_x, normal_x)(float, normal_y, normal_y)(
                float, normal_z, normal_z)(float, curvature, curvature))

POINT_CLOUD_REGISTER_POINT_STRUCT(
        PointXYZScalar,
        (float, x, x)(float, y, y)(float, z, z)(float, scalar, scalar))

POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZScalarRGB,
                                  (float, rgb, rgb)(float, x, x)(float, y, y)(
                                          float, z, z)(float, scalar, scalar))

POINT_CLOUD_REGISTER_POINT_STRUCT(
        PointXYZScalarRGBNormals,
        (float, rgb, rgb)(float, x, x)(float, y, y)(float, z, z)(
                float, scalar, scalar)(float, normal_x, normal_x)(float,
                                                                  normal_y,
                                                                  normal_y)(
                float, normal_z, normal_z)(float, curvature, curvature))
