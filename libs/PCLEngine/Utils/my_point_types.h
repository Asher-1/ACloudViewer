// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

/**
 * @file my_point_types.h
 * @brief Custom PCL point type definitions for CloudViewer
 * 
 * Defines specialized PCL point types for efficient data extraction and
 * conversion between CloudViewer and PCL formats. These types are registered
 * with PCL's point type system for use in PCL algorithms.
 * 
 * The custom types support:
 * - RGB/RGBA color data only
 * - Intensity values
 * - Scalar fields (various numeric types)
 * - Normal vectors
 * - Combined point+color+scalar types
 */

#pragma once

// CV_CORE_LIB
#include <Eigen.h>

// PCL
#include <pcl/point_types.h>
#include <pcl/register_point_struct.h>
#include <stdint.h>

/**
 * @brief RGB color data only (no position)
 * 
 * Custom point type for extracting only RGB/RGBA color information.
 * Uses union for efficient access as float, uint32, or individual bytes.
 * Memory layout matches PCL's standard RGBA format.
 */
struct OnlyRGB {
    union {
        union {
            struct {
                std::uint8_t b;  ///< Blue component (0-255)
                std::uint8_t g;  ///< Green component (0-255)
                std::uint8_t r;  ///< Red component (0-255)
                std::uint8_t a;  ///< Alpha component (0-255)
            };
            float rgb;  ///< Packed RGB as float
        };
        std::uint32_t rgba;  ///< Packed RGBA as 32-bit integer
    };
};

/**
 * @brief Intensity value only
 * 
 * Custom point type for reading intensity scalar fields.
 */
struct PointI {
    float intensity;  ///< Intensity value
};

// =====================================================================
// Scalar Field Types (various numeric types)
// =====================================================================

/**
 * @brief Single-precision floating point scalar
 * @note Field name is obfuscated to avoid name conflicts
 */
struct FloatScalar {
    float S5c4laR;
};

/// Double-precision floating point scalar
struct DoubleScalar {
    double S5c4laR;
};

/// Signed integer scalar
struct IntScalar {
    int S5c4laR;
};

/// Unsigned integer scalar
struct UIntScalar {
    unsigned S5c4laR;
};

/// Signed short scalar
struct ShortScalar {
    short S5c4laR;
};

/// Unsigned short scalar
struct UShortScalar {
    unsigned short S5c4laR;
};

/// Signed 8-bit scalar
struct Int8Scalar {
    std::int8_t S5c4laR;
};

/// Unsigned 8-bit scalar
struct UInt8Scalar {
    std::uint8_t S5c4laR;
};

/**
 * @brief Normal vector components only (no position)
 * 
 * Custom point type for extracting only surface normal information.
 */
struct OnlyNormals {
    float normal_x;  ///< Normal X component
    float normal_y;  ///< Normal Y component
    float normal_z;  ///< Normal Z component
};

/**
 * @brief Normal vector with curvature
 * 
 * Combines normal vector with surface curvature estimate.
 * Uses PCL's standard normal4D layout (x, y, z, curvature).
 */
struct OnlyNormalsCurvature {
    PCL_ADD_NORMAL4D;  ///< Standard normal vector fields

    union {
        struct {
            float curvature;  ///< Surface curvature estimate
        };
        float data_c[4];  ///< Array access to curvature
    };
};

/**
 * @brief Point position with scalar value
 * 
 * Combines XYZ coordinates with a single scalar field value.
 */
struct PointXYZScalar {
    PCL_ADD_POINT4D;  ///< Standard point position (x, y, z, padding)
    float scalar;     ///< Scalar field value
};

/**
 * @brief Point position with scalar and RGB color
 * 
 * Combines XYZ coordinates, a scalar field, and RGB/RGBA color.
 * Useful for visualizing scalar fields with custom colormaps.
 */
struct PointXYZScalarRGB {
    PCL_ADD_POINT4D;  ///< Standard point position (x, y, z, padding)
    float scalar;     ///< Scalar field value
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
