// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// Local
#include <Eigen/Core>

#include "CV_db.h"

// Qt
#include <QColor>

// system
#include <CVConst.h>
#include <Logging.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <type_traits>

//! Default color components type (R,G and B)
using ColorCompType = unsigned char;

//! Colors namespace
namespace ecvColor {
//! Max value of a single color component (default type)
constexpr ColorCompType MAX = 255;
constexpr ColorCompType OPACITY = 255;

template <typename T1, typename T2>
struct is_same_type {
    operator bool() { return false; }
};

template <typename T1>
struct is_same_type<T1, T1> {
    operator bool() { return true; }
};

//! RGB color structure
template <typename Type>
class RgbTpl {
public:
    //! 3-tuple as a union
    union {
        struct {
            Type r, g, b;
        };
        Type rgb[3];
    };

    //! Default constructor
    /** Inits color to (0,0,0).
     **/
    constexpr inline RgbTpl() : r(0), g(0), b(0) {}

    //! Constructor from a triplet of r,g,b values
    explicit constexpr inline RgbTpl(Type red, Type green, Type blue)
        : r(red), g(green), b(blue) {}

    //! Constructor from an array of 3 values
    explicit constexpr inline RgbTpl(const Type col[3])
        : r(col[0]), g(col[1]), b(col[2]) {}

    inline static Eigen::Vector3d ToEigen(const Type col[3]) {
        if (is_same_type<Type, float>()) {
            return Eigen::Vector3d(col[0], col[1], col[2]);
        } else if (is_same_type<Type, ColorCompType>()) {
            return Eigen::Vector3d(col[0] / 255.0, col[1] / 255.0,
                                   col[2] / 255.0);
        } else {
            assert(false);
            return Eigen::Vector3d(col[0], col[1], col[2]);
        }
    }
    inline static Eigen::Vector3d ToEigen(const RgbTpl<Type>& t) {
        return ToEigen(t.rgb);
    }
    constexpr inline static RgbTpl FromEigen(const Eigen::Vector3d& t) {
        Type newCol[3];
        if (is_same_type<Type, float>()) {
            newCol[0] = static_cast<Type>(std::min(1.0, std::max(0.0, t(0))));
            newCol[1] = static_cast<Type>(std::min(1.0, std::max(0.0, t(1))));
            newCol[2] = static_cast<Type>(std::min(1.0, std::max(0.0, t(2))));
        } else {
            if (t(0) > 1 + EPSILON_VALUE || t(1) > 1 + EPSILON_VALUE ||
                t(2) > 1 + EPSILON_VALUE) {
                cloudViewer::utility::LogWarning(
                        "[ecvColor] Find invalid color: ");
                std::cout << t << std::endl;
            }

            newCol[0] = static_cast<Type>(
                    std::min(255.0, std::max(0.0, t(0) * MAX)));
            newCol[1] = static_cast<Type>(
                    std::min(255.0, std::max(0.0, t(1) * MAX)));
            newCol[2] = static_cast<Type>(
                    std::min(255.0, std::max(0.0, t(2) * MAX)));
        }

        return RgbTpl(newCol[0], newCol[1], newCol[2]);
    }

    //! Direct coordinate access
    constexpr inline Type& operator()(unsigned i) { return rgb[i]; }
    //! Direct coordinate access (const)
    constexpr inline Type& operator()(unsigned i) const { return rgb[i]; }

    //! In-place addition operator
    constexpr inline RgbTpl& operator+=(const RgbTpl<Type>& c) {
        r += c.r;
        g += c.g;
        b += c.b;
        return *this;
    }
    //! In-place subtraction operator
    constexpr inline RgbTpl& operator-=(const RgbTpl<Type>& c) {
        r -= c.r;
        g -= c.g;
        b -= c.b;
        return *this;
    }
    //! Comparison operator
    inline bool operator!=(const RgbTpl<Type>& t) const {
        return (r != t.r || g != t.g || b != t.b);
    }
};

//! 3 components, float type
using Rgbf = RgbTpl<float>;
//! 3 components, unsigned byte type
using Rgbub = RgbTpl<unsigned char>;
//! 3 components, default type
using Rgb = RgbTpl<ColorCompType>;

//! RGBA color structure
template <class Type>
class RgbaTpl {
public:
    // 4-tuple values as a union
    union {
        struct {
            Type r, g, b, a;
        };
        Type rgba[4];
    };

    //! Default constructor
    /** Inits color to (0,0,0,0).
     **/
    constexpr inline RgbaTpl() : r(0), g(0), b(0), a(0) {}

    //! Constructor from a triplet of r,g,b values and a transparency value
    explicit constexpr inline RgbaTpl(Type red,
                                      Type green,
                                      Type blue,
                                      Type alpha)
        : r(red), g(green), b(blue), a(alpha) {}

    //! RgbaTpl from an array of 4 values
    explicit constexpr inline RgbaTpl(const Type col[4])
        : r(col[0]), g(col[1]), b(col[2]), a(col[3]) {}
    //! RgbaTpl from an array of 3 values and a transparency value
    explicit constexpr inline RgbaTpl(const Type col[3], Type alpha)
        : r(col[0]), g(col[1]), b(col[2]), a(alpha) {}

    //! Copy constructor
    constexpr inline RgbaTpl(const RgbTpl<Type>& c, Type alpha)
        : r(c.r), g(c.g), b(c.b), a(alpha) {}

    //! Cast operator
    constexpr inline operator RgbTpl<Type>() const {
        return RgbTpl<Type>(rgba);
    }
    //! Cast operator (const version)
    // constexpr inline operator const Type*() const { return rgba; }

    //! Comparison operator
    inline bool operator!=(const RgbaTpl<Type>& t) const {
        return (r != t.r || g != t.g || b != t.b || a != t.a);
    }
};

//! 4 components, float type
using Rgbaf = RgbaTpl<float>;
//! 4 components, unsigned byte type
using Rgbaub = RgbaTpl<unsigned char>;
//! 4 components, default type
using Rgba = RgbaTpl<ColorCompType>;

// Predefined colors (default type)
constexpr Rgb white(MAX, MAX, MAX);
constexpr Rgb lightGrey(static_cast<ColorCompType>(MAX * 0.8),
                        static_cast<ColorCompType>(MAX * 0.8),
                        static_cast<ColorCompType>(MAX * 0.8));
constexpr Rgb darkGrey(MAX / 2, MAX / 2, MAX / 2);
constexpr Rgb red(MAX, 0, 0);
constexpr Rgb green(0, MAX, 0);
constexpr Rgb blue(0, 0, MAX);
constexpr Rgb darkBlue(0, 0, MAX / 2);
constexpr Rgb magenta(MAX, 0, MAX);
constexpr Rgb cyan(0, MAX, MAX);
constexpr Rgb orange(MAX, MAX / 2, 0);
constexpr Rgb black(0, 0, 0);
constexpr Rgb yellow(MAX, MAX, 0);

constexpr Rgba owhite(MAX, MAX, MAX, OPACITY);
constexpr Rgba olightGrey(static_cast<ColorCompType>(MAX * 0.8),
                          static_cast<ColorCompType>(MAX * 0.8),
                          static_cast<ColorCompType>(MAX * 0.8),
                          OPACITY);
constexpr Rgba odarkGrey(MAX / 2, MAX / 2, MAX / 2, OPACITY);
constexpr Rgba ored(MAX, 0, 0, OPACITY);
constexpr Rgba ogreen(0, MAX, 0, OPACITY);
constexpr Rgba oblue(0, 0, MAX, OPACITY);
constexpr Rgba odarkBlue(0, 0, MAX / 2, OPACITY);
constexpr Rgba omagenta(MAX, 0, MAX, OPACITY);
constexpr Rgba ocyan(0, MAX, MAX, OPACITY);
constexpr Rgba oorange(MAX, MAX / 2, 0, OPACITY);
constexpr Rgba oblack(0, 0, 0, OPACITY);
constexpr Rgba oyellow(MAX, MAX, 0, OPACITY);

// Predefined materials (float)
constexpr Rgbaf bright(1.00f, 1.00f, 1.00f, 1.00f);
constexpr Rgbaf lighter(0.83f, 0.83f, 0.83f, 1.00f);
constexpr Rgbaf light(0.66f, 0.66f, 0.66f, 1.00f);
constexpr Rgbaf middle(0.50f, 0.50f, 0.50f, 1.00f);
constexpr Rgbaf dark(0.34f, 0.34f, 0.34f, 1.00f);
constexpr Rgbaf darker(0.17f, 0.17f, 0.17f, 1.00f);
constexpr Rgbaf darkest(0.08f, 0.08f, 0.08f, 1.00f);
constexpr Rgbaf night(0.00f, 0.00f, 0.00f, 1.00F);
constexpr Rgbaf defaultMeshFrontDiff(0.00f, 0.90f, 0.27f, 1.00f);
constexpr Rgbaf defaultMeshBackDiff(0.27f, 0.90f, 0.90f, 1.00f);
constexpr Rgbf defaultViewBkgColor(10 / 255.0f, 102 / 255.0f, 151 / 255.0f);

// Default foreground color (unsigned byte)
// constexpr Rgbub defaultBkgColor		( 10, 102, 151); //dark blue
constexpr Rgbub defaultBkgColor(135, 206, 235);        // sky blue
constexpr Rgbub defaultColor(MAX, MAX, MAX);           // white
constexpr Rgbub defaultLabelBkgColor(MAX, MAX, MAX);   // white
constexpr Rgbub defaultLabelMarkerColor(MAX, 0, MAX);  // magenta

//! Colors generator
class Generator {
public:
    //! Generates a random color
    CV_DB_LIB_API static Rgb Random(bool lightOnly = true);
};

//! Color space conversion
class Convert {
public:
    //! Converts a HSL color to RGB color space
    /** \param H [out] hue [0;360[
            \param S [out] saturation [0;1]
            \param L [out] light [0;1]
            \return RGB color (unsigned byte)
    **/
    CV_DB_LIB_API static Rgb hsl2rgb(float H, float S, float L) {
        H /= 360;
        float q = L < 0.5f ? L * (1.0f + S) : L + S - L * S;
        float p = 2 * L - q;

        float r = hue2rgb(p, q, H + 1.0f / 3.0f);
        float g = hue2rgb(p, q, H);
        float b = hue2rgb(p, q, H - 1.0f / 3.0f);

        return Rgb(static_cast<ColorCompType>(r * ecvColor::MAX),
                   static_cast<ColorCompType>(g * ecvColor::MAX),
                   static_cast<ColorCompType>(b * ecvColor::MAX));
    }

    //! Converts a HSV color to RGB color space
    /** \param H [out] hue [0;360[
            \param S [out] saturation [0;1]
            \param V [out] value [0;1]
            \return RGB color (unsigned byte)
    **/
    CV_DB_LIB_API static Rgb hsv2rgb(float H, float S, float V) {
        float hi = 0;
        float f = std::modf(H / 60.0f, &hi);

        float l = V * (1.0f - S);
        float m = V * (1.0f - f * S);
        float n = V * (1.0f - (1.0f - f) * S);

        Rgbf rgb(0, 0, 0);

        switch (static_cast<int>(hi) % 6) {
            case 0:
                rgb.r = V;
                rgb.g = n;
                rgb.b = l;
                break;
            case 1:
                rgb.r = m;
                rgb.g = V;
                rgb.b = l;
                break;
            case 2:
                rgb.r = l;
                rgb.g = V;
                rgb.b = n;
                break;
            case 3:
                rgb.r = l;
                rgb.g = m;
                rgb.b = V;
                break;
            case 4:
                rgb.r = n;
                rgb.g = l;
                rgb.b = V;
                break;
            case 5:
                rgb.r = V;
                rgb.g = l;
                rgb.b = m;
                break;
        }

        return Rgb(static_cast<ColorCompType>(rgb.r * ecvColor::MAX),
                   static_cast<ColorCompType>(rgb.g * ecvColor::MAX),
                   static_cast<ColorCompType>(rgb.b * ecvColor::MAX));
    }

protected:
    //! Method used by hsl2rgb
    static float hue2rgb(float m1, float m2, float hue) {
        if (hue < 0)
            hue += 1.0f;
        else if (hue > 1.0f)
            hue -= 1.0f;

        if (6 * hue < 1.0f)
            return m1 + (m2 - m1) * hue * 6;
        else if (2 * hue < 1.0f)
            return m2;
        else if (3 * hue < 2.0f)
            return m1 + (m2 - m1) * (4.0f - hue * 6);
        else
            return m1;
    }
};

namespace LookUpTable {
CV_DB_LIB_API Rgb at(size_t color_id);
}

//! Conversion from Rgbf
inline Rgb FromRgbfToRgb(const Rgbf& color) {
    return Rgb(static_cast<ColorCompType>(color.r * MAX),
               static_cast<ColorCompType>(color.g * MAX),
               static_cast<ColorCompType>(color.b * MAX));
}

//! Conversion from Rgbaf
inline Rgb FromRgbafToRgb(const Rgbaf& color) {
    return Rgb(static_cast<ColorCompType>(color.r * MAX),
               static_cast<ColorCompType>(color.g * MAX),
               static_cast<ColorCompType>(color.b * MAX));
}
//! Conversion from Rgb to Rgba
inline Rgba FromRgbToRgba(const Rgb& color) { return Rgba(color, MAX); }

//! Conversion from Rgba to Rgb
inline Rgb FromRgbaToRgb(const Rgba& color) {
    return Rgb(color.r, color.g, color.b);
}

//! Conversion from Rgbaf to Rgba
inline Rgba FromRgbafToRgba(const Rgbaf& color) {
    return Rgba(static_cast<ColorCompType>(color.r * MAX),
                static_cast<ColorCompType>(color.g * MAX),
                static_cast<ColorCompType>(color.b * MAX),
                static_cast<ColorCompType>(color.a * MAX));
}

inline Rgbf FromRgb(const Rgb& color) {
    return Rgbf(static_cast<float>(1.0 * color.r / MAX),
                static_cast<float>(1.0 * color.g / MAX),
                static_cast<float>(1.0 * color.b / MAX));
}

inline Rgbf FromRgb(const Rgba& color) {
    return Rgbf(static_cast<float>(1.0 * color.r / MAX),
                static_cast<float>(1.0 * color.g / MAX),
                static_cast<float>(1.0 * color.b / MAX));
}

inline Rgbaf FromRgba(const Rgba& color) {
    return Rgbaf(static_cast<float>(1.0 * color.r / MAX),
                 static_cast<float>(1.0 * color.g / MAX),
                 static_cast<float>(1.0 * color.b / MAX),
                 static_cast<float>(1.0 * color.a / MAX));
}
inline Rgbaf FromRgbub(const Rgbub& color) {
    return Rgbaf(static_cast<float>(1.0 * color.r / MAX),
                 static_cast<float>(1.0 * color.g / MAX),
                 static_cast<float>(1.0 * color.b / MAX), 1.0f);
}

//! Conversion from QRgb
inline Rgb FromQRgb(QRgb qColor) {
    return Rgb(static_cast<unsigned char>(qRed(qColor)),
               static_cast<unsigned char>(qGreen(qColor)),
               static_cast<unsigned char>(qBlue(qColor)));
}

//! Conversion from QRgb'a'
inline Rgba FromQRgba(QRgb qColor) {
    return Rgba(static_cast<unsigned char>(qRed(qColor)),
                static_cast<unsigned char>(qGreen(qColor)),
                static_cast<unsigned char>(qBlue(qColor)),
                static_cast<unsigned char>(qAlpha(qColor)));
}

//! Conversion from QColor
inline Rgb FromQColor(QColor qColor) {
    return Rgb(static_cast<unsigned char>(qColor.red()),
               static_cast<unsigned char>(qColor.green()),
               static_cast<unsigned char>(qColor.blue()));
}

//! Conversion from QColor'a'
inline Rgba FromQColora(QColor qColor) {
    return Rgba(static_cast<unsigned char>(qColor.red()),
                static_cast<unsigned char>(qColor.green()),
                static_cast<unsigned char>(qColor.blue()),
                static_cast<unsigned char>(qColor.alpha()));
}

//! Conversion from QColor (floating point)
inline Rgbf FromQColorf(QColor qColor) {
    return Rgbf(static_cast<float>(qColor.redF()),
                static_cast<float>(qColor.greenF()),
                static_cast<float>(qColor.blueF()));
}
//! Conversion from QColor'a' (floating point)
inline Rgbaf FromQColoraf(QColor qColor) {
    return Rgbaf(static_cast<float>(qColor.redF()),
                 static_cast<float>(qColor.greenF()),
                 static_cast<float>(qColor.blueF()),
                 static_cast<float>(qColor.alphaF()));
}

};  // namespace ecvColor
