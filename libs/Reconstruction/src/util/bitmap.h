// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <algorithm>
#include <cmath>
#include <ios>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "util/string.h"

namespace colmap {

// Templated bitmap color class.
template <typename T>
struct BitmapColor {
    BitmapColor();
    BitmapColor(const T gray);
    BitmapColor(const T r, const T g, const T b);

    template <typename D>
    BitmapColor<D> Cast() const;

    bool operator==(const BitmapColor<T>& rhs) const;
    bool operator!=(const BitmapColor<T>& rhs) const;

    template <typename D>
    friend std::ostream& operator<<(std::ostream& output,
                                    const BitmapColor<D>& color);

    T r;
    T g;
    T b;
};

// Cross-platform bitmap backed by OpenImageIO and tightly packed UINT8 data.
enum class BitmapFormat { kUnknown, kPng, kJpeg, kTiff };
enum class BitmapRescaleFilter { kBilinear, kBox };
enum class BitmapMetadataModel { kMain, kExif, kGps };

class Bitmap {
public:
    Bitmap();
    ~Bitmap();

    // Copy constructor.
    Bitmap(const Bitmap& other);
    // Move constructor.
    Bitmap(Bitmap&& other) noexcept;

    // Copy assignment.
    Bitmap& operator=(const Bitmap& other);
    // Move assignment.
    Bitmap& operator=(Bitmap&& other) noexcept;

    // Allocate bitmap by overwriting the existing data.
    bool Allocate(const int width, const int height, const bool as_rgb);

    // Deallocate the bitmap by releasing the existing data.
    void Deallocate();

    // Opaque pointer to the owned storage, retained for legacy null checks.
    const void* Data() const;
    void* Data();

    // Dimensions of bitmap.
    int Width() const;
    int Height() const;
    int Channels() const;

    // Number of bits per pixel. This is 8 for grey and 24 for RGB image.
    unsigned int BitsPerPixel() const;

    // Scan width in bytes. Bitmap storage is tightly packed row-major data.
    unsigned int ScanWidth() const;

    // Check whether image is grey- or colorscale.
    bool IsRGB() const;
    bool IsGrey() const;

    // Number of bytes required to store image.
    size_t NumBytes() const;

    // Copy raw image data to array.
    std::vector<uint8_t> ConvertToRawBits() const;
    std::vector<uint8_t> ConvertToRowMajorArray() const;
    std::vector<uint8_t> ConvertToColMajorArray() const;

    // Manipulate individual pixels. For grayscale images, only the red element
    // of the RGB color is used.
    bool GetPixel(const int x, const int y, BitmapColor<uint8_t>* color) const;
    bool SetPixel(const int x, const int y, const BitmapColor<uint8_t>& color);

    // Get pointer to y-th scanline, where the 0-th scanline is at the top.
    const uint8_t* GetScanline(const int y) const;

    // Fill entire bitmap with uniform color. For grayscale images, the first
    // element of the vector is used.
    void Fill(const BitmapColor<uint8_t>& color);

    // Interpolate color at given floating point position.
    bool InterpolateNearestNeighbor(const double x,
                                    const double y,
                                    BitmapColor<uint8_t>* color) const;
    bool InterpolateBilinear(const double x,
                             const double y,
                             BitmapColor<float>* color) const;

    // Extract EXIF information from bitmap. Returns false if no EXIF
    // information is embedded in the bitmap.
    bool ExifCameraModel(std::string* camera_model) const;
    bool ExifFocalLength(double* focal_length) const;
    bool ExifLatitude(double* latitude) const;
    bool ExifLongitude(double* longitude) const;
    bool ExifAltitude(double* altitude) const;

    // Read bitmap at given path and convert to grey- or colorscale.
    bool Read(const std::string& path, const bool as_rgb = true);

    // Write image to file. For JPEG, flags is the requested quality [1, 100].
    bool Write(const std::string& path,
               const BitmapFormat format = BitmapFormat::kUnknown,
               const int flags = 0) const;

    // Smooth the image using a Gaussian kernel.
    void Smooth(const float sigma_x, const float sigma_y);

    // Rescale image to the new dimensions.
    void Rescale(const int new_width,
                 const int new_height,
                 const BitmapRescaleFilter filter = BitmapRescaleFilter::kBilinear);

    // Clone the image to a new bitmap object.
    Bitmap Clone() const;
    Bitmap CloneAsGrey() const;
    Bitmap CloneAsRGB() const;

    // Clone metadata from this bitmap object to another target bitmap object.
    void CloneMetadata(Bitmap* target) const;

    // Read specific EXIF tag.
    bool ReadExifTag(const BitmapMetadataModel model,
                     const std::string& tag_name,
                     std::string* result) const;

private:
    struct Storage;
    std::unique_ptr<Storage> data_;
    int width_;
    int height_;
    int channels_;
};

// Jet colormap inspired by Matlab. Grayvalues are expected in the range [0, 1]
// and are converted to RGB values in the same range.
class JetColormap {
public:
    static float Red(const float gray);
    static float Green(const float gray);
    static float Blue(const float gray);

private:
    static float Interpolate(const float val,
                             const float y0,
                             const float x0,
                             const float y1,
                             const float x1);
    static float Base(const float val);
};

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

namespace internal {

template <typename T1, typename T2>
T2 BitmapColorCast(const T1 value) {
    return std::min(static_cast<T1>(std::numeric_limits<T2>::max()),
                    std::max(static_cast<T1>(std::numeric_limits<T2>::min()),
                             std::round(value)));
}

}  // namespace internal

template <typename T>
BitmapColor<T>::BitmapColor() : r(0), g(0), b(0) {}

template <typename T>
BitmapColor<T>::BitmapColor(const T gray) : r(gray), g(gray), b(gray) {}

template <typename T>
BitmapColor<T>::BitmapColor(const T r, const T g, const T b)
    : r(r), g(g), b(b) {}

template <typename T>
template <typename D>
BitmapColor<D> BitmapColor<T>::Cast() const {
    BitmapColor<D> color;
    color.r = internal::BitmapColorCast<T, D>(r);
    color.g = internal::BitmapColorCast<T, D>(g);
    color.b = internal::BitmapColorCast<T, D>(b);
    return color;
}

template <typename T>
bool BitmapColor<T>::operator==(const BitmapColor<T>& rhs) const {
    return r == rhs.r && g == rhs.g && b == rhs.b;
}

template <typename T>
bool BitmapColor<T>::operator!=(const BitmapColor<T>& rhs) const {
    return r != rhs.r || g != rhs.g || b != rhs.b;
}

template <typename T>
std::ostream& operator<<(std::ostream& output, const BitmapColor<T>& color) {
    output << StringPrintf("RGB(%f, %f, %f)", static_cast<double>(color.r),
                           static_cast<double>(color.g),
                           static_cast<double>(color.b));
    return output;
}

}  // namespace colmap
