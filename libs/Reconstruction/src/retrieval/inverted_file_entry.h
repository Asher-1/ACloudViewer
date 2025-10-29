// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <bitset>
#include <fstream>

#include "retrieval/geometry.h"

namespace colmap {
namespace retrieval {

// An inverted file entry. The template defines the dimensionality of the binary
// string used to approximate the descriptor in the Hamming space.
// This class is based on an original implementation by Torsten Sattler.
template <int N>
struct InvertedFileEntry {
    void Read(std::istream* ifs);
    void Write(std::ostream* ofs) const;

    // The identifier of the image this entry is associated with.
    int image_id = -1;

    // The index of the feature within the image's keypoints list.
    int feature_idx = -1;

    // The geometry of the feature, used for spatial verification.
    FeatureGeometry geometry;

    // The binary signature in the Hamming embedding.
    std::bitset<N> descriptor;
};

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

template <int N>
void InvertedFileEntry<N>::Read(std::istream* ifs) {
    static_assert(N <= 64, "Dimensionality too large");
    static_assert(sizeof(unsigned long long) >= 8,
                  "Expected unsigned long to be at least 8 byte");
    static_assert(sizeof(FeatureGeometry) == 16, "Geometry type size mismatch");

    int32_t image_id_data = 0;
    ifs->read(reinterpret_cast<char*>(&image_id_data), sizeof(int32_t));
    image_id = static_cast<int>(image_id_data);

    int32_t feature_idx_data = 0;
    ifs->read(reinterpret_cast<char*>(&feature_idx_data), sizeof(int32_t));
    feature_idx = static_cast<int>(feature_idx_data);

    ifs->read(reinterpret_cast<char*>(&geometry), sizeof(FeatureGeometry));

    uint64_t descriptor_data = 0;
    ifs->read(reinterpret_cast<char*>(&descriptor_data), sizeof(uint64_t));
    descriptor = std::bitset<N>(descriptor_data);
}

template <int N>
void InvertedFileEntry<N>::Write(std::ostream* ofs) const {
    static_assert(N <= 64, "Dimensionality too large");
    static_assert(sizeof(unsigned long long) >= 8,
                  "Expected unsigned long to be at least 8 byte");
    static_assert(sizeof(FeatureGeometry) == 16, "Geometry type size mismatch");

    const int32_t image_id_data = image_id;
    ofs->write(reinterpret_cast<const char*>(&image_id_data), sizeof(int32_t));

    const int32_t feature_idx_data = feature_idx;
    ofs->write(reinterpret_cast<const char*>(&feature_idx_data),
               sizeof(int32_t));

    ofs->write(reinterpret_cast<const char*>(&geometry),
               sizeof(FeatureGeometry));

    const uint64_t descriptor_data =
            static_cast<uint64_t>(descriptor.to_ullong());
    ofs->write(reinterpret_cast<const char*>(&descriptor_data),
               sizeof(uint64_t));
}

}  // namespace retrieval
}  // namespace colmap
