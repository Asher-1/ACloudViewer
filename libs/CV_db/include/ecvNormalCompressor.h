// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "CV_db.h"

// Local
#include "ecvBasicTypes.h"

//! Normal compressor
class CV_DB_LIB_API ccNormalCompressor {
public:
    //! Compressed normals quantization level (number of directions: 2^(2*N+3))
    /** \warning Never pass a 'constant initializer' by reference
     **/
    static const unsigned char QUANTIZE_LEVEL =
            9;  // 2097152 normals * 12 bytes = 24 Mb of memory

    //! Last valid normal code
    static const unsigned MAX_VALID_NORM_CODE =
            (1 << (QUANTIZE_LEVEL * 2 + 3)) - 1;
    //! Null normal code
    static const unsigned NULL_NORM_CODE = MAX_VALID_NORM_CODE + 1;

    //! Compression algorithm
    static unsigned Compress(const PointCoordinateType N[3]);

    //! Decompression algorithm
    static void Decompress(unsigned index,
                           PointCoordinateType N[3],
                           unsigned char level = QUANTIZE_LEVEL);

    //! Inverts a (compressed) normal
    static void InvertNormal(CompressedNormType &code);
};
