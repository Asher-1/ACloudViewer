// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvAdvancedTypes.h"

NormsIndexesTableType::NormsIndexesTableType()
    : ccArray<CompressedNormType, 1, CompressedNormType>("Compressed normals") {
}

bool NormsIndexesTableType::fromFile_MeOnly(QFile& in,
                                            short dataVersion,
                                            int flags,
                                            LoadedIDMap& oldToNewIDMap) {
    if (dataVersion < 41) {
        // in previous versions (< 41) the normals were compressed on 15 bytes
        // (2*6+3) as unsigned short
        static const unsigned OLD_QUANTIZE_LEVEL = 6;

        ccArray<unsigned short, 1, unsigned short>* oldNormals =
                new ccArray<unsigned short, 1, unsigned short>();
        if (!ccSerializationHelper::GenericArrayFromFile<unsigned short, 1,
                                                         unsigned short>(
                    *oldNormals, in, dataVersion, "old normals")) {
            oldNormals->release();
            return false;
        }

        bool success = false;
        try {
            resize(oldNormals->size());
        } catch (const std::bad_alloc&) {
            oldNormals->release();
            return false;
        }

        // convert old normals to new ones
        for (size_t i = 0; i < oldNormals->size(); ++i) {
            CCVector3 N;
            // decompress (with the old parameters)
            {
                unsigned short n = oldNormals->at(i);
                ccNormalCompressor::Decompress(n, N.u, OLD_QUANTIZE_LEVEL);
            }
            // and recompress
            CompressedNormType index = static_cast<CompressedNormType>(
                    ccNormalCompressor::Compress(N.u));
            at(i) = index;
        }

        oldNormals->release();
        return true;
    } else {
        return ccSerializationHelper::GenericArrayFromFile<
                CompressedNormType, 1, CompressedNormType>(
                *this, in, dataVersion, "compressed normals");
    }
}
