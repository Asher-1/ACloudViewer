// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <string>

namespace colmap {
namespace retrieval {

#ifdef COLMAP_DOWNLOAD_ENABLED
// Default vocabulary tree URI in format: "<url>;<name>;<sha256>"
// This is a placeholder - replace with actual URL and SHA256 when available
const static std::string kDefaultVocabTreeUri =
        "https://github.com/Asher-1/cloudViewer_downloads/releases/download/"
        "vocab_tree/"
        "vocab_tree_faiss_flickr100K_words256K.bin;"
        "vocab_tree_faiss_flickr100K_words256K.bin;"
        "96ca8ec8ea60b1f73465aaf2c401fd3b3ca75cdba2d3c50d6a2f6f760f275ddc";
#else
const static std::string kDefaultVocabTreeUri = "";
#endif

}  // namespace retrieval
}  // namespace colmap
