// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <cstdio>
#include <cstring>
#include <set>
#include <string>

#include "aicore/model_catalog_capi.h"

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        ++failures;
    }
}

}  // namespace

int main() {
    const aicore_model_family families[] = {
            AICORE_MODEL_FAMILY_DEPTH, AICORE_MODEL_FAMILY_DEEPLSD,
            AICORE_MODEL_FAMILY_LIGHTGLUE, AICORE_MODEL_FAMILY_GAUSSIAN,
            AICORE_MODEL_FAMILY_ALIKED};
    for (aicore_model_family family : families) {
        const int count = aicore_model_count(family);
        expect(count > 0, "catalog family must not be empty");
        std::set<std::string> filenames;
        for (int i = 0; i < count; ++i) {
            const aicore_model_entry* entry = aicore_model_at(family, i);
            expect(entry != nullptr, "in-range catalog lookup must succeed");
            if (!entry) continue;
            expect(entry->filename && *entry->filename,
                   "catalog filename must be non-empty");
            expect(entry->display_name && *entry->display_name,
                   "catalog display name must be non-empty");
            expect(entry->download_url && std::strncmp(entry->download_url,
                                                       "https://", 8) == 0,
                   "catalog URL must use HTTPS");
            expect(entry->sha256 && std::strlen(entry->sha256) == 64,
                   "published catalog row must expose a pinned SHA-256");
            if (entry->filename) {
                expect(filenames.insert(entry->filename).second,
                       "catalog filenames must be unique within a family");
            }
        }
        expect(aicore_model_at(family, -1) == nullptr,
               "negative catalog index must fail");
        expect(aicore_model_at(family, count) == nullptr,
               "past-end catalog index must fail");
        const aicore_model_entry* first = aicore_model_at(family, 0);
        const std::string firstFilename = first ? first->filename : "";
        const aicore_model_entry* found =
                aicore_model_by_filename(family, firstFilename.c_str());
        expect(found && firstFilename == found->filename,
               "filename lookup must return the requested row");
    }
    expect(aicore_model_by_filename(AICORE_MODEL_FAMILY_DEPTH,
                                    "missing.gguf") == nullptr,
           "unknown filename must fail");
    if (failures == 0) {
        std::printf("test_shared_model_catalog: all checks passed\n");
        return 0;
    }
    return 1;
}
