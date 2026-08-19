// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Dump every model-catalog download URL (one per line) to stdout.
// Feeds check_catalog_assets.py (test_catalog_remote_assets), which verifies
// that every catalog entry is actually published in the corresponding GitHub
// release. Without this check, a catalog/release drift (e.g. the historical
// rmbg_q4_K.gguf entry that 404'd) only surfaces at download time.

#include <cstdio>

#include "aicore/rfdetr_capi.h"
#include "aicore/rmbg_capi.h"

int main() {
    for (int i = 0; i < aicore_rmbg_model_count(); ++i) {
        const aicore_rmbg_model_entry* e = aicore_rmbg_model_at(i);
        if (e != nullptr && e->download_url != nullptr) {
            std::printf("%s\n", e->download_url);
        }
    }
    for (int i = 0; i < aicore_rfdetr_model_count(); ++i) {
        const aicore_rfdetr_model_entry* e = aicore_rfdetr_model_at(i);
        if (e != nullptr && e->download_url != nullptr) {
            std::printf("%s\n", e->download_url);
        }
    }
    return 0;
}
