// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "aicore/depth_capi.h"
int main() {
    const char* gguf = std::getenv("DA_TEST_GGUF");
    if (!gguf) return 77;
    if (aicore_depth_abi_version() < 5) return 1;
    aicore_depth_ctx* c = aicore_depth_load(gguf, 1);
    if (!c) {
        std::fprintf(stderr, "load failed\n");
        return 1;
    }
    char* j = aicore_depth_info_json(c);
    bool ok = j && std::strstr(j, "embed_dim");
    std::fprintf(stderr, "info json: %s -> %s\n", j ? j : "(null)",
                 ok ? "OK" : "FAIL");
    aicore_depth_free_string(j);
    // Export wrappers (best-effort): exercise glb + colmap on the native
    // fixture.
    const char* png = std::getenv("DA_TEST_NATIVE_PNG");
    if (ok && png) {
        const char* glb = "/tmp/aicore_depth_export.glb";
        const char* col = "/tmp/aicore_depth_export_colmap";
        std::remove(glb);
        int rg = aicore_depth_export_glb(c, png, glb);
        int rc = aicore_depth_export_colmap(c, png, col, 1);
        FILE* f = std::fopen(glb, "rb");
        long sz = 0;
        if (f) {
            std::fseek(f, 0, SEEK_END);
            sz = std::ftell(f);
            std::fclose(f);
        }
        std::fprintf(stderr, "export glb=%d (%ld bytes) colmap=%d\n", rg, sz,
                     rc);
        if (rg != 0 || rc != 0 || sz <= 0) {
            ok = false;
        }
    }
    aicore_depth_free(c);
    return ok ? 0 : 1;
}
