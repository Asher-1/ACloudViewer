// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "aicore/aliked_capi.h"
#include "aicore/deeplsd_capi.h"

int main(int argc, char** argv) {
    if (argc != 5) {
        std::fprintf(
                stderr,
                "usage: %s aliked|deeplsd input.gguf output.gguf f16|q8_0\n",
                argv[0]);
        return 2;
    }
    const char* module = argv[1];
    const char* input = argv[2];
    const char* output = argv[3];
    const char* type = argv[4];
    int rc = -1;
    if (std::strcmp(module, "deeplsd") == 0) {
        rc = aicore_deeplsd_quantize(input, output, type);
    } else if (std::strcmp(module, "aliked") == 0) {
        rc = aicore_aliked_quantize_gguf(input, output, type);
    } else {
        std::fprintf(stderr, "unknown module: %s (want aliked or deeplsd)\n",
                     module);
        return 2;
    }
    if (rc != 0) {
        std::fprintf(stderr, "quantize failed for module=%s\n", module);
        return 1;
    }
    std::fprintf(stderr, "quantized %s -> %s (%s)\n", input, output, type);
    return 0;
}
