// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <cstdio>

int main(int argc, char **argv) {
    if (argc < 3) {
        printf("ShaderLinker:\n"
               "Unexpected number of arguments %d\n"
               "Expected usage \"ShaderLinker <output> <input1> [<input2> "
               "...]\"\n",
               argc);
        return 1;
    }

    FILE *file_out = fopen(argv[1], "w");
    if (file_out == 0) {
        printf("ShaderLinker:\n"
               "Cannot open file %s\n",
               argv[1]);
        return 1;
    }

    fprintf(file_out, "// Automatically generated header file for shader.\n");
    fprintf(file_out, "// See LICENSE.txt for full license statement.\n");
    fprintf(file_out, "\n");
    fprintf(file_out, "#pragma once\n");
    fprintf(file_out, "\n");
    fprintf(file_out, "namespace cloudViewer {\n");
    fprintf(file_out, "namespace visualization {\n");
    fprintf(file_out, "namespace glsl {\n");
    fprintf(file_out, "// clang-format off\n");
    fprintf(file_out, "\n");

    char buffer[1024];
    for (int i = 2; i < argc; ++i) {
        FILE *file_in = fopen(argv[i], "r");
        if (file_in == nullptr) {
            printf("ShaderLinker:\n"
                   "Cannot open file %s\n",
                   argv[i]);
            continue;
        }

        // Skip first 3 comment lines which only contain license information
        for (int i = 0; i < 3; ++i) {
            auto ignored = fgets(buffer, sizeof(buffer), file_in);
            (void)ignored;
        }

        // Copy content into "linked" file
        while (fgets(buffer, sizeof(buffer), file_in)) {
            fprintf(file_out, "%s", buffer);
        }
        fprintf(file_out, "\n");

        fclose(file_in);
    }

    fprintf(file_out, "// clang-format on\n");
    fprintf(file_out, "}  // namespace cloudViewer::glsl\n");
    fprintf(file_out, "}  // namespace cloudViewer::visualization\n");
    fprintf(file_out, "}  // namespace cloudViewer\n");
    fprintf(file_out, "\n");

    fclose(file_out);

    return 0;
}
