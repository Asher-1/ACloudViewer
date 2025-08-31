// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "StaticVector.h"

#include <cstdio>

namespace cloudViewer {

int getArrayLengthFromFile(std::string fileName)
{
    FILE* f = fopen(fileName.c_str(), "rb");
    if(f == nullptr)
    {
        // printf("WARNING: file %s does not exists!\n", fileName.c_str());
        return 0;
    }

    int n = 0;
    size_t retval = fread(&n, sizeof(int), 1, f);
    if( retval != sizeof(int) )
    {
        CVLog::Warning("[IO] getArrayLengthFromFile: can't read array length (1)");
    }
    if(n == -1)
    {
        retval = fread(&n, sizeof(int), 1, f);
        if( retval != sizeof(int) )
        {
            CVLog::Warning("[IO] getArrayLengthFromFile: can't read array length (2)");
        }
    }
    fclose(f);
    return n;
}

} // namespace cloudViewer
