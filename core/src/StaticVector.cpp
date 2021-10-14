// ----------------------------------------------------------------------------
// -                        cloudViewer: asher-1.github.io                          -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 asher-1.github.io
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
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
