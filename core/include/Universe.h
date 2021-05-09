// ----------------------------------------------------------------------------
// -                        cloudViewer: www.erow.cn                          -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.erow.cn
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

#pragma once

// LOCAL
#include "CVCoreLib.h"

namespace cloudViewer {

typedef struct
{
    int rank;
    int p;
    int size;
} uni_elt;

/**
 * @brief Allows to perform labelling by creating node and connecting them.
 */
class CV_CORE_LIB_API Universe
{
public:
    explicit Universe(int elements);
    ~Universe();
    /// Initialize all elements to the default values
    void initialize();
    /// Retrieve the smallest index of the elements connected to x.
    /// @warning: it updates the local indexes along the way
    int find(int x);
    void join(int x, int y);
    void addEdge(int x, int y);

    inline int size(int x) const
    {
        return elts[x].size;
    }

public:
    uni_elt* elts;
    int num, allelems;
};

} // namespace cloudViewer
