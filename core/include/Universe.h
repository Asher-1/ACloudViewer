// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
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
