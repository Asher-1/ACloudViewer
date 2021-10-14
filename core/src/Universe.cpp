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
#include "Universe.h"

namespace cloudViewer {

Universe::Universe(int elements)
{
    elts = new uni_elt[static_cast<unsigned long>(elements)];
    allelems = elements;
    num = elements;
    initialize();
}

void Universe::initialize()
{
    num = allelems;
    for(int i = 0; i < allelems; i++)
    {
        elts[i].rank = 0;
        elts[i].size = 1;
        elts[i].p = i; // initialized to the index
    }
}

Universe::~Universe()
{
    delete[] elts;
}

int Universe::find(int x)
{
    int y = x;
    while(y != elts[y].p) // follow the index stored in p if not the same that the index
        y = elts[y].p;
    elts[x].p = y; // update x element to the final value (instead of keeping multiple indirections), so next time we will access it directly.
    return y;
}

void Universe::join(int x, int y)
{
    // join elements in the one with the highest rank
    if(elts[x].rank > elts[y].rank)
    {
        elts[y].p = x;
        elts[x].size += elts[y].size;
    }
    else
    {
        elts[x].p = y;
        elts[y].size += elts[x].size;
        if(elts[x].rank == elts[y].rank)
            elts[y].rank++;
    }
    num--; // the number of elements has been reduced by one
}

void Universe::addEdge(int x, int y)
{
    int a = find(x);
    int b = find(y);
    if(a != b)
    {
        join(a, b);
    }
}

} // namespace cloudViewer
