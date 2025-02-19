// ----------------------------------------------------------------------------
// -                        CloudViewer: asher-1.github.io                          -
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

#include "t/geometry/TensorMap.h"

#include <fmt/format.h>

#include <sstream>
#include <string>
#include <unordered_map>

#include <Logging.h>

namespace cloudViewer {
namespace t {
namespace geometry {

bool TensorMap::IsSizeSynchronized() const {
    const int64_t primary_size = GetPrimarySize();
    for (auto& kv : *this) {
        if (kv.second.GetLength() != primary_size) {
            return false;
        }
    }
    return true;
}

void TensorMap::AssertPrimaryKeyInMapOrEmpty() const {
    if (this->size() != 0 && this->count(primary_key_) == 0) {
        utility::LogError("TensorMap does not contain primary key \"{}\".",
                          primary_key_);
    }
}

void TensorMap::AssertSizeSynchronized() const {
    if (!IsSizeSynchronized()) {
        const int64_t primary_size = GetPrimarySize();
        std::stringstream ss;
        ss << fmt::format("Primary Tensor \"{}\" has size {}, however: \n",
                          primary_key_, primary_size);
        for (auto& kv : *this) {
            if (kv.first != primary_key_ &&
                kv.second.GetLength() != primary_size) {
                fmt::format("    > Tensor \"{}\" has size {}.\n", kv.first,
                            kv.second.GetLength());
            }
        }
        utility::LogError("{}", ss.str());
    }
}

}  // namespace geometry
}  // namespace t
}  // namespace cloudViewer
