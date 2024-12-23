// ----------------------------------------------------------------------------
// -                        CloudViewer: asher-1.github.io                    -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 asher-1.github.io
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

#include "visualization/gui/Font.h"

namespace cloudViewer {
namespace visualization {
namespace gui {

// assigned in header in constexpr declaration, but still need to be defined
constexpr const char *FontDescription::SANS_SERIF;
constexpr const char *FontDescription::MONOSPACE;

FontDescription::FontDescription(const char *typeface,
                                 FontStyle style /*= FontStyle::NORMAL*/,
                                 int point_size /*= 0*/) {
    ranges_.push_back({typeface, "en", {}});
    style_ = style;
    point_size_ = point_size;
}

void FontDescription::AddTypefaceForLanguage(const char *typeface,
                                             const char *lang) {
    ranges_.push_back({typeface, lang, {}});
}

void FontDescription::AddTypefaceForCodePoints(
        const char *typeface, const std::vector<uint32_t> &code_points) {
    ranges_.push_back({typeface, "", code_points});
}

}  // namespace gui
}  // namespace visualization
}  // namespace cloudViewer
