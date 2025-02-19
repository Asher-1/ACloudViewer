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

#pragma once

#include <memory>

#include "visualization/gui/Color.h"
namespace cloudViewer {
namespace visualization {
namespace gui {

// Label3D is a helper class for labels (like UI Labels) at 3D points as opposed
// to screen points. It is NOT a UI widget but is instead used via CloudViewerScene
// class. See CloudViewerScene::AddLabel/RemoveLabel.
class Label3D {
public:
    /// Copies text
    explicit Label3D(const Eigen::Vector3f& pos, const char* text = nullptr);
    ~Label3D();

    const char* GetText() const;
    /// Sets the text of the label (copies text)
    void SetText(const char* text);

    Eigen::Vector3f GetPosition() const;
    void SetPosition(const Eigen::Vector3f& pos);

    Color GetTextColor() const;
    void SetTextColor(const Color& color);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace gui
}  // namespace visualization
}  // namespace cloudViewer
