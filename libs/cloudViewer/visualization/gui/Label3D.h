// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <memory>

#include "visualization/gui/Color.h"
namespace cloudViewer {
namespace visualization {
namespace gui {

// Label3D is a helper class for labels (like UI Labels) at 3D points as opposed
// to screen points. It is NOT a UI widget but is instead used via
// CloudViewerScene class. See CloudViewerScene::AddLabel/RemoveLabel.
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
