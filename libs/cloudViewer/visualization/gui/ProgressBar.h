// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "visualization/gui/Widget.h"

namespace cloudViewer {
namespace visualization {
namespace gui {

class ProgressBar : public Widget {
public:
    ProgressBar();
    ~ProgressBar();

    /// ProgressBar values ranges from 0.0 (incomplete) to 1.0 (complete)
    void SetValue(float value);
    float GetValue() const;

    Size CalcPreferredSize(const LayoutContext& context,
                           const Constraints& constraints) const override;

    Widget::DrawResult Draw(const DrawContext& context) override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace gui
}  // namespace visualization
}  // namespace cloudViewer
