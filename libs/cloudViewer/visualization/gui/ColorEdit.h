// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <functional>

#include "visualization/gui/Widget.h"

namespace cloudViewer {
namespace visualization {
namespace gui {

class ColorEdit : public Widget {
public:
    ColorEdit();
    ~ColorEdit() override;

    void SetValue(const Color& color);
    void SetValue(float r, float g, float b);

    const Color& GetValue() const;

    Size CalcPreferredSize(const LayoutContext& context,
                           const Constraints& constraints) const override;

    DrawResult Draw(const DrawContext& context) override;

    /// Specifies a callback function which will be called when the value
    /// changes as a result of user action.
    void SetOnValueChanged(std::function<void(const Color&)> on_value_changed);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace gui
}  // namespace visualization
}  // namespace cloudViewer
