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

class Checkbox : public Widget {
public:
    explicit Checkbox(const char* name);
    ~Checkbox();

    bool IsChecked() const;
    void SetChecked(bool checked);

    Size CalcPreferredSize(const LayoutContext& context,
                           const Constraints& constraints) const override;

    DrawResult Draw(const DrawContext& context) override;

    /// Specifies a callback function which will be called when the box
    /// changes checked state as a result of user action.
    void SetOnChecked(std::function<void(bool)> on_checked);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace gui
}  // namespace visualization
}  // namespace cloudViewer
