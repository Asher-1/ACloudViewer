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

class TabControl : public Widget {
    using Super = Widget;

public:
    TabControl();
    ~TabControl();

    void AddTab(const char* name, std::shared_ptr<Widget> panel);

    Size CalcPreferredSize(const LayoutContext& context,
                           const Constraints& constraints) const override;
    void Layout(const LayoutContext& context) override;

    DrawResult Draw(const DrawContext& context) override;

    void SetOnSelectedTabChanged(std::function<void(int)> on_changed);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace gui
}  // namespace visualization
}  // namespace cloudViewer
