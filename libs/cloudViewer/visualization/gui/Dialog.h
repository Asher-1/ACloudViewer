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

class Window;

/// Base class for dialogs.
class Dialog : public Widget {
    using Super = Widget;

public:
    explicit Dialog(const char* title);
    virtual ~Dialog();

    Size CalcPreferredSize(const LayoutContext& context,
                           const Constraints& constraints) const override;
    void Layout(const LayoutContext& context) override;
    DrawResult Draw(const DrawContext& context) override;

    virtual void OnWillShow();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace gui
}  // namespace visualization
}  // namespace cloudViewer
