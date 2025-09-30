// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "visualization/gui/Button.h"

namespace cloudViewer {
namespace visualization {

class SmallButton : public gui::Button {
    using Super = Button;

public:
    explicit SmallButton(const char *title);

    gui::Size CalcPreferredSize(const gui::LayoutContext &context,
                                const Constraints &constraints) const override;
};

class SmallToggleButton : public SmallButton {
    using Super = SmallButton;

public:
    explicit SmallToggleButton(const char *title);
};

}  // namespace visualization
}  // namespace cloudViewer
