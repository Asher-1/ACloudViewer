// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "visualization/visualizer/GuiWidgets.h"

#include "visualization/gui/Theme.h"

namespace cloudViewer {
namespace visualization {

SmallButton::SmallButton(const char *title) : gui::Button(title) {}

gui::Size SmallButton::CalcPreferredSize(const gui::LayoutContext &context,
                                         const Constraints &constraints) const {
    auto em = context.theme.font_size;
    auto size = Super::CalcPreferredSize(context, constraints);
    return gui::Size(size.width - em, int(std::round(1.2 * em)));
}

SmallToggleButton::SmallToggleButton(const char *title) : SmallButton(title) {
    SetToggleable(true);
}

}  // namespace visualization
}  // namespace cloudViewer
