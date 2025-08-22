// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <functional>
#include <string>

#include "visualization/gui/Widget.h"

namespace cloudViewer {
namespace visualization {
namespace gui {

class ListView : public Widget {
    using Super = Widget;

public:
    ListView();
    virtual ~ListView();

    void SetItems(const std::vector<std::string>& items);

    /// Returns the currently selected item in the list.
    int GetSelectedIndex() const;
    /// Returns the value of the currently selected item in the list.
    const char* GetSelectedValue() const;
    /// Selects the indicated row of the list. Does not call onValueChanged.
    void SetSelectedIndex(int index);

    Size CalcPreferredSize(const LayoutContext& context,
                           const Constraints& constraints) const override;

    Size CalcMinimumSize(const LayoutContext& context) const override;

    DrawResult Draw(const DrawContext& context) override;

    /// Calls onValueChanged(const char *selectedText, bool isDoubleClick)
    /// when the list selection changes because of user action.
    void SetOnValueChanged(
            std::function<void(const char*, bool)> on_value_changed);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace gui
}  // namespace visualization
}  // namespace cloudViewer
