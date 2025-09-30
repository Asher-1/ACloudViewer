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

class Label : public Widget {
    using Super = Widget;

public:
    /// Copies text
    explicit Label(const char* text = nullptr);
    ~Label();

    const char* GetText() const;
    /// Sets the text of the label (copies text)
    void SetText(const char* text);

    Color GetTextColor() const;
    void SetTextColor(const Color& color);

    FontId GetFontId() const;
    void SetFontId(const FontId font_id);

    Size CalcPreferredSize(const LayoutContext& context,
                           const Constraints& constraints) const override;

    DrawResult Draw(const DrawContext& context) override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace gui
}  // namespace visualization
}  // namespace cloudViewer
