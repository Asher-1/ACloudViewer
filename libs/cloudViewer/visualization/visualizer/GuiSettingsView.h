// ----------------------------------------------------------------------------
// -                        CloudViewer: asher-1.github.io                          -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 asher-1.github.io
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#pragma once

#include <functional>
#include <memory>
#include <string>

#include "visualization/gui/Layout.h"

namespace cloudViewer {

namespace visualization {

namespace gui {
class Button;
class Checkbox;
class Combobox;
class ColorEdit;
class CollapsableVert;
class Slider;
class VectorEdit;
}  // namespace gui

class GuiSettingsModel;

class GuiSettingsView : public gui::Vert {
public:
    GuiSettingsView(GuiSettingsModel& model,
                    const gui::Theme& theme,
                    const std::string& resource_path,
                    std::function<void(const char*)> on_load_ibl);

    void ShowFileMaterialEntry(bool show);
    void Update();

private:
    GuiSettingsModel& model_;
    std::function<void(const char*)> on_load_ibl_;

    std::shared_ptr<gui::Combobox> lighting_profile_;
    std::shared_ptr<gui::Checkbox> show_axes_;
    std::shared_ptr<gui::Checkbox> show_ground_;
    std::shared_ptr<gui::ColorEdit> bg_color_;
    std::shared_ptr<gui::Checkbox> show_skybox_;

    std::shared_ptr<gui::CollapsableVert> advanced_;
    std::shared_ptr<gui::Checkbox> ibl_enabled_;
    std::shared_ptr<gui::Checkbox> sun_enabled_;
    std::shared_ptr<gui::Combobox> ibls_;
    std::shared_ptr<gui::Slider> ibl_intensity_;
    std::shared_ptr<gui::Slider> sun_intensity_;
    std::shared_ptr<gui::VectorEdit> sun_dir_;
    std::shared_ptr<gui::Checkbox> sun_follows_camera_;
    std::shared_ptr<gui::ColorEdit> sun_color_;

    std::shared_ptr<gui::Combobox> material_type_;
    std::shared_ptr<gui::Combobox> prefab_material_;
    std::shared_ptr<gui::ColorEdit> material_color_;
    std::shared_ptr<gui::Button> reset_material_color_;
    std::shared_ptr<gui::Slider> point_size_;
};

}  // namespace visualization
}  // namespace cloudViewer
