// ----------------------------------------------------------------------------
// -                        CloudViewer: asher-1.github.io                    -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 asher-1.github.io
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

#include "visualization/gui/ImageWidget.h"

#include <imgui.h>

#include <Image.h>
#include "visualization/gui/Theme.h"
#include "visualization/gui/Util.h"

namespace cloudViewer {
namespace visualization {
namespace gui {

struct ImageWidget::Impl {
    std::shared_ptr<UIImage> image_;
};

ImageWidget::ImageWidget() : impl_(new ImageWidget::Impl()) {
    impl_->image_ =
            cloudViewer::make_shared<UIImage>(cloudViewer::make_shared<geometry::Image>());
}

ImageWidget::ImageWidget(const char* image_path)
    : impl_(new ImageWidget::Impl()) {
    impl_->image_ = cloudViewer::make_shared<UIImage>(image_path);
}

ImageWidget::ImageWidget(std::shared_ptr<geometry::Image> image)
    : impl_(new ImageWidget::Impl()) {
    impl_->image_ = cloudViewer::make_shared<UIImage>(image);
}

ImageWidget::ImageWidget(std::shared_ptr<t::geometry::Image> image)
    : impl_(new ImageWidget::Impl()) {
    impl_->image_ = cloudViewer::make_shared<UIImage>(image);
}

ImageWidget::ImageWidget(visualization::rendering::TextureHandle texture_id,
                         float u0 /*= 0.0f*/,
                         float v0 /*= 0.0f*/,
                         float u1 /*= 1.0f*/,
                         float v1 /*= 1.0f*/)
    : impl_(new ImageWidget::Impl()) {
    impl_->image_ = cloudViewer::make_shared<UIImage>(texture_id, u0, v0, u1, v1);
}

ImageWidget::ImageWidget(std::shared_ptr<UIImage> image)
    : impl_(new ImageWidget::Impl()) {
    impl_->image_ = image;
}

ImageWidget::~ImageWidget() {}

void ImageWidget::UpdateImage(std::shared_ptr<geometry::Image> image) {
    GetUIImage()->UpdateImage(image);
}

void ImageWidget::UpdateImage(std::shared_ptr<t::geometry::Image> image) {
    GetUIImage()->UpdateImage(image);
}

std::shared_ptr<UIImage> ImageWidget::GetUIImage() const {
    return impl_->image_;
}

void ImageWidget::SetUIImage(std::shared_ptr<UIImage> image) {
    impl_->image_ = image;
}

Size ImageWidget::CalcPreferredSize(const LayoutContext& context,
                                    const Constraints& constraints) const {
    Size pref;
    if (impl_->image_) {
        pref = impl_->image_->CalcPreferredSize(context, constraints);
    }

    if (pref.width != 0 && pref.height != 0) {
        return pref;
    } else {
        return Size(5 * context.theme.font_size, 5 * context.theme.font_size);
    }
}

void ImageWidget::Layout(const LayoutContext& context) {
    Super::Layout(context);
}

Widget::DrawResult ImageWidget::Draw(const DrawContext& context) {
    auto& frame = GetFrame();
    UIImage::DrawParams params;  // .texture defaults to kBad handle
    if (impl_->image_) {
        params = impl_->image_->CalcDrawParams(context.renderer, frame);
    }

    if (params.texture != visualization::rendering::TextureHandle::kBad) {
        ImTextureID image_id =
                reinterpret_cast<ImTextureID>(params.texture.GetId());
        ImGui::SetCursorScreenPos(
                ImVec2(params.pos_x, params.pos_y - ImGui::GetScrollY()));
        ImGui::Image(image_id, ImVec2(params.width, params.height),
                     ImVec2(params.u0, params.v0),
                     ImVec2(params.u1, params.v1));
    } else {
        // Draw error message if we don't have an image, instead of
        // quietly failing or something.
        Rect frame = GetFrame();         // hide reference with a copy...
        frame.y -= ImGui::GetScrollY();  // ... so we can adjust for scrolling
        const char* error_text = "  Error\nloading\n image";
        Color fg(1.0, 1.0, 1.0);
        ImGui::GetWindowDrawList()->AddRectFilled(
                ImVec2(float(frame.x), float(frame.y)),
                ImVec2(float(frame.GetRight()), float(frame.GetBottom())),
                IM_COL32(255, 0, 0, 255));
        ImGui::GetWindowDrawList()->AddRect(
                ImVec2(float(frame.x), float(frame.y)),
                ImVec2(float(frame.GetRight()), float(frame.GetBottom())),
                IM_COL32(255, 255, 255, 255));
        ImGui::PushStyleColor(ImGuiCol_Text, colorToImgui(fg));

        auto padding = ImGui::GetStyle().FramePadding;
        float wrap_width = frame.width - std::ceil(2.0f * padding.x);
        float wrapX = ImGui::GetCursorPos().x + wrap_width;
        auto* font = ImGui::GetFont();
        auto text_size =
                font->CalcTextSizeA(float(context.theme.font_size), wrap_width,
                                    wrap_width, error_text);
        float x = (float(frame.width) - text_size.x) / 2.0f;
        float y = (float(frame.height) - text_size.y) / 2.0f;

        ImGui::SetCursorScreenPos(ImVec2(x + frame.x, y + frame.y));
        ImGui::PushTextWrapPos(wrapX);
        ImGui::TextWrapped("%s", error_text);
        ImGui::PopTextWrapPos();

        ImGui::PopStyleColor();
    }
    DrawImGuiTooltip();

    if (params.image_size_changed) {
        return Widget::DrawResult::RELAYOUT;
    }
    return Widget::DrawResult::NONE;
}

}  // namespace gui
}  // namespace visualization
}  // namespace cloudViewer
