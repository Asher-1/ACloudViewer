// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "base/CVContextItem.h"

#include <vtkBrush.h>
#include <vtkContext2D.h>
#include <vtkImageData.h>
#include <vtkObjectFactory.h>
#include <vtkPen.h>
#include <vtkTextProperty.h>

namespace PclUtils {
vtkStandardNewMacro(ContextItem);
vtkStandardNewMacro(ContextImageItem);
namespace context_items {
vtkStandardNewMacro(Point);
vtkStandardNewMacro(Line);
vtkStandardNewMacro(Circle);
vtkStandardNewMacro(Disk);
vtkStandardNewMacro(Rectangle);
vtkStandardNewMacro(FilledRectangle);
vtkStandardNewMacro(Points);
vtkStandardNewMacro(Polygon);
vtkStandardNewMacro(Text);
vtkStandardNewMacro(Markers);
}  // namespace context_items
}  // namespace PclUtils

///////////////////////////////////////////////////////////////////////////////////////////
void PclUtils::ContextItem::setColors(unsigned char r,
                                      unsigned char g,
                                      unsigned char b) {
    colors[0] = r;
    colors[1] = g;
    colors[2] = b;
}

///////////////////////////////////////////////////////////////////////////////////////////
PclUtils::ContextImageItem::ContextImageItem() {
    image = vtkSmartPointer<vtkImageData>::New();
}

void PclUtils::ContextImageItem::set(float _x, float _y, vtkImageData* _image) {
    x = _x;
    y = _y;
    image->DeepCopy(_image);
}

bool PclUtils::ContextImageItem::Paint(vtkContext2D* painter) {
    SetOpacity(1.0);
    painter->DrawImage(x, y, image);
    return true;
}

///////////////////////////////////////////////////////////////////////////////////////////
void PclUtils::context_items::Point::set(float x, float y) {
    params.resize(2);
    params[0] = x;
    params[1] = y;
}

bool PclUtils::context_items::Point::Paint(vtkContext2D* painter) {
    painter->GetPen()->SetColor(
            colors[0], colors[1], colors[2],
            static_cast<unsigned char>(255.0 * GetOpacity()));
    painter->DrawPoint(params[0], params[1]);
    return true;
}

///////////////////////////////////////////////////////////////////////////////////////////
void PclUtils::context_items::Line::set(float start_x,
                                        float start_y,
                                        float end_x,
                                        float end_y) {
    params.resize(4);
    params[0] = start_x;
    params[1] = start_y;
    params[2] = end_x;
    params[3] = end_y;
}

bool PclUtils::context_items::Line::Paint(vtkContext2D* painter) {
    painter->GetPen()->SetColor(
            colors[0], colors[1], colors[2],
            static_cast<unsigned char>(255.0 * GetOpacity()));
    painter->DrawLine(params[0], params[1], params[2], params[3]);
    return true;
}

///////////////////////////////////////////////////////////////////////////////////////////
void PclUtils::context_items::Circle::set(float x, float y, float radius) {
    params.resize(4);
    params[0] = x;
    params[1] = y;
    params[2] = radius;
    params[3] = radius - 1;
}

bool PclUtils::context_items::Circle::Paint(vtkContext2D* painter) {
    painter->GetBrush()->SetColor(
            colors[0], colors[1], colors[2],
            static_cast<unsigned char>(255.0 * GetOpacity()));
    painter->GetPen()->SetColor(
            colors[0], colors[1], colors[2],
            static_cast<unsigned char>(255.0 * GetOpacity()));
    painter->DrawWedge(params[0], params[1], params[2], params[3], 0.0, 360.0);
    return true;
}

///////////////////////////////////////////////////////////////////////////////////////////
bool PclUtils::context_items::Disk::Paint(vtkContext2D* painter) {
    painter->GetBrush()->SetColor(
            colors[0], colors[1], colors[2],
            static_cast<unsigned char>(255.0 * GetOpacity()));
    painter->GetPen()->SetColor(
            colors[0], colors[1], colors[2],
            static_cast<unsigned char>(255.0 * GetOpacity()));
    painter->DrawEllipse(params[0], params[1], params[2], params[2]);
    return true;
}

///////////////////////////////////////////////////////////////////////////////////////////
void PclUtils::context_items::Rectangle::set(float x,
                                             float y,
                                             float w,
                                             float h) {
    params.resize(4);
    params[0] = x;
    params[1] = y;
    params[2] = w;
    params[3] = h;
}

bool PclUtils::context_items::Rectangle::Paint(vtkContext2D* painter) {
    painter->GetPen()->SetColor(
            colors[0], colors[1], colors[2],
            static_cast<unsigned char>(255.0 * GetOpacity()));
    float p[] = {params[0], params[1], params[2], params[1], params[2],
                 params[3], params[0], params[3], params[0], params[1]};
    painter->DrawPoly(p, 5);
    return true;
}

///////////////////////////////////////////////////////////////////////////////////////////
bool PclUtils::context_items::FilledRectangle::Paint(vtkContext2D* painter) {
    painter->GetBrush()->SetColor(
            colors[0], colors[1], colors[2],
            static_cast<unsigned char>(255.0 * GetOpacity()));
    painter->GetPen()->SetColor(
            colors[0], colors[1], colors[2],
            static_cast<unsigned char>(255.0 * GetOpacity()));
    painter->DrawRect(params[0], params[1], params[2], params[3]);
    return true;
}

///////////////////////////////////////////////////////////////////////////////////////////
bool PclUtils::context_items::Points::Paint(vtkContext2D* painter) {
    painter->GetPen()->SetColor(
            colors[0], colors[1], colors[2],
            static_cast<unsigned char>(255.0 * GetOpacity()));
    painter->DrawPoints(&params[0], static_cast<int>(params.size() / 2));
    return true;
}

///////////////////////////////////////////////////////////////////////////////////////////
bool PclUtils::context_items::Polygon::Paint(vtkContext2D* painter) {
    painter->GetBrush()->SetColor(
            colors[0], colors[1], colors[2],
            static_cast<unsigned char>(255.0 * GetOpacity()));
    painter->GetPen()->SetColor(
            colors[0], colors[1], colors[2],
            static_cast<unsigned char>(255.0 * GetOpacity()));
    painter->DrawPolygon(&params[0], static_cast<int>(params.size() / 2));
    return true;
}

///////////////////////////////////////////////////////////////////////////////////////////
void PclUtils::context_items::Text::set(float x,
                                        float y,
                                        const std::string& _text) {
    params.resize(2);
    params[0] = x;
    params[1] = y;
    text = _text;
}

bool PclUtils::context_items::Text::Paint(vtkContext2D* painter) {
    vtkTextProperty* text_property = painter->GetTextProp();
    text_property->SetColor(255.0 * colors[0], 255.0 * colors[1],
                            255.0 * colors[2]);
    text_property->SetOpacity(GetOpacity());
    text_property->SetFontFamilyToArial();
    text_property->SetFontSize(fontSize_);
    text_property->SetJustificationToLeft();
    bold_ ? text_property->BoldOn() : text_property->BoldOff();
    text_property->ShadowOff();
    painter->DrawString(params[0], params[1], text.c_str());
    return true;
}

///////////////////////////////////////////////////////////////////////////////////////////
void PclUtils::context_items::Markers::setPointColors(unsigned char r,
                                                      unsigned char g,
                                                      unsigned char b) {
    point_colors[0] = r;
    point_colors[1] = g;
    point_colors[2] = b;
}

void PclUtils::context_items::Markers::setPointColors(unsigned char rgb[3]) {
    memcpy(point_colors, rgb, 3 * sizeof(unsigned char));
}

bool PclUtils::context_items::Markers::Paint(vtkContext2D* painter) {
    int nb_points(params.size() / 2);
    if (size <= 0) size = 2.3f * painter->GetPen()->GetWidth();

    painter->GetPen()->SetWidth(size);
    painter->GetPen()->SetColor(
            colors[0], colors[1], colors[2],
            static_cast<unsigned char>(255.0 * GetOpacity()));
    painter->DrawPointSprites(nullptr, &params[0], nb_points);
    painter->GetPen()->SetWidth(1);
    painter->GetPen()->SetColor(
            point_colors[0], point_colors[1], point_colors[2],
            static_cast<unsigned char>(255.0 * GetOpacity()));
    painter->DrawPointSprites(nullptr, &params[0], nb_points);
    return true;
}
