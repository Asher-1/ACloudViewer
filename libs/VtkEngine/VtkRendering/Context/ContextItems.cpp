// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

/**
 * @file ContextItems.cpp
 * @brief Implementation of VTK context 2D drawing primitives (Point, Line,
 * Circle, etc.).
 */

#include "ContextItems.h"

#include <vtkObjectFactory.h>

namespace VtkRendering {
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

////////////////////////////////////////////////////////////////////////////////
void Point::set(float x, float y) {
    params.resize(2);
    params[0] = x;
    params[1] = y;
}

////////////////////////////////////////////////////////////////////////////////
void Line::set(float x1, float y1, float x2, float y2) {
    params.resize(4);
    params[0] = x1;
    params[1] = y1;
    params[2] = x2;
    params[3] = y2;
}

////////////////////////////////////////////////////////////////////////////////
void Circle::set(float x, float y, float radius) {
    params.resize(4);
    params[0] = x;
    params[1] = y;
    params[2] = radius;
    params[3] = radius - 1;
}

////////////////////////////////////////////////////////////////////////////////
void Rectangle::set(float x, float y, float w, float h) {
    params.resize(4);
    params[0] = x;
    params[1] = y;
    params[2] = w;
    params[3] = h;
}

// Note: PCL addRectangle passes (x_min, y_min, x_max, y_max) to set(), so
// params are (x1,y1,x2,y2). DrawPoly uses corners (x1,y1)-(x2,y2).

////////////////////////////////////////////////////////////////////////////////
void Text::set(float x, float y, const std::string& text_str) {
    params.resize(2);
    params[0] = x;
    params[1] = y;
    text = text_str;
}

////////////////////////////////////////////////////////////////////////////////
void Markers::setPointColors(unsigned char r,
                             unsigned char g,
                             unsigned char b) {
    point_colors[0] = r;
    point_colors[1] = g;
    point_colors[2] = b;
}

////////////////////////////////////////////////////////////////////////////////
void Markers::setPointColors(unsigned char rgb[3]) {
    memcpy(point_colors, rgb, 3 * sizeof(unsigned char));
}

////////////////////////////////////////////////////////////////////////////////
bool Point::Paint(vtkContext2D* painter) {
    painter->GetPen()->SetColor(
            colors[0], colors[1], colors[2],
            static_cast<unsigned char>(255.0 * GetOpacity()));
    painter->DrawPoint(params[0], params[1]);
    return true;
}

////////////////////////////////////////////////////////////////////////////////
bool Line::Paint(vtkContext2D* painter) {
    painter->GetPen()->SetColor(
            colors[0], colors[1], colors[2],
            static_cast<unsigned char>(255.0 * GetOpacity()));
    painter->DrawLine(params[0], params[1], params[2], params[3]);
    return true;
}

////////////////////////////////////////////////////////////////////////////////
bool Circle::Paint(vtkContext2D* painter) {
    painter->GetBrush()->SetColor(
            colors[0], colors[1], colors[2],
            static_cast<unsigned char>(255.0 * GetOpacity()));
    painter->GetPen()->SetColor(
            colors[0], colors[1], colors[2],
            static_cast<unsigned char>(255.0 * GetOpacity()));
    painter->DrawWedge(params[0], params[1], params[2], params[3], 0.0, 360.0);
    return true;
}

////////////////////////////////////////////////////////////////////////////////
bool Disk::Paint(vtkContext2D* painter) {
    painter->GetBrush()->SetColor(
            colors[0], colors[1], colors[2],
            static_cast<unsigned char>(255.0 * GetOpacity()));
    painter->GetPen()->SetColor(
            colors[0], colors[1], colors[2],
            static_cast<unsigned char>(255.0 * GetOpacity()));
    painter->DrawEllipse(params[0], params[1], params[2], params[2]);
    return true;
}

////////////////////////////////////////////////////////////////////////////////
bool Rectangle::Paint(vtkContext2D* painter) {
    painter->GetPen()->SetColor(
            colors[0], colors[1], colors[2],
            static_cast<unsigned char>(255.0 * GetOpacity()));
    // params are (x1,y1,x2,y2) - PCL addRectangle passes
    // (x_min,y_min,x_max,y_max)
    float p[] = {params[0], params[1], params[2], params[1], params[2],
                 params[3], params[0], params[3], params[0], params[1]};
    painter->DrawPoly(p, 5);
    return true;
}

////////////////////////////////////////////////////////////////////////////////
bool FilledRectangle::Paint(vtkContext2D* painter) {
    painter->GetBrush()->SetColor(
            colors[0], colors[1], colors[2],
            static_cast<unsigned char>(255.0 * GetOpacity()));
    painter->GetPen()->SetColor(
            colors[0], colors[1], colors[2],
            static_cast<unsigned char>(255.0 * GetOpacity()));
    // FilledRectangle uses (x, y, w, h) - addFilledRectangle passes (x_min,
    // y_min, width, height)
    painter->DrawRect(params[0], params[1], params[2], params[3]);
    return true;
}

////////////////////////////////////////////////////////////////////////////////
bool Points::Paint(vtkContext2D* painter) {
    if (params.empty()) return true;
    painter->GetPen()->SetColor(
            colors[0], colors[1], colors[2],
            static_cast<unsigned char>(255.0 * GetOpacity()));
    painter->DrawPoints(params.data(), static_cast<int>(params.size() / 2));
    return true;
}

////////////////////////////////////////////////////////////////////////////////
bool Polygon::Paint(vtkContext2D* painter) {
    if (params.empty()) return true;
    painter->GetBrush()->SetColor(
            colors[0], colors[1], colors[2],
            static_cast<unsigned char>(255.0 * GetOpacity()));
    painter->GetPen()->SetColor(
            colors[0], colors[1], colors[2],
            static_cast<unsigned char>(255.0 * GetOpacity()));
    painter->DrawPolygon(params.data(), static_cast<int>(params.size() / 2));
    return true;
}

////////////////////////////////////////////////////////////////////////////////
bool Text::Paint(vtkContext2D* painter) {
    vtkTextProperty* text_property = painter->GetTextProp();
    text_property->SetColor(colors[0] / 255.0, colors[1] / 255.0,
                            colors[2] / 255.0);
    text_property->SetOpacity(GetOpacity());
    text_property->SetFontFamilyToArial();
    text_property->SetFontSize(fontSize_);
    text_property->SetJustificationToLeft();
    bold_ ? text_property->BoldOn() : text_property->BoldOff();
    text_property->ShadowOff();
    painter->DrawString(params[0], params[1], text.c_str());
    return true;
}

////////////////////////////////////////////////////////////////////////////////
bool Markers::Paint(vtkContext2D* painter) {
    int nb_points = static_cast<int>(params.size() / 2);
    if (nb_points <= 0) return true;

    float size = size_;
    if (size <= 0) {
        size = 2.3f * painter->GetPen()->GetWidth();
    }

    painter->GetPen()->SetWidth(size);
    painter->GetPen()->SetColor(
            colors[0], colors[1], colors[2],
            static_cast<unsigned char>(255.0 * GetOpacity()));
    painter->DrawPointSprites(nullptr, params.data(), nb_points);
    painter->GetPen()->SetWidth(1);
    painter->GetPen()->SetColor(
            point_colors[0], point_colors[1], point_colors[2],
            static_cast<unsigned char>(255.0 * GetOpacity()));
    painter->DrawPointSprites(nullptr, params.data(), nb_points);
    return true;
}

}  // namespace context_items
}  // namespace VtkRendering
