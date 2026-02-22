// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
// CVContextItem.h - Pure VTK context items replacing pcl::visualization context
// items. These are used by CVImageViewer and CustomContextItem for 2D overlay
// rendering.

#pragma once

#include <vtkContextItem.h>
#include <vtkSmartPointer.h>

#include <string>
#include <vector>

#include "qPCL.h"

class vtkImageData;
class vtkContext2D;

namespace PclUtils {

/** \brief Base context item with color/opacity helpers.
 *  Replaces pcl::visualization::PCLContextItem. */
struct QPCL_ENGINE_LIB_API ContextItem : public vtkContextItem {
    vtkTypeMacro(ContextItem, vtkContextItem);
    static ContextItem* New();
    bool Paint(vtkContext2D*) override { return false; }
    void setColors(unsigned char r, unsigned char g, unsigned char b);
    void setColors(unsigned char rgb[3]) {
        memcpy(colors, rgb, 3 * sizeof(unsigned char));
    }
    void setOpacity(double opacity) { SetOpacity(opacity); }
    unsigned char colors[3]{0, 0, 0};
    std::vector<float> params;
};

/** \brief Image context item for drawing a vtkImageData in a 2D scene.
 *  Replaces pcl::visualization::PCLContextImageItem. */
struct QPCL_ENGINE_LIB_API ContextImageItem : public vtkContextItem {
    vtkTypeMacro(ContextImageItem, vtkContextItem);
    ContextImageItem();
    static ContextImageItem* New();
    bool Paint(vtkContext2D* painter) override;
    void set(float _x, float _y, vtkImageData* _image);
    vtkSmartPointer<vtkImageData> image;
    float x{0.0f}, y{0.0f};
};

namespace context_items {

struct QPCL_ENGINE_LIB_API Point : public ContextItem {
    vtkTypeMacro(Point, ContextItem);
    static Point* New();
    bool Paint(vtkContext2D* painter) override;
    virtual void set(float _x, float _y);
};

struct QPCL_ENGINE_LIB_API Line : public ContextItem {
    vtkTypeMacro(Line, ContextItem);
    static Line* New();
    bool Paint(vtkContext2D* painter) override;
    virtual void set(float _x_1, float _y_1, float _x_2, float _y_2);
};

struct QPCL_ENGINE_LIB_API Circle : public ContextItem {
    vtkTypeMacro(Circle, ContextItem);
    static Circle* New();
    bool Paint(vtkContext2D* painter) override;
    virtual void set(float _x, float _y, float _r);
};

struct QPCL_ENGINE_LIB_API Disk : public Circle {
    vtkTypeMacro(Disk, Circle);
    static Disk* New();
    bool Paint(vtkContext2D* painter) override;
};

struct QPCL_ENGINE_LIB_API Rectangle : public ContextItem {
    vtkTypeMacro(Rectangle, Point);
    static Rectangle* New();
    bool Paint(vtkContext2D* painter) override;
    virtual void set(float _x, float _y, float _w, float _h);
};

struct QPCL_ENGINE_LIB_API FilledRectangle : public Rectangle {
    vtkTypeMacro(FilledRectangle, Rectangle);
    static FilledRectangle* New();
    bool Paint(vtkContext2D* painter) override;
};

struct QPCL_ENGINE_LIB_API Points : public ContextItem {
    vtkTypeMacro(Points, ContextItem);
    static Points* New();
    bool Paint(vtkContext2D* painter) override;
    void set(const std::vector<float>& _xy) { params = _xy; }
};

struct QPCL_ENGINE_LIB_API Polygon : public Points {
    vtkTypeMacro(Polygon, Points);
    static Polygon* New();
    bool Paint(vtkContext2D* painter) override;
};

struct QPCL_ENGINE_LIB_API Text : public ContextItem {
    vtkTypeMacro(Text, ContextItem);
    static Text* New();
    bool Paint(vtkContext2D* painter) override;
    virtual void set(float x, float y, const std::string& _text);
    inline void setBold(bool state = false) { bold_ = state; }
    inline void setFontSize(int fontSize = 10) { fontSize_ = fontSize; }
    std::string text;
    int fontSize_{10};
    bool bold_{false};
};

struct QPCL_ENGINE_LIB_API Markers : public Points {
    vtkTypeMacro(Markers, Points);
    static Markers* New();
    bool Paint(vtkContext2D* painter) override;
    void setSize(float _size) { size = _size; }
    void setPointColors(unsigned char r, unsigned char g, unsigned char b);
    void setPointColors(unsigned char rgb[3]);
    float size{0.0f};
    unsigned char point_colors[3]{0, 0, 0};
};

}  // namespace context_items
}  // namespace PclUtils
