// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/** @file ContextItems.h
 *  @brief VTK context items for 2D overlay rendering (points, lines, shapes,
 * text)
 */

#include <vtkBrush.h>
#include <vtkContext2D.h>
#include <vtkContextItem.h>
#include <vtkPen.h>
#include <vtkTextProperty.h>

#include <cstring>
#include <string>
#include <vector>

#include "qVTK.h"

namespace VtkRendering {
namespace context_items {

/** @class VtkContextItemBase
 *  @brief Base context item with colors and opacity. Replaces PCLContextItem.
 */
class QVTK_ENGINE_LIB_API VtkContextItemBase : public vtkContextItem {
public:
    vtkTypeMacro(VtkContextItemBase, vtkContextItem);

    /// @param r Red component (0-255)
    /// @param g Green component (0-255)
    /// @param b Blue component (0-255)
    void setColors(unsigned char r, unsigned char g, unsigned char b) {
        colors[0] = r;
        colors[1] = g;
        colors[2] = b;
    }
    /// @param rgb RGB array [r, g, b]
    void setColors(unsigned char rgb[3]) {
        memcpy(colors, rgb, 3 * sizeof(unsigned char));
    }
    /// @param opacity Opacity value [0, 1]
    void setOpacity(double opacity) { SetOpacity(opacity); }

    unsigned char colors[3] = {0, 255, 0};
    std::vector<float> params;
};

/** @class Point
 *  @brief Point: (x, y) with color and size.
 */
class QVTK_ENGINE_LIB_API Point : public VtkContextItemBase {
public:
    vtkTypeMacro(Point, VtkContextItemBase);
    static Point* New();
    bool Paint(vtkContext2D* painter) override;
    /// @param x X coordinate
    /// @param y Y coordinate
    virtual void set(float x, float y);
};

/** @class Line
 *  @brief Line: (x1, y1, x2, y2) with color.
 */
class QVTK_ENGINE_LIB_API Line : public VtkContextItemBase {
public:
    vtkTypeMacro(Line, VtkContextItemBase);
    static Line* New();
    bool Paint(vtkContext2D* painter) override;
    /// @param x1 Start X
    /// @param y1 Start Y
    /// @param x2 End X
    /// @param y2 End Y
    virtual void set(float x1, float y1, float x2, float y2);
};

/** @class Circle
 *  @brief Circle: (x, y, radius) outline with color.
 */
class QVTK_ENGINE_LIB_API Circle : public VtkContextItemBase {
public:
    vtkTypeMacro(Circle, VtkContextItemBase);
    static Circle* New();
    bool Paint(vtkContext2D* painter) override;
    /// @param x Center X
    /// @param y Center Y
    /// @param radius Circle radius
    virtual void set(float x, float y, float radius);
};

/** @class Disk
 *  @brief Disk: filled circle (cx, cy, r) with color.
 */
class QVTK_ENGINE_LIB_API Disk : public Circle {
public:
    vtkTypeMacro(Disk, Circle);
    static Disk* New();
    bool Paint(vtkContext2D* painter) override;
};

/** @class Rectangle
 *  @brief Rectangle: (x, y, w, h) or (x_min, y_min, x_max, y_max) outline with
 * color. PCL addRectangle passes (x_min, y_min, x_max, y_max); params stored
 * for pickItem compatibility.
 */
class QVTK_ENGINE_LIB_API Rectangle : public VtkContextItemBase {
public:
    vtkTypeMacro(Rectangle, VtkContextItemBase);
    static Rectangle* New();
    bool Paint(vtkContext2D* painter) override;
    /// @param x Left X or x_min
    /// @param y Top Y or y_min
    /// @param w Width or x_max
    /// @param h Height or y_max
    virtual void set(float x, float y, float w, float h);
};

/** @class FilledRectangle
 *  @brief FilledRectangle: (x, y, w, h) filled with color.
 */
class QVTK_ENGINE_LIB_API FilledRectangle : public Rectangle {
public:
    vtkTypeMacro(FilledRectangle, Rectangle);
    static FilledRectangle* New();
    bool Paint(vtkContext2D* painter) override;
};

/** @class Points
 *  @brief Points: vector of (x,y) with color and size.
 */
class QVTK_ENGINE_LIB_API Points : public VtkContextItemBase {
public:
    vtkTypeMacro(Points, VtkContextItemBase);
    static Points* New();
    bool Paint(vtkContext2D* painter) override;
    /// @param xy Interleaved [x0,y0, x1,y1, ...] coordinates
    void set(const std::vector<float>& xy) { params = xy; }
};

/** @class Polygon
 *  @brief Polygon: closed polygon from vector of (x,y) with color.
 */
class QVTK_ENGINE_LIB_API Polygon : public Points {
public:
    vtkTypeMacro(Polygon, Points);
    static Polygon* New();
    bool Paint(vtkContext2D* painter) override;
};

/** @class Text
 *  @brief Text: (x, y) with text, color, bold, fontSize.
 */
class QVTK_ENGINE_LIB_API Text : public VtkContextItemBase {
public:
    vtkTypeMacro(Text, VtkContextItemBase);
    static Text* New();
    bool Paint(vtkContext2D* painter) override;
    /// @param x X position
    /// @param y Y position
    /// @param text Text string to render
    virtual void set(float x, float y, const std::string& text);
    void setBold(bool state = false) { bold_ = state; }
    void setFontSize(int fontSize = 10) { fontSize_ = fontSize; }

    std::string text;
    int fontSize_ = 10;
    bool bold_ = false;
};

/** @class Markers
 *  @brief Markers: points with size and separate point colors (like Points with
 * sprites).
 */
class QVTK_ENGINE_LIB_API Markers : public Points {
public:
    vtkTypeMacro(Markers, Points);
    static Markers* New();
    bool Paint(vtkContext2D* painter) override;
    /// @param size Marker size in pixels
    void setSize(float size) { size_ = size; }
    /// @param r Red component
    /// @param g Green component
    /// @param b Blue component
    void setPointColors(unsigned char r, unsigned char g, unsigned char b);
    /// @param rgb RGB array [r, g, b]
    void setPointColors(unsigned char rgb[3]);

    float size_ = 2.3f;
    unsigned char point_colors[3] = {255, 255, 255};
};

}  // namespace context_items
}  // namespace VtkRendering
