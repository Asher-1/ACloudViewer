// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <pcl/pcl_macros.h>
#include <vtkContextItem.h>
#include <vtkSmartPointer.h>

#include <vector>
class vtkImageData;
class vtkContext2D;

namespace pcl {
namespace visualization {
/** Struct PCLContextItem represents our own custom version of vtkContextItem,
 * used by the ImageViewer class.
 *
 * \author Nizar Sallem
 */
struct PCL_EXPORTS PCLContextItem : public vtkContextItem {
    vtkTypeMacro(PCLContextItem, vtkContextItem);
    static PCLContextItem* New();
    bool Paint(vtkContext2D*) override { return (false); };
    void setColors(unsigned char r, unsigned char g, unsigned char b);
    void setColors(unsigned char rgb[3]) {
        memcpy(colors, rgb, 3 * sizeof(unsigned char));
    }
    void setOpacity(double opacity) { SetOpacity(opacity); };
    unsigned char colors[3];
    std::vector<float> params;
};

/** Struct PCLContextImageItem a specification of vtkContextItem, used to add an
 * image to the scene in the ImageViewer class.
 *
 * \author Nizar Sallem
 */
struct PCL_EXPORTS PCLContextImageItem : public vtkContextItem {
    vtkTypeMacro(PCLContextImageItem, vtkContextItem);
    PCLContextImageItem();

    static PCLContextImageItem* New();
    bool Paint(vtkContext2D* painter) override;
    void set(float _x, float _y, vtkImageData* _image);
    vtkSmartPointer<vtkImageData> image;
    float x, y;
};

namespace context_items {
struct PCL_EXPORTS Point : public PCLContextItem {
    vtkTypeMacro(Point, PCLContextItem);
    static Point* New();
    bool Paint(vtkContext2D* painter) override;
    virtual void set(float _x, float _y);
};

struct PCL_EXPORTS Line : public PCLContextItem {
    vtkTypeMacro(Line, PCLContextItem);
    static Line* New();
    bool Paint(vtkContext2D* painter) override;
    virtual void set(float _x_1, float _y_1, float _x_2, float _y_2);
};

struct PCL_EXPORTS Circle : public PCLContextItem {
    vtkTypeMacro(Circle, PCLContextItem);
    static Circle* New();
    bool Paint(vtkContext2D* painter) override;
    virtual void set(float _x, float _y, float _r);
};

struct PCL_EXPORTS Disk : public Circle {
    vtkTypeMacro(Disk, Circle);
    static Disk* New();
    bool Paint(vtkContext2D* painter) override;
};

struct PCL_EXPORTS Rectangle : public PCLContextItem {
    vtkTypeMacro(Rectangle, Point);
    static Rectangle* New();
    bool Paint(vtkContext2D* painter) override;
    virtual void set(float _x, float _y, float _w, float _h);
};

struct PCL_EXPORTS FilledRectangle : public Rectangle {
    vtkTypeMacro(FilledRectangle, Rectangle);
    static FilledRectangle* New();
    bool Paint(vtkContext2D* painter) override;
};

struct PCL_EXPORTS Points : public PCLContextItem {
    vtkTypeMacro(Points, PCLContextItem);
    static Points* New();
    bool Paint(vtkContext2D* painter) override;
    void set(const std::vector<float>& _xy) { params = _xy; }
};

struct PCL_EXPORTS Polygon : public Points {
    vtkTypeMacro(Polygon, Points);
    static Polygon* New();
    bool Paint(vtkContext2D* painter) override;
};

struct PCL_EXPORTS Text : public PCLContextItem {
    vtkTypeMacro(Text, PCLContextItem);
    static Text* New();
    bool Paint(vtkContext2D* painter) override;
    virtual void set(float x, float y, const std::string& _text);
    std::string text;
};

struct PCL_EXPORTS Markers : public Points {
    vtkTypeMacro(Markers, Points);
    static Markers* New();
    bool Paint(vtkContext2D* painter) override;
    void setSize(float _size) { size = _size; }
    void setPointColors(unsigned char r, unsigned char g, unsigned char b);
    void setPointColors(unsigned char rgb[3]);
    float size;
    unsigned char point_colors[3];
};
}  // namespace context_items
}  // namespace visualization
}  // namespace pcl
