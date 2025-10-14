// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef vtkBoundingRectContextDevice2D_h
#define vtkBoundingRectContextDevice2D_h

#include "qPCL.h"
#include "vtkContextDevice2D.h"

class vtkAbstractContextItem;

class QPCL_ENGINE_LIB_API vtkBoundingRectContextDevice2D
    : public vtkContextDevice2D {
public:
    vtkTypeMacro(vtkBoundingRectContextDevice2D,
                 vtkContextDevice2D) void PrintSelf(ostream& os,
                                                    vtkIndent indent) override;
    static vtkBoundingRectContextDevice2D* New();

    /**
     * Set/get delegate device used to compute bounding boxes around strings.
     */
    vtkSetObjectMacro(DelegateDevice, vtkContextDevice2D);
    vtkGetObjectMacro(DelegateDevice, vtkContextDevice2D);

    /**
     * reset the bounding box.
     */
    void Reset();

    /**
     * Get the bounding rect that contains the given vtkAbstractContextItem.
     */
    static vtkRectf GetBoundingRect(vtkAbstractContextItem* item,
                                    vtkViewport* viewport);

    /**
     * Get the bounding rect that contains all the primitives rendered by
     * the device so far.
     */
    vtkRectf GetBoundingRect();

    /**
     * Expand bounding box to contain the string's bounding box.
     */
    void DrawString(float* point, const vtkStdString& string) override;

    /**
     * Expand bounding box to contain the string's bounding box.
     */
    void DrawMathTextString(float* point, const vtkStdString& string) override;

    /**
     * Expand bounding box to contain the image's bounding box.
     */
    void DrawImage(float p[2], float scale, vtkImageData* image) override;

    /**
     * Expand bounding box to contain the image's bounding box.
     */
    void DrawImage(const vtkRectf& pos, vtkImageData* image) override;

    /**
     * Draw the supplied PolyData at the given x, y (p[0], p[1]) (bottom
     * corner), scaled by scale (1.0 would match the actual dataset).
     * @warning Not currently implemented.
     */
    void DrawPolyData(float vtkNotUsed(p)[2],
                      float vtkNotUsed(scale),
                      vtkPolyData* vtkNotUsed(polyData),
                      vtkUnsignedCharArray* vtkNotUsed(colors),
                      int vtkNotUsed(scalarMode)) override {}

    /**
     * Implement pure virtual member function. Does not affect bounding rect.
     */
    void SetColor4(unsigned char color[4]) override;

    /**
     * Implement pure virtual member function. Does not affect bounding rect.
     */
    void SetTexture(vtkImageData* image, int properties) override;

    /**
     * Implement pure virtual member function. Does not affect bounding rect.
     */
    void SetPointSize(float size) override;

    /**
     * Implement pure virtual member function. Forward line width to
     * delegate device.
     */
    void SetLineWidth(float width) override;

    /**
     * Implement pure virtual member function. Forward line type to
     * delegate device.
     */
    void SetLineType(int type) override;

    /**
     * Forward current matrix to delegate device.
     */
    void SetMatrix(vtkMatrix3x3* m) override;

    /**
     * Get current matrix from delegate device.
     */
    void GetMatrix(vtkMatrix3x3* m) override;

    /**
     * Multiply the current matrix in the delegate device by this one.
     */
    void MultiplyMatrix(vtkMatrix3x3* m) override;

    /**
     * Push matrix in the delegate device.
     */
    void PushMatrix() override;

    /**
     * Pope matrix from the delegate device.
     */
    void PopMatrix() override;

    /**
     * Implement pure virtual member function. Does nothing.
     */
    void EnableClipping(bool enable) override;

    /**
     * Implement pure virtual member function. Does nothing.
     */
    void SetClipping(int* x) override;

    /**
     * Forward the pen to the delegate device.
     */
    void ApplyPen(vtkPen* pen) override;

    /**
     * Get the pen from the delegate device.
     */
    vtkPen* GetPen() override;

    /**
     * Forward the brush to the delegate device.
     */
    void ApplyBrush(vtkBrush* brush) override;

    /**
     * Get the brush from the delegate device.
     */
    vtkBrush* GetBrush() override;

    /**
     * Forward the text property to the delegate device.
     */
    void ApplyTextProp(vtkTextProperty* prop) override;

    /**
     * Get the text property from the delegate device.
     */
    vtkTextProperty* GetTextProp() override;

    /**
     * Expand bounding box to contain the given polygon.
     */
    void DrawPoly(float* points,
                  int n,
                  unsigned char* colors = 0,
                  int nc_comps = 0) override;

    /**
     * Expand bounding rect to contain the given lines.
     */
    void DrawLines(float* f,
                   int n,
                   unsigned char* colors = 0,
                   int nc_comps = 0) override;

    /**
     * Expand bounding rect to contain the given points.
     */
    void DrawPoints(float* points,
                    int n,
                    unsigned char* colors = 0,
                    int nc_comps = 0) override;

    /**
     * Expand bounding rect to contain the point sprites.
     */
    void DrawPointSprites(vtkImageData* sprite,
                          float* points,
                          int n,
                          unsigned char* colors = 0,
                          int nc_comps = 0) override;

    /**
     * Expand bounding rect to contain the markers.
     */
    void DrawMarkers(int shape,
                     bool highlight,
                     float* points,
                     int n,
                     unsigned char* colors = 0,
                     int nc_comps = 0) override;

    /**
     * Expand bounding rect to contain the ellipse.
     */
    void DrawEllipseWedge(float x,
                          float y,
                          float outRx,
                          float outRy,
                          float inRx,
                          float inRy,
                          float startAngle,
                          float stopAngle) override;

    /**
     * Expand bounding rect to contain the elliptic arc.
     */
    void DrawEllipticArc(float x,
                         float y,
                         float rX,
                         float rY,
                         float startAngle,
                         float stopAngle) override;

    /**
     * Forward string bounds calculation to the delegate device.
     */
    void ComputeStringBounds(const vtkStdString& string,
                             float bounds[4]) override;

    /**
     * Forward string bounds calculation to the delegate device.
     */
    void ComputeJustifiedStringBounds(const char* string,
                                      float bounds[4]) override;

    /**
     * Call before drawing to this device.
     */
    void Begin(vtkViewport*) override;

    /**
     * Call after drawing to this device.
     */
    void End() override;

    /**
     * Get value from delegate device.
     */
    bool GetBufferIdMode() const override;

    /**
     * Begin ID buffering mode.
     */
    void BufferIdModeBegin(vtkAbstractContextBufferId* bufferId) override;

    /**
     * End ID buffering mode.
     */
    void BufferIdModeEnd() override;

protected:
    vtkBoundingRectContextDevice2D();
    ~vtkBoundingRectContextDevice2D() override;

    /**
     * Is the bounding rect initialized?
     */
    bool Initialized;

    /**
     * Cumulative rect holding the bounds of the primitives rendered by the
     * device.
     */
    vtkRectf BoundingRect;

    /**
     * Delegate ContextDevice2D to handle certain computations
     */
    vtkContextDevice2D* DelegateDevice;

    /**
     * Add a point to the cumulative bounding rect.
     */
    void AddPoint(float x, float y);
    void AddPoint(float point[2]);

    /**
     * Add a rect to the cumulative bounding rect.
     */
    void AddRect(const vtkRectf& rect);

private:
    vtkBoundingRectContextDevice2D(const vtkBoundingRectContextDevice2D&) =
            delete;
    void operator=(const vtkBoundingRectContextDevice2D&) = delete;
};

#endif  // vtkBoundingRectContextDevice2D
