#pragma once
#include <vtkSmartPointer.h>
#include <vtkActor2D.h>
#include <vtkTextActor.h>
#include <vtkLineSource.h>
#include <vtkPolyDataMapper2D.h>
#include <vtkRenderer.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkCamera.h>

// Qt version compatibility handling
#include <QApplication>
#include <QDesktopWidget>
#include <QScreen>

class ScaleBar {
public:
    ScaleBar(vtkRenderer* renderer);
    ~ScaleBar();
    void update(vtkRenderer* renderer, vtkRenderWindowInteractor* interactor);
    void setVisible(bool visible);
    
private:
    vtkSmartPointer<vtkActor2D> lineActor;
    vtkSmartPointer<vtkTextActor> textActor;
    vtkSmartPointer<vtkActor2D> leftTickActor;   // Left tick mark
    vtkSmartPointer<vtkActor2D> rightTickActor;  // Right tick mark
    double lastLength = 0.0;
    bool visible = true;
    double dpiScale = 1.0;
    
    // DPI retrieval method compatible with different Qt versions
    double getDPIScale();
    
    // Cross-platform font size optimization function
    int getOptimizedFontSize(int baseFontSize = 18);
    
    // Cross-platform DPI scaling function
    double getPlatformAwareDPIScale();
    
    // Create tick mark actor
    vtkSmartPointer<vtkActor2D> createTickActor(double x, double y, double length);
}; 