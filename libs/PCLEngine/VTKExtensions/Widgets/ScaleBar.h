#pragma once
#include <vtkSmartPointer.h>
#include <vtkActor2D.h>
#include <vtkTextActor.h>
#include <vtkLineSource.h>
#include <vtkPolyDataMapper2D.h>
#include <vtkRenderer.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkCamera.h>

// Qt版本兼容性处理
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
    vtkSmartPointer<vtkActor2D> leftTickActor;   // 左端刻度线
    vtkSmartPointer<vtkActor2D> rightTickActor;  // 右端刻度线
    double lastLength = 0.0;
    bool visible = true;
    double dpiScale = 1.0;
    
    // 兼容不同Qt版本的DPI获取方法
    double getDPIScale();
    
    // 跨平台字体大小优化函数
    int getOptimizedFontSize(int baseFontSize = 18);
    
    // 跨平台DPI缩放处理函数
    double getPlatformAwareDPIScale();
    
    // 创建刻度线
    vtkSmartPointer<vtkActor2D> createTickActor(double x, double y, double length);
}; 