#include "ScaleBar.h"
#include <vtkTextProperty.h>
#include <vtkRenderWindow.h>
#include <vtkCoordinate.h>
#include <vtkProperty2D.h>
#include <vtkLineSource.h>
#include <vtkPolyDataMapper2D.h>
#include <vtkActor2D.h>
#include <vtkTextActor.h>
#include <vtkCamera.h>
#include <vtkRenderer.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkAlgorithmOutput.h>
#include <vtkPolyData.h>
#include <vtkSmartPointer.h>
#include <sstream>
#include <cmath>
#include <cstring>
#include <QString>
#include <QProcessEnvironment>

ScaleBar::ScaleBar(vtkRenderer* renderer) {
    // 获取DPI缩放
    dpiScale = getDPIScale();
    
    // 创建线段
    auto lineSource = vtkSmartPointer<vtkLineSource>::New();
    lineSource->SetPoint1(0.0, 0.0, 0.0);
    lineSource->SetPoint2(100.0, 0.0, 0.0); // 初始长度

    auto mapper = vtkSmartPointer<vtkPolyDataMapper2D>::New();
    mapper->SetInputConnection(lineSource->GetOutputPort());

    lineActor = vtkSmartPointer<vtkActor2D>::New();
    lineActor->SetMapper(mapper);
    lineActor->GetProperty()->SetColor(1.0, 1.0, 1.0);
    lineActor->GetProperty()->SetLineWidth(3.0 * dpiScale);

    // 创建文本
    textActor = vtkSmartPointer<vtkTextActor>::New();
    textActor->SetInput("1 m");
    textActor->GetTextProperty()->SetFontSize(static_cast<int>(18.0 * dpiScale));
    textActor->GetTextProperty()->SetColor(1.0, 1.0, 1.0);
    textActor->GetTextProperty()->SetJustificationToCentered();
    textActor->GetTextProperty()->SetVerticalJustificationToTop();
    textActor->SetPosition(50.0 * dpiScale, 25.0 * dpiScale); // 初始位置，会在update中调整

    // 创建刻度线
    leftTickActor = createTickActor(0.0, 0.0, 10.0 * dpiScale);
    rightTickActor = createTickActor(100.0, 0.0, 10.0 * dpiScale);

    if (renderer) {
        renderer->AddActor2D(lineActor);
        renderer->AddActor2D(textActor);
        renderer->AddActor2D(leftTickActor);
        renderer->AddActor2D(rightTickActor);
    }
}

ScaleBar::~ScaleBar() {}

void ScaleBar::setVisible(bool v) {
    visible = v;
    lineActor->SetVisibility(v);
    textActor->SetVisibility(v);
    leftTickActor->SetVisibility(v);
    rightTickActor->SetVisibility(v);
}

double ScaleBar::getDPIScale() {
    // 兼容不同Qt版本的DPI获取方法
    if (!QApplication::instance()) {
        return 1.0;
    }
    
    // 方法1: Qt 5.6+ 使用 QScreen::devicePixelRatio() (推荐)
    #if QT_VERSION >= QT_VERSION_CHECK(5, 6, 0)
    QScreen* screen = QApplication::primaryScreen();
    if (screen) {
        return screen->devicePixelRatio();
    }
    #endif
    
    // 方法2: Qt 5.0+ 使用 QApplication::devicePixelRatio()
    #if QT_VERSION >= QT_VERSION_CHECK(5, 0, 0)
    return QApplication::devicePixelRatio();
    #endif
    
    // 方法3: Qt 4.x 使用 QDesktopWidget 计算
    #if QT_VERSION < QT_VERSION_CHECK(5, 0, 0)
    QDesktopWidget* desktop = QApplication::desktop();
    if (desktop) {
        // 通过物理DPI和逻辑DPI计算缩放
        int physicalDPI = desktop->physicalDpiX();
        int logicalDPI = desktop->logicalDpiX();
        if (logicalDPI > 0) {
            return static_cast<double>(physicalDPI) / logicalDPI;
        }
    }
    #endif
    
    // 方法4: 通过环境变量或系统检测
    const char* qt_scale_factor = qgetenv("QT_SCALE_FACTOR");
    if (qt_scale_factor) {
        bool ok;
        double scale = QString(qt_scale_factor).toDouble(&ok);
        if (ok && scale > 0) {
            return scale;
        }
    }
    
    return 1.0; // 默认缩放
}

vtkSmartPointer<vtkActor2D> ScaleBar::createTickActor(double x, double y, double length) {
    auto lineSource = vtkSmartPointer<vtkLineSource>::New();
    lineSource->SetPoint1(x, y, 0.0);
    lineSource->SetPoint2(x, y + length, 0.0); // 垂直刻度线

    auto mapper = vtkSmartPointer<vtkPolyDataMapper2D>::New();
    mapper->SetInputConnection(lineSource->GetOutputPort());

    auto actor = vtkSmartPointer<vtkActor2D>::New();
    actor->SetMapper(mapper);
    actor->GetProperty()->SetColor(1.0, 1.0, 1.0);
    actor->GetProperty()->SetLineWidth(2.0 * dpiScale);

    return actor;
}

void ScaleBar::update(vtkRenderer* renderer, vtkRenderWindowInteractor* interactor) {
    if (!visible || !renderer || !renderer->GetRenderWindow()) return;
    
    // 动态更新DPI缩放（以防窗口移动到不同DPI的显示器）
    double currentDPIScale = getDPIScale();
    if (std::abs(currentDPIScale - dpiScale) > 0.1) {
        dpiScale = currentDPIScale;
        // 更新字体大小和线宽
        if (textActor) {
            textActor->GetTextProperty()->SetFontSize(static_cast<int>(18.0 * dpiScale));
            textActor->GetTextProperty()->SetJustificationToCentered();
            textActor->GetTextProperty()->SetVerticalJustificationToTop();
        }
        if (lineActor) {
            lineActor->GetProperty()->SetLineWidth(3.0 * dpiScale);
        }
        if (leftTickActor) {
            leftTickActor->GetProperty()->SetLineWidth(2.0 * dpiScale);
        }
        if (rightTickActor) {
            rightTickActor->GetProperty()->SetLineWidth(2.0 * dpiScale);
        }
    }
    
    // 获取窗口大小
    int* size = renderer->GetRenderWindow()->GetSize();
    int winW = size[0];
    int winH = size[1];
    // 比例尺长度（像素），考虑DPI缩放
    int barPixelLen = static_cast<int>((winW / 6.0) * dpiScale); // 约为窗口宽度的1/6
    // 获取相机
    vtkCamera* cam = renderer->GetActiveCamera();
    double p1[3] = {50.0 * dpiScale, 50.0 * dpiScale, 0.0}; // 屏幕左下角偏移
    double p2[3] = {50.0 * dpiScale + static_cast<double>(barPixelLen), 50.0 * dpiScale, 0.0};
    // 屏幕坐标转世界坐标
    double world1[4], world2[4];
    renderer->SetDisplayPoint(static_cast<int>(p1[0]), static_cast<int>(p1[1]), 0);
    renderer->DisplayToWorld();
    memcpy(world1, renderer->GetWorldPoint(), sizeof(double) * 4);
    renderer->SetDisplayPoint(static_cast<int>(p2[0]), static_cast<int>(p2[1]), 0);
    renderer->DisplayToWorld();
    memcpy(world2, renderer->GetWorldPoint(), sizeof(double) * 4);
    // 计算世界距离
    double dx = (world2[0]/world2[3]) - (world1[0]/world1[3]);
    double dy = (world2[1]/world2[3]) - (world1[1]/world1[3]);
    double dz = (world2[2]/world2[3]) - (world1[2]/world1[3]);
    double dist = sqrt(dx*dx + dy*dy + dz*dz);
    // 取合适的显示单位
    double showLen = dist;
    std::string unit = "m";
    if (showLen < 0.01) {
        showLen *= 1000;
        unit = "mm";
    } else if (showLen < 1) {
        showLen *= 100;
        unit = "cm";
    } else if (showLen > 1000) {
        showLen /= 1000;
        unit = "km";
    }
    // 取整
    double niceLen = showLen;
    if (showLen > 10) niceLen = round(showLen / 10) * 10;
    else if (showLen > 1) niceLen = round(showLen);
    else niceLen = round(showLen * 10) / 10.0;
    // 更新线段
    auto mapper = dynamic_cast<vtkPolyDataMapper2D*>(lineActor->GetMapper());
    if (mapper) {
        auto lineSource = dynamic_cast<vtkLineSource*>(mapper->GetInputConnection(0,0)->GetProducer());
        if (lineSource) {
                    lineSource->SetPoint1(p1[0], p1[1], 0.0);
        lineSource->SetPoint2(p1[0] + static_cast<double>(barPixelLen), p1[1], 0.0);
            lineSource->Update();
        }
    }
    // 更新刻度线位置
    if (leftTickActor && rightTickActor) {
        auto leftMapper = dynamic_cast<vtkPolyDataMapper2D*>(leftTickActor->GetMapper());
        auto rightMapper = dynamic_cast<vtkPolyDataMapper2D*>(rightTickActor->GetMapper());
        
        if (leftMapper && rightMapper) {
            auto leftSource = dynamic_cast<vtkLineSource*>(leftMapper->GetInputConnection(0,0)->GetProducer());
            auto rightSource = dynamic_cast<vtkLineSource*>(rightMapper->GetInputConnection(0,0)->GetProducer());
            
            if (leftSource && rightSource) {
                double tickLength = 8.0 * dpiScale;
                leftSource->SetPoint1(p1[0], p1[1], 0.0);
                leftSource->SetPoint2(p1[0], p1[1] + tickLength, 0.0);
                leftSource->Update();
                
                rightSource->SetPoint1(p1[0] + static_cast<double>(barPixelLen), p1[1], 0.0);
                rightSource->SetPoint2(p1[0] + static_cast<double>(barPixelLen), p1[1] + tickLength, 0.0);
                rightSource->Update();
            }
        }
    }
    
    // 更新文本
    std::ostringstream oss;
    oss.precision(2);
    oss << std::fixed << niceLen << " " << unit;
    textActor->SetInput(oss.str().c_str());
    
    // 设置文本居中对齐
    textActor->GetTextProperty()->SetJustificationToCentered();
    textActor->GetTextProperty()->SetVerticalJustificationToTop();
    
    // 计算文本位置：在线条下方居中
    double textX = p1[0] + static_cast<double>(barPixelLen) / 2.0; // 线条中心X坐标
    double textY = p1[1] - 20.0 * dpiScale; // 线条下方
    textActor->SetPosition(textX, textY);
} 