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
#include <QScreen>
#include <algorithm>
#include <QCoreApplication>
#include <QApplication>
#include <QDesktopWidget>

ScaleBar::ScaleBar(vtkRenderer* renderer) {
    // 获取跨平台优化的DPI缩放
    dpiScale = getPlatformAwareDPIScale();
    
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

    // 创建文本 - 使用跨平台优化的字体大小
    textActor = vtkSmartPointer<vtkTextActor>::New();
    textActor->SetInput("1 m");
    int optimizedFontSize = getOptimizedFontSize(18);
    textActor->GetTextProperty()->SetFontSize(optimizedFontSize);
    textActor->GetTextProperty()->SetColor(1.0, 1.0, 1.0);
    textActor->GetTextProperty()->SetJustificationToCentered();
    textActor->GetTextProperty()->SetVerticalJustificationToTop();
    // 初始位置设置为居中，会在update中调整到正确位置
    textActor->SetPosition(100.0 * dpiScale, 25.0 * dpiScale);

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
    QCoreApplication* coreApp = QCoreApplication::instance();
    QApplication* app = qobject_cast<QApplication*>(coreApp);
    if (app) {
        return app->devicePixelRatio();
    }
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

int ScaleBar::getOptimizedFontSize(int baseFontSize) {
    // 获取屏幕信息
    QScreen* screen = QApplication::primaryScreen();
    if (!screen) {
        return baseFontSize;
    }
    
    // 获取屏幕分辨率信息
    QSize screenSize = screen->size();
    int screenWidth = screenSize.width();
    int screenHeight = screenSize.height();
    int screenDPI = screen->physicalDotsPerInch();
    int dpiScale = static_cast<int>(getDPIScale());
    
    // 平台特定的基础字体大小调整
    int platformBaseSize = baseFontSize;
    #ifdef Q_OS_MAC
        // macOS: 默认字体稍大，但需要考虑Retina显示器的过度放大
        platformBaseSize = baseFontSize;
        if (dpiScale > 1) {
            // Retina显示器：使用较小的字体避免过度放大
            platformBaseSize = std::max(12, baseFontSize - (dpiScale - 1) * 3);
        }
    #elif defined(Q_OS_WIN)
        // Windows: 根据DPI调整字体大小
        if (screenDPI > 120) {
            // 高DPI显示器
            platformBaseSize = std::max(12, baseFontSize - 2);
        } else if (screenDPI < 96) {
            // 低DPI显示器
            platformBaseSize = baseFontSize + 2;
        }
    #elif defined(Q_OS_LINUX)
        // Linux: 根据屏幕分辨率调整
        if (screenWidth >= 1920 && screenHeight >= 1080) {
            // 高分辨率显示器
            platformBaseSize = std::max(12, baseFontSize - 2);
        } else if (screenWidth < 1366) {
            // 低分辨率显示器
            platformBaseSize = baseFontSize + 2;
        }
    #endif
    
    // 分辨率特定的调整
    int resolutionFactor = 0;
    if (screenWidth >= 2560 && screenHeight >= 1440) {
        // 2K及以上分辨率
        resolutionFactor = -1;
    } else if (screenWidth < 1366) {
        // 低分辨率
        resolutionFactor = 1;
    }
    
    // 最终字体大小计算
    int finalSize = platformBaseSize + resolutionFactor;
    
    // 确保字体大小在合理范围内
    finalSize = std::max(10, std::min(32, finalSize));
    
    return finalSize;
}

double ScaleBar::getPlatformAwareDPIScale() {
    double dpiScale = getDPIScale();
    QScreen* screen = QApplication::primaryScreen();
    if (!screen) {
        return dpiScale;
    }
    
    // 获取屏幕信息
    QSize screenSize = screen->size();
    int screenWidth = screenSize.width();
    int screenHeight = screenSize.height();
    int screenDPI = screen->physicalDotsPerInch();
    
    // 平台特定的DPI缩放调整
    double adjustedScale = dpiScale;
    
    #ifdef Q_OS_MAC
        // macOS: Retina显示器需要特殊处理
        if (dpiScale > 1) {
            // 对于UI元素，使用较小的缩放以避免过度放大
            adjustedScale = 1.0 + (dpiScale - 1.0) * 0.6;
        }
    #elif defined(Q_OS_WIN)
        // Windows: 根据DPI设置调整
        if (screenDPI > 120) {
            // 高DPI显示器，适当减小缩放
            adjustedScale = std::min(adjustedScale, 1.4);
        } else if (screenDPI < 96) {
            // 低DPI显示器，适当增加缩放
            adjustedScale = std::max(adjustedScale, 1.0);
        }
    #elif defined(Q_OS_LINUX)
        // Linux: 根据分辨率调整
        if (screenWidth >= 2560 && screenHeight >= 1440) {
            // 超高分辨率，减小缩放
            adjustedScale = std::min(adjustedScale, 1.2);
        } else if (screenWidth < 1366) {
            // 低分辨率，增加缩放
            adjustedScale = std::max(adjustedScale, 1.0);
        }
    #endif
    
    // 确保缩放在合理范围内
    adjustedScale = std::max(0.8, std::min(1.8, adjustedScale));
    
    return adjustedScale;
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
    double currentDPIScale = getPlatformAwareDPIScale();
    if (std::abs(currentDPIScale - dpiScale) > 0.1) {
        dpiScale = currentDPIScale;
        // 更新字体大小和线宽 - 使用跨平台优化的字体大小
        if (textActor) {
            int optimizedFontSize = getOptimizedFontSize(18);
            textActor->GetTextProperty()->SetFontSize(optimizedFontSize);
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
    
    // 计算ScaleBar在窗口底部居中的位置
    double bottomMargin = 25.0 * dpiScale; // 底部边距
    double centerX = static_cast<double>(winW) / 2.0; // 窗口中心X坐标
    double bottomY = bottomMargin; // 底部Y坐标
    
    // 计算ScaleBar的起始和结束位置（居中）
    double p1[3] = {centerX - static_cast<double>(barPixelLen) / 2.0, bottomY, 0.0}; // 线条左端点
    double p2[3] = {centerX + static_cast<double>(barPixelLen) / 2.0, bottomY, 0.0}; // 线条右端点
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
            lineSource->SetPoint2(p2[0], p2[1], 0.0);
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
                
                rightSource->SetPoint1(p2[0], p2[1], 0.0);
                rightSource->SetPoint2(p2[0], p2[1] + tickLength, 0.0);
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
    double textX = centerX; // 使用窗口中心X坐标，确保文本居中
    double textY = bottomY - 10.0 * dpiScale; // 线条下方
    textActor->SetPosition(textX, textY);
} 