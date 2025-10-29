// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <vtkSmartPointer.h>

#include <QWidget>

#include "ui_basewidgetwindow.h"

// namespace Ui
//{
//     class BaseWidgetWindow;
// }

namespace VtkUtils {
class VtkWidget;
}

class vtkActor;
class vtkAbstractWidget;
class ccHObject;
class BaseWidgetWindow : public QWidget {
    Q_OBJECT

public:
    explicit BaseWidgetWindow(QWidget* parent = nullptr);
    ~BaseWidgetWindow();

    virtual void createWidget() = 0;

    bool setInput(const ccHObject* obj);
    ccHObject* getOutput() const;

protected:
    template <class T>
    void setupConfigWidget(T* ui) {
        QWidget* configWidget = new QWidget(this);
        ui->setupUi(configWidget);
        m_ui->setupUi(this);
        m_ui->configLayout->addWidget(configWidget);
    }

    virtual void createUi();

protected:
    Ui::BaseWidgetWindow* m_ui = nullptr;
    VtkUtils::VtkWidget* m_vtkWidget = nullptr;
    vtkSmartPointer<vtkActor> m_theActor;
};
