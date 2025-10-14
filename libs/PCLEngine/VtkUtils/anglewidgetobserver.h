// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef ANGLEWIDGETOBSERVER_H
#define ANGLEWIDGETOBSERVER_H

#include "abstractwidgetobserver.h"

namespace VtkUtils {

class QPCL_ENGINE_LIB_API AngleWidgetObserver : public AbstractWidgetObserver {
    Q_OBJECT
public:
    explicit AngleWidgetObserver(QObject* parent = nullptr);

signals:
    void angleChanged(double angle);
    void worldPoint1Changed(double* pos);
    void worldPoint2Changed(double* pos);
    void worldCenterChanged(double* pos);
    void displayPoint1Changed(double* pos);
    void displayPoint2Changed(double* pos);
    void displayCenterChanged(double* pos);

protected:
    void Execute(vtkObject* caller, unsigned long eventId, void* callData);
};

}  // namespace VtkUtils
#endif  // ANGLEWIDGETOBSERVER_H
