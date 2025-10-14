// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef SPHEREWIDGETOBSERVER_H
#define SPHEREWIDGETOBSERVER_H

#include "abstractwidgetobserver.h"

namespace VtkUtils {

class QPCL_ENGINE_LIB_API SphereWidgetObserver : public AbstractWidgetObserver {
    Q_OBJECT
public:
    explicit SphereWidgetObserver(QObject* parent = 0);

signals:
    void centerChanged(double* center);
    void radiusChanged(double radius);

protected:
    void Execute(vtkObject* caller, unsigned long eventId, void* callData);
};

}  // namespace VtkUtils

#endif  // SPHEREWIDGETOBSERVER_H
