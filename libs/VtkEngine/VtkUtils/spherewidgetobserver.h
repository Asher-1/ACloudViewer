// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/**
 * @file spherewidgetobserver.h
 * @brief Observer for vtkSphereWidget interaction events.
 */

#include "abstractwidgetobserver.h"

namespace VtkUtils {

/**
 * @class SphereWidgetObserver
 * @brief Emits Qt signals when a VTK sphere widget's center or radius changes.
 */
class QVTK_ENGINE_LIB_API SphereWidgetObserver : public AbstractWidgetObserver {
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
