// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/// @file anglewidgetobserver.h
/// @brief Observer for vtkAngleWidget that emits angle and point positions.

#include "abstractwidgetobserver.h"

namespace VtkUtils {

/// @class AngleWidgetObserver
/// @brief Observes vtkAngleWidget EndInteractionEvent and emits angle (degrees)
/// and world/display positions.
class QVTK_ENGINE_LIB_API AngleWidgetObserver : public AbstractWidgetObserver {
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
