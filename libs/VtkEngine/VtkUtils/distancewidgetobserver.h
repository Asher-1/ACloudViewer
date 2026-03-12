// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/// @file distancewidgetobserver.h
/// @brief Observer for distance widget that emits distance and endpoint
/// positions.

#include "abstractwidgetobserver.h"

namespace VtkUtils {

/// @class DistanceWidgetObserver
/// @brief Observes distance widget EndInteractionEvent and emits distance and
/// world/display points.
class QVTK_ENGINE_LIB_API DistanceWidgetObserver
    : public AbstractWidgetObserver {
    Q_OBJECT
public:
    explicit DistanceWidgetObserver(QObject* parent = nullptr);

signals:
    void distanceChanged(double dist);
    void worldPoint1Changed(double* pos);
    void worldPoint2Changed(double* pos);
    void displayPoint1Changed(double* pos);
    void displayPoint2Changed(double* pos);

protected:
    void Execute(vtkObject* caller, unsigned long eventId, void* callData);
};

}  // namespace VtkUtils
