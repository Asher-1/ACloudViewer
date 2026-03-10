// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/// @file pointwidgetobserver.h
/// @brief Observer for point widget that emits position changes.

#include "abstractwidgetobserver.h"

namespace VtkUtils {

/// @class PointWidgetObserver
/// @brief Observes point widget EndInteractionEvent and emits the new position.
class QVTK_ENGINE_LIB_API PointWidgetObserver : public AbstractWidgetObserver {
    Q_OBJECT
public:
    explicit PointWidgetObserver(QObject* parent = nullptr);

signals:
    void positionChanged(double* position);

protected:
    void Execute(vtkObject* caller, unsigned long eventId, void* callData);
};

}  // namespace VtkUtils
