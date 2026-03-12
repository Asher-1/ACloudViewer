// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/// @file linewidgetobserver.h
/// @brief Observer for vtkLineWidget that emits endpoint positions.

#include "abstractwidgetobserver.h"

class vtkLineWidget;

namespace VtkUtils {

/// @class LineWidgetObserver
/// @brief Observes vtkLineWidget EndInteractionEvent and emits the two endpoint
/// positions.
class QVTK_ENGINE_LIB_API LineWidgetObserver : public AbstractWidgetObserver {
    Q_OBJECT
public:
    explicit LineWidgetObserver(QObject* parent = 0);

signals:
    void pointsChanged(double* point1, double* point2);

protected:
    void Execute(vtkObject* caller, unsigned long eventId, void*);
};

}  // namespace VtkUtils
