// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "abstractwidgetobserver.h"

namespace VtkUtils {

class QPCL_ENGINE_LIB_API PointWidgetObserver : public AbstractWidgetObserver {
    Q_OBJECT
public:
    explicit PointWidgetObserver(QObject* parent = nullptr);

signals:
    void positionChanged(double* position);

protected:
    void Execute(vtkObject* caller, unsigned long eventId, void* callData);
};

}  // namespace VtkUtils
