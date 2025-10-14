// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "abstractwidgetobserver.h"

class vtkLineWidget;

namespace VtkUtils {

class QPCL_ENGINE_LIB_API LineWidgetObserver : public AbstractWidgetObserver {
    Q_OBJECT
public:
    explicit LineWidgetObserver(QObject* parent = 0);

signals:
    void pointsChanged(double* point1, double* point2);

protected:
    void Execute(vtkObject* caller, unsigned long eventId, void*);
};

}  // namespace VtkUtils
