// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <vtkSmartPointer.h>

#include "abstractwidgetobserver.h"

class vtkPlanes;
namespace VtkUtils {

class QPCL_ENGINE_LIB_API BoxWidgetObserver : public AbstractWidgetObserver {
    Q_OBJECT
public:
    explicit BoxWidgetObserver(QObject* parent = nullptr);

signals:
    void planesChanged(vtkPlanes* planes);

protected:
    void Execute(vtkObject* caller, unsigned long eventId, void* callData);

    vtkSmartPointer<vtkPlanes> m_planes;
};

}  // namespace VtkUtils
