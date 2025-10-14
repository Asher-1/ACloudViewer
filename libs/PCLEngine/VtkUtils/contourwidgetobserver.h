// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <vtkSmartPointer.h>

#include "abstractwidgetobserver.h"

class vtkPolyData;
namespace VtkUtils {

class QPCL_ENGINE_LIB_API ContourWidgetObserver
    : public AbstractWidgetObserver {
    Q_OBJECT
public:
    explicit ContourWidgetObserver(QObject* parent = nullptr);

signals:
    void dataChanged(vtkPolyData* data);

protected:
    void Execute(vtkObject* caller, unsigned long eventId, void* callData);

    vtkSmartPointer<vtkPolyData> m_polyData;
};

}  // namespace VtkUtils
