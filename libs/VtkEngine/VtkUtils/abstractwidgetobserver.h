// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/// @file abstractwidgetobserver.h
/// @brief Base class for VTK interactor widget observers that emit Qt signals.

#include <vtkCommand.h>

#include <QObject>

#include "qVTK.h"

class vtkInteractorObserver;
namespace VtkUtils {

/// @class AbstractWidgetObserver
/// @brief Base observer for VTK interactor widgets; observes
/// EndInteractionEvent and emits Qt signals.
class QVTK_ENGINE_LIB_API AbstractWidgetObserver : public QObject,
                                                   public vtkCommand {
    Q_OBJECT
public:
    explicit AbstractWidgetObserver(QObject* parent = 0);
    virtual ~AbstractWidgetObserver();

    /// @param widget VTK interactor observer to attach and observe
    void attach(vtkInteractorObserver* widget);

protected:
    virtual void Execute(vtkObject* caller,
                         unsigned long eventId,
                         void* callData) = 0;

protected:
    vtkInteractorObserver* m_widget = nullptr;
};

}  // namespace VtkUtils
