// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef ABSTRACTWIDGETOBSERVER_H
#define ABSTRACTWIDGETOBSERVER_H

#include <vtkCommand.h>

#include <QObject>

#include "../qPCL.h"

class vtkInteractorObserver;
namespace VtkUtils {

class QPCL_ENGINE_LIB_API AbstractWidgetObserver : public QObject,
                                                   public vtkCommand {
    Q_OBJECT
public:
    explicit AbstractWidgetObserver(QObject* parent = 0);
    virtual ~AbstractWidgetObserver();

    void attach(vtkInteractorObserver* widget);

protected:
    virtual void Execute(vtkObject* caller,
                         unsigned long eventId,
                         void* callData) = 0;

protected:
    vtkInteractorObserver* m_widget = nullptr;
};

}  // namespace VtkUtils

#endif  // ABSTRACTWIDGETOBSERVER_H
