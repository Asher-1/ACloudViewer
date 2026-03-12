// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/// @file moveactorinteractorstyle.h
/// @brief VTK interactor style for picking and moving actors with mouse.

#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkSmartPointer.h>

#include <QObject>

#include "qVTK.h"

namespace VtkUtils {

/// @class MoveActorInteractorStyle
/// @brief Trackball camera style extended for picking and moving actors; emits
/// actorMoved.
class QVTK_ENGINE_LIB_API MoveActorInteractorStyle
    : public QObject,
      public vtkInteractorStyleTrackballCamera {
    Q_OBJECT
public:
    static MoveActorInteractorStyle *New();
    vtkTypeMacro(MoveActorInteractorStyle, vtkInteractorStyleTrackballCamera);

    virtual void OnLeftButtonDown();
    virtual void OnLeftButtonUp();
    virtual void Rotate();
    virtual void Spin();
    virtual void OnMouseMove();
    virtual void OnChar();

    /// @param highlight Enable highlight on picked actor
    void setHighlightActor(bool highlight);
    /// @return true if highlight enabled
    bool highlightActor() const;

    // Todo: add some signals
signals:
    void actorMoved(vtkActor *actor);

protected:
    explicit MoveActorInteractorStyle(QObject *parent = 0);

protected:
    vtkSmartPointer<vtkActor> m_pickedActor;

    bool m_useHighlight = true;
    vtkSmartPointer<vtkActor> m_highlightActor;
};

}  // namespace VtkUtils
