// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef MOVEACTORINTERACTORSTYLE_H
#define MOVEACTORINTERACTORSTYLE_H

#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkSmartPointer.h>

#include <QObject>

#include "../qPCL.h"

namespace VtkUtils {

class QPCL_ENGINE_LIB_API MoveActorInteractorStyle
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

    void setHighlightActor(bool highlight);
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
#endif  // MOVEACTORINTERACTORSTYLE_H
