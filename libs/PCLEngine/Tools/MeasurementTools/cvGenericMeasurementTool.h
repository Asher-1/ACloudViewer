// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "ui_cvGenericMeasurementToolDlg.h"

// CV_CORE_LIB
#include <CVGeom.h>
#include <CVLog.h>
#include <CVTools.h>

// VTK
#include <vtkSmartPointer.h>

// QT
#include <QList>
#include <QObject>
#include <QWidget>

namespace PclUtils {
class PCLVis;
}
class ecvGenericVisualizer3D;
class cvPointPickingHelper;

class vtkActor;
class vtkProp;
class vtkRenderWindowInteractor;
class vtkRenderer;
class vtkAbstractWidget;

class ccHObject;

class cvGenericMeasurementTool : public QWidget {
    Q_OBJECT
signals:
    //! Signal sent when the measurement value changes
    void measurementValueChanged();
    
    //! Signal sent when point picking is requested
    //! @param pointIndex: 1=point1, 2=point2, 3=center
    void pointPickingRequested(int pointIndex);
    
    //! Signal sent when point picking is cancelled
    void pointPickingCancelled();
    
public:
    explicit cvGenericMeasurementTool(QWidget* parent = nullptr);
    virtual ~cvGenericMeasurementTool();

    virtual void start();
    virtual void update();
    virtual void reset();
    virtual ccHObject* getOutput();

    virtual bool initModel();
    virtual bool setInput(ccHObject* obj);

    virtual void showWidget(bool state) { /* not impl */ }
    virtual void clearAllActor();

    //! Get measurement value (distance or angle)
    virtual double getMeasurementValue() const { return 0.0; }
    
    //! Get point 1 coordinates
    virtual void getPoint1(double pos[3]) const { 
        if (pos) { pos[0] = pos[1] = pos[2] = 0.0; }
    }
    
    //! Get point 2 coordinates
    virtual void getPoint2(double pos[3]) const { 
        if (pos) { pos[0] = pos[1] = pos[2] = 0.0; }
    }
    
    //! Get center point coordinates (for angle/protractor)
    virtual void getCenter(double pos[3]) const { 
        if (pos) { pos[0] = pos[1] = pos[2] = 0.0; }
    }
    
    //! Set point 1 coordinates
    virtual void setPoint1(double pos[3]) { /* not impl */ }
    
    //! Set point 2 coordinates
    virtual void setPoint2(double pos[3]) { /* not impl */ }
    
    //! Set center point coordinates (for angle/protractor)
    virtual void setCenter(double pos[3]) { /* not impl */ }

public:
    void setUpViewer(PclUtils::PCLVis* viewer);
    void setInteractor(vtkRenderWindowInteractor* interactor);
    inline vtkRenderWindowInteractor* getInteractor() { return m_interactor; }
    inline vtkRenderer* getRenderer() { return m_renderer; }
    
    //! Setup keyboard shortcuts for point picking (call after linking with VTK widget)
    void setupShortcuts(QWidget* vtkWidget);

protected:
    virtual void modelReady();
    virtual void dataChanged() { /* not impl */ }

    void safeOff(vtkAbstractWidget* widget);

    virtual void initTool() { /* not impl */ }
    virtual void createUi() { /* not impl */ }
    
    //! Setup keyboard shortcuts for point picking
    //! Override in derived classes to add specific shortcuts
    //! @param vtkWidget The VTK render window widget to bind shortcuts to
    virtual void setupPointPickingShortcuts(QWidget* vtkWidget) { Q_UNUSED(vtkWidget); }
    
    //! Update point picking helpers with current interactor/renderer
    void updatePickingHelpers();

    void addActor(const vtkSmartPointer<vtkProp> actor);
    void removeActor(const vtkSmartPointer<vtkProp> actor);

protected:
    Ui::GenericMeasurementToolDlg* m_ui = nullptr;

    std::string m_id;
    ccHObject* m_entity = nullptr;
    PclUtils::PCLVis* m_viewer = nullptr;
    vtkRenderWindowInteractor* m_interactor = nullptr;
    vtkRenderer* m_renderer = nullptr;

    vtkSmartPointer<vtkActor> m_modelActor;
    
    //! List of point picking helpers for keyboard shortcuts
    QList<cvPointPickingHelper*> m_pickingHelpers;
};

