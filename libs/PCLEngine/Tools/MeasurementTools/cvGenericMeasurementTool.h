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
#include <QString>
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
        if (pos) {
            pos[0] = pos[1] = pos[2] = 0.0;
        }
    }

    //! Get point 2 coordinates
    virtual void getPoint2(double pos[3]) const {
        if (pos) {
            pos[0] = pos[1] = pos[2] = 0.0;
        }
    }

    //! Get center point coordinates (for angle/protractor)
    virtual void getCenter(double pos[3]) const {
        if (pos) {
            pos[0] = pos[1] = pos[2] = 0.0;
        }
    }

    //! Set point 1 coordinates
    virtual void setPoint1(double pos[3]) { /* not impl */ }

    //! Set point 2 coordinates
    virtual void setPoint2(double pos[3]) { /* not impl */ }

    //! Set center point coordinates (for angle/protractor)
    virtual void setCenter(double pos[3]) { /* not impl */ }

    //! Set measurement color (RGB values in range [0.0, 1.0])
    virtual void setColor(double r, double g, double b) { /* not impl */ }

    //! Lock tool interaction (disable VTK widget interaction and UI controls)
    virtual void lockInteraction() { /* not impl */ }

    //! Unlock tool interaction (enable VTK widget interaction and UI controls)
    virtual void unlockInteraction() { /* not impl */ }

    //! Set instance label suffix (e.g., "#1", "#2") for display in 3D view
    virtual void setInstanceLabel(const QString& label) { /* not impl */ }

    //! Set font family for measurement labels (e.g., "Arial", "Times New Roman")
    virtual void setFontFamily(const QString& family);

    //! Set font size for measurement labels
    virtual void setFontSize(int size);

    //! Set font bold state for measurement labels
    virtual void setBold(bool bold);

    //! Set font italic state for measurement labels
    virtual void setItalic(bool italic);

    //! Set font shadow state for measurement labels
    virtual void setShadow(bool shadow);

    //! Set font opacity for measurement labels (0.0 to 1.0)
    virtual void setFontOpacity(double opacity);

    //! Set font color for measurement labels (RGB values 0.0-1.0)
    virtual void setFontColor(double r, double g, double b);

    //! Set horizontal justification for measurement labels ("Left", "Center", "Right")
    virtual void setHorizontalJustification(const QString& justification);

    //! Set vertical justification for measurement labels ("Top", "Center", "Bottom")
    virtual void setVerticalJustification(const QString& justification);

protected:
    //! Apply font properties to VTK text properties
    //! Must be implemented by derived classes to apply to their specific VTK actors
    virtual void applyFontProperties() = 0;

public:
    void setUpViewer(PclUtils::PCLVis* viewer);
    void setInteractor(vtkRenderWindowInteractor* interactor);
    inline vtkRenderWindowInteractor* getInteractor() { return m_interactor; }
    inline vtkRenderer* getRenderer() { return m_renderer; }

    //! Setup keyboard shortcuts for point picking (call after linking with VTK
    //! widget)
    void setupShortcuts(QWidget* vtkWidget);

    //! Disable all keyboard shortcuts (call before tool destruction)
    void disableShortcuts();

    //! Clear selection cache in all picking helpers (call when scene/camera
    //! changes)
    void clearPickingCache();

protected:
    virtual void modelReady();
    virtual void dataChanged() { /* not impl */ }

    void safeOff(vtkAbstractWidget* widget);

    virtual void initTool() { /* not impl */ }
    virtual void createUi() { /* not impl */ }

    //! Setup keyboard shortcuts for point picking
    //! Override in derived classes to add specific shortcuts
    //! @param vtkWidget The VTK render window widget to bind shortcuts to
    virtual void setupPointPickingShortcuts(QWidget* vtkWidget) {
        Q_UNUSED(vtkWidget);
    }

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

    //! VTK widget reference for creating shortcuts (saved from linkWith)
    QWidget* m_vtkWidget = nullptr;

    //! Font properties for measurement labels (shared by all tools)
    QString m_fontFamily = "Arial";
    int m_fontSize = 6;  // Default font size for better readability
    double m_fontColor[3] = {1.0, 1.0, 1.0};  // White by default
    bool m_fontBold = false;
    bool m_fontItalic = false;
    bool m_fontShadow = true;
    double m_fontOpacity = 1.0;
    QString m_horizontalJustification = "Left";
    QString m_verticalJustification = "Bottom";
};
