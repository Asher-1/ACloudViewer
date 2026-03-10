// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/**
 * @file cvGenericFilter.h
 * @brief Base class for VTK-based filters with UI and 3D interactors.
 */

#include "ui_cvGenericFilterDlg.h"

// CV_CORE_LIB
#include <CVGeom.h>
// CV_CORE_LIB
#include <CVLog.h>
#include <CVTools.h>

// VTK
#include <vtkSmartPointer.h>

// QT
#include <QObject>
#include <QWidget>

namespace Visualization {
class VtkVis;
}
class ecvGenericVisualizer3D;

class vtkActor;
class vtkProp;
class vtkDataSet;
class vtkDataObject;
class vtkDataArray;
class vtkScalarBarActor;
class vtkLODActor;
class vtk3DWidget;
class vtkLookupTable;
class vtkRenderWindowInteractor;

class ccBBox;
class ccPolyline;
class ccHObject;
class ccGLMatrixd;

/**
 * @class cvGenericFilter
 * @brief Abstract base for VTK filters with config UI and scalar visualization.
 */
class cvGenericFilter : public QWidget {
    Q_OBJECT
public:
    explicit cvGenericFilter(QWidget* parent = nullptr);
    virtual ~cvGenericFilter();

    virtual void apply() = 0;
    virtual void start();
    virtual void update();
    virtual void reset();
    virtual void restoreOrigin();
    /// @return Filter output as ccHObject, or nullptr.
    virtual ccHObject* getOutput();
    /// @param outputSlices Output slice objects.
    /// @param outputContours Output contour polylines.
    virtual void getOutput(std::vector<ccHObject*>& outputSlices,
                           std::vector<ccPolyline*>& outputContours);

    virtual bool initModel();
    /// @param obj Input entity.
    /// @return true if input was accepted.
    virtual bool setInput(ccHObject* obj);

    //! Shifts the current interactor
    virtual void shift(const CCVector3d& v) { /* not impl */ }

    virtual void updateUi() { /* not impl */ }
    virtual void showInteractor(bool state) { /* not impl */ }
    /// @param bbox Output bounding box of interactor.
    /// @param trans Output transformation of interactor.
    virtual void getInteractorInfos(ccBBox& bbox, ccGLMatrixd& trans);
    virtual void getInteractorTransformation(
            ccGLMatrixd& trans) { /* not impl */ }
    virtual void getInteractorBounds(ccBBox& bbox) { /* not impl */ }
    virtual void clearAllActor();

protected:
    virtual void modelReady();
    virtual void dataChanged() { /* not impl */ }
    virtual void colorsChanged();

public:
    /// @param viewer PCL visualizer.
    void setUpViewer(Visualization::VtkVis* viewer);
    /// @param show If true, show outline.
    void showOutline(bool show = true);
    void setNegative(bool state) { m_negative = state; }
    /// @param interactor VTK render window interactor.
    void setInteractor(vtkRenderWindowInteractor* interactor);
    inline vtkRenderWindowInteractor* getInteractor() { return m_interactor; }

protected:
    enum DisplayEffect { Opaque, Transparent, Points, Wireframe };
    void setDisplayEffect(DisplayEffect effect);
    DisplayEffect displayEffect() const;

    void safeOff(vtk3DWidget* widget);

    virtual void initFilter() { /* not impl */ }
    virtual void createUi() { /* not impl */ }

    void updateSize();

    void UpdateScalarRange();

    void applyDisplayEffect();
    void setScalarBarColors(const QColor& clr1, const QColor& clr2);
    QColor color1() const;
    QColor color2() const;

    /// @param min Minimum scalar value.
    /// @param max Maximum scalar value.
    void setScalarRange(double min, double max);
    double scalarMin() const;
    double scalarMax() const;

    /// @param actor VTK actor to get scalars from.
    /// @return Scalar array from actor, or nullptr.
    vtkSmartPointer<vtkDataArray> getActorScalars(
            vtkSmartPointer<vtkActor> actor);

    // Helper function called by child methods.
    // This function determines the default setting of
    // vtkMapper::InterpolateScalarsBeforeMapping. Return 0, interpolation off,
    // if data is a vtkPolyData that contains only vertices. Return 1,
    // interpolation on, for anything else.
    int getDefaultScalarInterpolationForDataSet(vtkDataSet* data);

    /// @param min Minimum scalar.
    /// @param max Maximum scalar.
    /// @return Lookup table for the scalar range.
    vtkSmartPointer<vtkLookupTable> createLookupTable(double min, double max);

    template <class ConfigClass>
    void setupConfigWidget(ConfigClass* cc) {
        QWidget* configWidget = new QWidget(this);
        cc->setupUi(configWidget);
        m_ui->setupUi(this);
        m_ui->configLayout->addWidget(configWidget);
        m_ui->groupBox->setTitle(configWidget->windowTitle() +
                                 tr(" Paramters"));
    }

    template <class DataObject, class Mapper>
    void createActorFromData(vtkDataObject* dataObj);

    /// @param show If true, show scalar bar.
    void showScalarBar(bool show = true);
    /// @param clr Outline color.
    void setOutlineColor(const QColor& clr);

    bool isValidPolyData() const;
    bool isValidDataSet() const;

    /// @param actor Actor to add to the scene.
    void addActor(const vtkSmartPointer<vtkProp> actor);
    /// @param actor Actor to remove.
    void removeActor(const vtkSmartPointer<vtkProp> actor);
    /// @param data Result data object.
    void setResultData(vtkSmartPointer<vtkDataObject> data);
    vtkSmartPointer<vtkDataObject> resultData() const;

protected slots:
    void onDoubleClick(int x, int y);

protected:
    Ui::GenericFilterDlg* m_ui = nullptr;

    DisplayEffect m_displayEffect = Opaque;
    vtkDataObject* m_dataObject = nullptr;
    vtkSmartPointer<vtkDataObject> m_resultData;

    bool m_keepMode;
    bool m_negative;
    bool m_meshMode;
    bool m_preview;
    std::string m_id;
    ccHObject* m_entity = nullptr;
    Visualization::VtkVis* m_viewer = nullptr;
    vtkRenderWindowInteractor* m_interactor = nullptr;

    vtkSmartPointer<vtkActor> m_modelActor;
    vtkSmartPointer<vtkLODActor> m_filterActor;
    vtkSmartPointer<vtkScalarBarActor> m_scalarBar;
    vtkSmartPointer<vtkActor> m_outlineActor;

    QColor m_color1 = Qt::blue;
    QColor m_color2 = Qt::red;
    double m_scalarMin = 0.0;
    double m_scalarMax = 1.0;
    static const int MAX_PREVIEW_NUMBER = 20e4;
};
