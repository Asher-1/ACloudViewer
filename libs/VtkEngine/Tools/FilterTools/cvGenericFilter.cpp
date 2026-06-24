// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

/**
 * @file cvGenericFilter.cpp
 * @brief Implementation of generic VTK filter base class.
 */

#include "cvGenericFilter.h"

#ifdef _MSC_VER
#pragma warning(disable : 4996)  // Use of [[deprecated]] feature
#endif

// LOCAL
#include <Converters/Vtk2Cc.h>

#include "Visualization/VtkVis.h"
#include "VtkUtils/vtkutils.h"

// CV_CORE_LIB
#include <CVLog.h>

// CV_DB_LIB
#include <ecvBBox.h>
#include <ecvHObject.h>
#include <ecvHObjectCaster.h>
#include <ecvMesh.h>
#include <ecvPointCloud.h>
#include <ecvPolyline.h>
#include <ecvRepresentationManager.h>
#include <ecvViewManager.h>
#include <ecvViewRepresentation.h>

// VTK
#include <VTKExtensions/Views/vtkPVAxesActor.h>
#include <VTKExtensions/Views/vtkPVLODActor.h>
#include <vtk3DWidget.h>
#include <vtkActor.h>
#include <vtkCallbackCommand.h>
#include <vtkCommand.h>
#include <vtkDataArray.h>
#include <vtkDataObject.h>
#include <vtkDataSet.h>
#include <vtkDataSetMapper.h>
#include <vtkFieldData.h>
#include <vtkLookupTable.h>
#include <vtkMapper.h>
#include <vtkOutlineFilter.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkProp.h>
#include <vtkProperty.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkRendererCollection.h>
#include <vtkScalarBarActor.h>
#include <vtkTextProperty.h>
#include <vtkUnstructuredGrid.h>

#include <QTimer>

cvGenericFilter::cvGenericFilter(QWidget* parent)
    : QWidget(parent),
      m_ui(new Ui::GenericFilterDlg),
      m_keepMode(false),
      m_negative(false),
      m_meshMode(false),
      m_preview(true) {
    setWindowTitle(tr("GenericFilter"));
    connect(&ecvViewManager::instance(), &ecvViewManager::doubleButtonClicked,
            this, &cvGenericFilter::onDoubleClick);
}

cvGenericFilter::~cvGenericFilter() {
    removeDisplayEffectObserver();
    restoreInputEntityVisibility();
    VtkUtils::vtkSafeDelete(m_dataObject);
    delete m_ui;
}

void cvGenericFilter::onDoubleClick(int x, int y) { applyDisplayEffect(); }

////////////////////Initialization///////////////////////////
void cvGenericFilter::start() {
    // call child function
    modelReady();

    // update data
    dataChanged();

    // according to data type
    initFilter();

    installDisplayEffectObserver();

    // update screen
    update();
}

bool cvGenericFilter::setInput(ccHObject* obj) {
    restoreInputEntityVisibility();
    m_entity = obj;
    m_id = m_entity->getViewId().toStdString();

    if (m_entity->isKindOf(CV_TYPES::POINT_CLOUD)) {
        m_meshMode = false;
    } else if (m_entity->isKindOf(CV_TYPES::MESH)) {
        m_meshMode = true;
    } else {
        CVLog::Error("only mesh and point cloud are supported!");
        return false;
    }

    if (!initModel()) {
        return false;
    }

    updateUi();
    return true;
}

bool cvGenericFilter::initModel() {
    if (m_dataObject) {
        m_dataObject->Delete();
        m_dataObject = nullptr;
    }
    assert(m_viewer);
    m_modelActor = m_viewer->getActorById(m_id);
    if (!m_modelActor) {
        return false;
    }

    vtkMapper* mapper = m_modelActor->GetMapper();
    if (!mapper) {
        return false;
    }

    vtkPolyData* polydata = vtkPolyData::SafeDownCast(mapper->GetInput());
    if (!polydata) {
        return false;
    }

    int npoints = static_cast<int>(polydata->GetNumberOfPoints());
    if (!m_meshMode) {
        // Point clouds use mapper clipping planes for Opaque preview
        // (GPU-based, no geometry copy), so preview stays enabled regardless of
        // size.
        m_preview = true;
    } else if (npoints > MAX_PREVIEW_NUMBER) {
        m_preview = false;
    } else {
        m_preview = true;
    }

    if (!m_dataObject) {
        m_dataObject = vtkPolyData::New();
    }
    m_dataObject->DeepCopy(polydata);

    if (!m_resultData) {
        m_resultData.TakeReference(vtkPolyData::New());
    }

    return true;
}

ccHObject* cvGenericFilter::getOutput() {
    vtkDataObject* vtkData = resultData();
    if (!vtkData) {
        return nullptr;
    }

    vtkPolyData* polydata = vtkPolyData::SafeDownCast(vtkData);
    if (!polydata) {
        return nullptr;
    }

    Converters::Vtk2CcOptions options;
    options.sourceEntity = m_entity;

    ccHObject* result =
            Converters::Vtk2Cc::Convert(polydata, m_meshMode, options);
    if (!result && m_meshMode) {
        CVLog::Warning(QString("try to save in cloud format"));
        result = Converters::Vtk2Cc::Convert(polydata, false, options);
    }

    if (!result || !m_entity) {
        return result;
    }

    auto propagateGlobalShiftScale = [](ccHObject* output,
                                        const ccPointCloud* sourceCloud) {
        if (!output || !sourceCloud) {
            return;
        }

        if (ccPointCloud* outputCloud = ccHObjectCaster::ToPointCloud(output)) {
            outputCloud->setGlobalScale(sourceCloud->getGlobalScale());
            outputCloud->setGlobalShift(sourceCloud->getGlobalShift());
            return;
        }

        if (ccMesh* outputMesh = ccHObjectCaster::ToMesh(output)) {
            ccPointCloud* outputVertices = ccHObjectCaster::ToPointCloud(
                    outputMesh->getAssociatedCloud());
            if (outputVertices) {
                outputVertices->setGlobalScale(sourceCloud->getGlobalScale());
                outputVertices->setGlobalShift(sourceCloud->getGlobalShift());
            }
        }
    };

    if (m_entity->isKindOf(CV_TYPES::POINT_CLOUD)) {
        ccPointCloud* sourceCloud = ccHObjectCaster::ToPointCloud(m_entity);
        propagateGlobalShiftScale(result, sourceCloud);
    } else if (m_entity->isKindOf(CV_TYPES::MESH)) {
        ccMesh* sourceMesh = ccHObjectCaster::ToMesh(m_entity);
        ccPointCloud* sourceCloud =
                sourceMesh ? ccHObjectCaster::ToPointCloud(
                                     sourceMesh->getAssociatedCloud())
                           : nullptr;
        propagateGlobalShiftScale(result, sourceCloud);
    }

    const QString sourceName = m_entity->getName();
    if (!sourceName.isEmpty()) {
        result->setName(sourceName + ".filtered");
    }

    return result;
}

void cvGenericFilter::getOutput(std::vector<ccHObject*>& outputSlices,
                                std::vector<ccPolyline*>& outputContours) {
    outputContours.clear();
    ccHObject* slices = getOutput();
    if (slices) {
        outputSlices.push_back(slices);
    }
}

void cvGenericFilter::modelReady() {
    showOutline(false);
    if (auto* v = ecvViewManager::instance().getEffectiveView()) {
        v->updateCamera();
    }
}

void cvGenericFilter::setUpViewer(Visualization::VtkVis* viewer) {
    if (!viewer) return;
    m_viewer = viewer;
    setInteractor(viewer->getRenderWindowInteractor());
}

void cvGenericFilter::getInteractorInfos(ccBBox& bbox, ccGLMatrixd& trans) {
    getInteractorBounds(bbox);
    getInteractorTransformation(trans);
}

void cvGenericFilter::colorsChanged() {
    vtkSmartPointer<vtkLookupTable> lut =
            createLookupTable(m_scalarMin, m_scalarMax);
    if (m_scalarBar) m_scalarBar->SetLookupTable(lut);
    if (m_modelActor) m_modelActor->GetMapper()->SetLookupTable(lut);
    if (m_filterActor) m_filterActor->GetMapper()->SetLookupTable(lut);
    if (m_viewer) update();
}

////////////////////Visualization///////////////////////////
void cvGenericFilter::update() {
    QWidget::update();
    if (auto* w = ecvViewManager::instance().activeWidget()) {
        w->update();
    }
    if (m_viewer) {
        if (auto rw = m_viewer->getRenderWindow()) {
            rw->Render();
        }
    } else if (m_interactor) {
        m_interactor->Render();
    }
}

void cvGenericFilter::reset() {
    dataChanged();
    update();
}

void cvGenericFilter::restoreOrigin() {
    if (!m_viewer) return;
    if (initModel()) {
        start();
    }
}

void cvGenericFilter::setDisplayEffect(cvGenericFilter::DisplayEffect effect) {
    if (m_displayEffect != effect) {
        m_displayEffect = effect;
        apply();
    }
}

cvGenericFilter::DisplayEffect cvGenericFilter::displayEffect() const {
    return m_displayEffect;
}

void cvGenericFilter::updateSize() {
    adjustSize();
    QWidget* widget = topLevelWidget();
    if (widget) {
        widget->adjustSize();
        if (widget->topLevelWidget()) {
            widget->topLevelWidget()->adjustSize();
        }
    }
}

void cvGenericFilter::UpdateScalarRange() {
    double scalarRange[2];
    if (isValidPolyData()) {
        vtkPolyData* polyData = vtkPolyData::SafeDownCast(m_dataObject);
        polyData->GetScalarRange(scalarRange);
    } else if (isValidDataSet()) {
        vtkDataSet* dataSet = vtkDataSet::SafeDownCast(m_dataObject);
        dataSet->GetScalarRange(scalarRange);
    }
    setScalarRange(scalarRange[0], scalarRange[1]);
}

void cvGenericFilter::hideInputEntityForOpaquePreview(bool hide) {
    if (!m_entity) return;

    if (hide) {
        auto* view = ecvViewManager::instance().getEffectiveView();
        if (!view) return;

        auto& repMgr = ecvRepresentationManager::instance();
        auto* rep = repMgr.ensureRepresentation(m_entity, view);
        if (!rep) return;

        if (!m_entityVisibilityOverridden || m_entityVisibilityView != view) {
            if (m_entityVisibilityOverridden) {
                restoreInputEntityVisibility();
            }
            m_entityHadVisibilityOverride = rep->hasVisibilityOverride();
            m_entityOriginalVisibility = rep->isVisible();
            m_entityVisibilityView = view;
            m_entityVisibilityOverridden = true;
        }

        rep->setVisible(false);
        if (m_modelActor) {
            m_modelActor->SetVisibility(0);
        }
        return;
    }

    restoreInputEntityVisibility();
}

void cvGenericFilter::restoreInputEntityVisibility() {
    if (m_entityVisibilityOverridden && m_entity && m_entityVisibilityView) {
        auto* rep = ecvRepresentationManager::instance().getRepresentation(
                m_entity, m_entityVisibilityView);
        if (rep) {
            if (m_entityHadVisibilityOverride) {
                rep->setVisible(m_entityOriginalVisibility);
            } else {
                rep->clearVisibilityOverride();
            }
        }
    }
    m_entityVisibilityOverridden = false;
    m_entityHadVisibilityOverride = false;
    m_entityVisibilityView = nullptr;
}

void cvGenericFilter::applyModelDisplayEffect() {
    if (m_viewer && !m_id.empty()) {
        if (vtkActor* actor = m_viewer->getActorById(m_id)) {
            m_modelActor = actor;
        }
    }

    if (m_displayEffect != Opaque) {
        hideInputEntityForOpaquePreview(false);
    }

    if (!m_modelActor) {
        return;
    }

    switch (m_displayEffect) {
        case Transparent:
            hideInputEntityForOpaquePreview(false);
            m_modelActor->GetProperty()->SetOpacity(0.3);
            m_modelActor->SetVisibility(1);
            m_modelActor->GetProperty()->SetRepresentationToSurface();
            break;

        case Opaque:
            m_modelActor->GetProperty()->SetOpacity(1.0);
            m_modelActor->SetVisibility(0);
            hideInputEntityForOpaquePreview(true);
            break;

        case Points:
            hideInputEntityForOpaquePreview(false);
            m_modelActor->GetProperty()->SetOpacity(1.0);
            m_modelActor->SetVisibility(1);
            m_modelActor->GetProperty()->SetRepresentationToPoints();
            break;

        case Wireframe:
            hideInputEntityForOpaquePreview(false);
            m_modelActor->GetProperty()->SetOpacity(1.0);
            m_modelActor->SetVisibility(1);
            m_modelActor->GetProperty()->SetRepresentationToWireframe();
            break;
    }

    vtkSmartPointer<vtkLookupTable> lut =
            createLookupTable(m_scalarMin, m_scalarMax);
    m_modelActor->GetMapper()->SetLookupTable(lut);
}

bool cvGenericFilter::restoreDisplayEffectIfNeeded() {
    if (m_viewer && !m_id.empty()) {
        if (vtkActor* actor = m_viewer->getActorById(m_id)) {
            m_modelActor = actor;
        }
    }
    if (!m_modelActor) {
        return false;
    }

    vtkProperty* prop = m_modelActor->GetProperty();
    const int representation = prop->GetRepresentation();
    const double opacity = prop->GetOpacity();
    const int visibility = m_modelActor->GetVisibility();
    bool needsRestore = false;

    switch (m_displayEffect) {
        case Transparent:
            needsRestore = opacity != 0.3 || representation != VTK_SURFACE ||
                           visibility != 1;
            break;
        case Points:
            needsRestore = opacity != 1.0 || representation != VTK_POINTS ||
                           visibility != 1;
            break;
        case Wireframe:
            needsRestore = opacity != 1.0 || representation != VTK_WIREFRAME ||
                           visibility != 1;
            break;
        case Opaque:
            needsRestore = opacity != 1.0 || visibility != 0 ||
                           !m_entityVisibilityOverridden;
            break;
    }

    if (!needsRestore) {
        return false;
    }

    applyModelDisplayEffect();
    return true;
}

void cvGenericFilter::installDisplayEffectObserver() {
    removeDisplayEffectObserver();
    if (!m_viewer) {
        return;
    }

    auto renderers = m_viewer->getRendererCollection();
    vtkRenderer* renderer = renderers ? renderers->GetFirstRenderer() : nullptr;
    if (!renderer) {
        return;
    }

    m_renderEndCallback = vtkSmartPointer<vtkCallbackCommand>::New();
    m_renderEndCallback->SetClientData(this);
    m_renderEndCallback->SetCallback(&cvGenericFilter::OnRenderEnd);
    // Use StartEvent instead of EndEvent: the draw pipeline
    // (drawMesh / setMeshRenderingMode / setMeshOpacity) runs BEFORE
    // rw->Render(). StartEvent fires at the beginning of the render pass,
    // allowing us to restore the correct properties before the frame is
    // composited — eliminating the one-frame flash that EndEvent caused.
    m_renderEndObserverTag =
            renderer->AddObserver(vtkCommand::StartEvent, m_renderEndCallback);
}

void cvGenericFilter::removeDisplayEffectObserver() {
    if (m_renderEndObserverTag == 0 || !m_viewer) {
        m_renderEndObserverTag = 0;
        m_renderEndCallback = nullptr;
        return;
    }

    auto renderers = m_viewer->getRendererCollection();
    vtkRenderer* renderer = renderers ? renderers->GetFirstRenderer() : nullptr;
    if (renderer) {
        renderer->RemoveObserver(m_renderEndObserverTag);
    }
    m_renderEndObserverTag = 0;
    m_renderEndCallback = nullptr;
}

void cvGenericFilter::OnRenderEnd(vtkObject* /*caller*/,
                                  unsigned long /*eid*/,
                                  void* clientData,
                                  void* /*callData*/) {
    auto* self = static_cast<cvGenericFilter*>(clientData);
    if (!self || self->m_applyingDisplayEffect) {
        return;
    }

    self->m_applyingDisplayEffect = true;
    self->restoreDisplayEffectIfNeeded();
    self->m_applyingDisplayEffect = false;
}

void cvGenericFilter::applyDisplayEffect() {
    applyModelDisplayEffect();

    if (m_filterActor) {
        m_filterActor->SetVisibility(1);
        m_filterActor->GetProperty()->SetOpacity(1.0);
        if (m_meshMode) {
            m_filterActor->GetProperty()->SetRepresentationToSurface();
        } else {
            m_filterActor->GetProperty()->SetRepresentationToPoints();
            if (m_modelActor) {
                m_filterActor->GetProperty()->SetPointSize(
                        m_modelActor->GetProperty()->GetPointSize());
            }
        }
        if (auto* mapper = m_filterActor->GetMapper()) {
            vtkSmartPointer<vtkLookupTable> lut =
                    createLookupTable(m_scalarMin, m_scalarMax);
            mapper->SetLookupTable(lut);
            mapper->SetScalarRange(m_scalarMin, m_scalarMax);
            mapper->Update();
        }
    }

    update();
}

void cvGenericFilter::scheduleDisplayEffectRefresh() {
    QTimer::singleShot(0, this, [this]() { applyDisplayEffect(); });
    QTimer::singleShot(30, this, [this]() { applyDisplayEffect(); });
}

void cvGenericFilter::showScalarBar(bool show) {
    if (!m_scalarBar) {
        vtkSmartPointer<vtkLookupTable> lut =
                createLookupTable(m_scalarMin, m_scalarMax);
        lut->Build();

        m_scalarBar = vtkScalarBarActor::New();
        m_scalarBar->SetPosition(0.0, 0.0);
        m_scalarBar->SetWidth(.1);   // fraction of window width
        m_scalarBar->SetHeight(.8);  // fraction of window height
        m_scalarBar->GetTitleTextProperty()->SetFontSize(3);
        m_scalarBar->GetTitleTextProperty()->SetBold(0);
        m_scalarBar->GetTitleTextProperty()->SetItalic(0);
        m_scalarBar->SetOrientationToVertical();
        m_scalarBar->SetTitle("Scalar");
        m_scalarBar->SetLookupTable(lut);
        addActor(m_scalarBar);
    }

    m_scalarBar->SetVisibility(show);
    update();
}

void cvGenericFilter::showOutline(bool show) {
    if (!m_dataObject) {
        CVLog::Error(QString(
                "cvGenericFilter::showOutline: null data object, quit."));
        return;
    }

    if (!m_outlineActor) {
        VTK_CREATE(vtkOutlineFilter, outline);
        outline->SetInputData(m_dataObject);

        VTK_CREATE(vtkPolyDataMapper, mapper);
        mapper->SetInputConnection(outline->GetOutputPort());

        VtkUtils::vtkInitOnce(m_outlineActor);
        m_outlineActor->SetMapper(mapper);
        m_outlineActor->PickableOff();
        addActor(m_outlineActor);
    }

    m_outlineActor->SetVisibility(show);
    update();
}

void cvGenericFilter::setOutlineColor(const QColor& clr) {
    if (!m_outlineActor) return;

    double vtkClr[3];
    Utils::vtkColor(clr, vtkClr);
    m_outlineActor->GetProperty()->SetColor(vtkClr);
    update();
}

void cvGenericFilter::setScalarBarColors(const QColor& clr1,
                                         const QColor& clr2) {
    if (m_color1 == clr1 && m_color2 == clr2) return;

    m_color1 = clr1;
    m_color2 = clr2;
    colorsChanged();
}

QColor cvGenericFilter::color1() const { return m_color1; }

QColor cvGenericFilter::color2() const { return m_color2; }

void cvGenericFilter::setScalarRange(double min, double max) {
    m_scalarMin = qMin(min, max);
    m_scalarMax = qMax(min, max);
}

double cvGenericFilter::scalarMin() const { return m_scalarMin; }

double cvGenericFilter::scalarMax() const { return m_scalarMax; }

vtkSmartPointer<vtkDataArray> cvGenericFilter::getActorScalars(
        vtkSmartPointer<vtkActor> actor) {
    if (!actor) return {};

    return actor->GetMapper()->GetInput()->GetPointData()->GetScalars();
}

int cvGenericFilter::getDefaultScalarInterpolationForDataSet(vtkDataSet* data) {
    vtkPolyData* polyData = vtkPolyData::SafeDownCast(
            data);  // Check that polyData != nullptr in case of segfault
    return (polyData &&
            polyData->GetNumberOfCells() != polyData->GetNumberOfVerts());
}

vtkSmartPointer<vtkLookupTable> cvGenericFilter::createLookupTable(double min,
                                                                   double max) {
    double hsv1[3];
    double hsv2[3];
    Utils::qColor2HSV(m_color1, hsv1);
    Utils::qColor2HSV(m_color2, hsv2);

    VTK_CREATE(vtkLookupTable, lut);
    lut->SetHueRange(hsv1[0], hsv2[0]);
    lut->SetSaturationRange(hsv1[1], hsv2[1]);
    lut->SetValueRange(hsv1[2], hsv2[2]);
    lut->SetTableRange(min, max);
    lut->Build();

    return lut;
}

void cvGenericFilter::addActor(const vtkSmartPointer<vtkProp> actor) {
    if (!m_viewer) return;
    m_viewer->addActorToRenderer(actor);
}

void cvGenericFilter::removeActor(const vtkSmartPointer<vtkProp> actor) {
    if (!m_viewer) return;
    m_viewer->removeActorFromRenderer(actor);
}

void cvGenericFilter::clearAllActor() {
    removeDisplayEffectObserver();
    restoreInputEntityVisibility();

    if (!m_viewer) return;

    if (m_modelActor) {
        m_modelActor->GetProperty()->SetOpacity(1.0);
        m_modelActor->SetVisibility(1);
        if (m_meshMode) {
            m_modelActor->GetProperty()->SetRepresentationToSurface();
        }
    }

    if (m_outlineActor) {
        removeActor(m_outlineActor);
    }

    if (m_scalarBar) {
        removeActor(m_scalarBar);
    }

    if (m_filterActor) {
        removeActor(m_filterActor);
    }
}

////////////////////Util Function///////////////////////////
template <class DataObject, class Mapper>
void cvGenericFilter::createActorFromData(vtkDataObject* dataObj) {
    if (!dataObj) return;

    m_dataObject = dataObj;
    DataObject* data = DataObject::SafeDownCast(dataObj);

    if (!data) {
        CVLog::Error(QString("ansys parsing: null data set."));
        return;
    }

    VTK_CREATE(Mapper, mapper);
    mapper->SetInputData(data);
    mapper->Update();

    //	VtkUtils::vtkInitOnce(m_modelActor);
    m_modelActor.TakeReference(vtkPVLODActor::New());
    m_modelActor->SetMapper(mapper);

    m_modelActor->GetProperty()->SetInterpolationToFlat();

    addActor(m_modelActor);
    modelReady();

    update();
    dataChanged();
}

void cvGenericFilter::setInteractor(vtkRenderWindowInteractor* interactor) {
    m_interactor = interactor;
}

void cvGenericFilter::setResultData(vtkSmartPointer<vtkDataObject> data) {
    m_resultData->DeepCopy(data);

    // VTK filter algorithms typically do NOT propagate field data (which stores
    // material names, texture library paths, dataset flags, etc.) from input to
    // output. Restore it from the original input so that Vtk2Cc conversion can
    // reconstruct materials and textures.
    if (m_dataObject && m_dataObject->GetFieldData() &&
        m_dataObject->GetFieldData()->GetNumberOfArrays() > 0) {
        vtkFieldData* srcFD = m_dataObject->GetFieldData();
        vtkFieldData* dstFD = m_resultData->GetFieldData();
        if (dstFD) {
            for (int i = 0; i < srcFD->GetNumberOfArrays(); ++i) {
                vtkAbstractArray* arr = srcFD->GetAbstractArray(i);
                if (arr && arr->GetName() &&
                    !dstFD->GetAbstractArray(arr->GetName())) {
                    dstFD->AddArray(arr);
                }
            }
        }
    }

    scheduleDisplayEffectRefresh();
}

vtkSmartPointer<vtkDataObject> cvGenericFilter::resultData() const {
    return m_resultData;
}

bool cvGenericFilter::isValidPolyData() const {
    return vtkPolyData::SafeDownCast(m_dataObject) != nullptr;
}

bool cvGenericFilter::isValidDataSet() const {
    return vtkDataSet::SafeDownCast(m_dataObject) != nullptr;
}

void cvGenericFilter::safeOff(vtk3DWidget* widget) {
    if (widget) widget->Off();
}
