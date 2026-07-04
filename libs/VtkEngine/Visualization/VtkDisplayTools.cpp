// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

/**
 * @file VtkDisplayTools.cpp
 * @brief Implementation of display tools for VTK object registration and
 * visualization.
 */

#include "VtkDisplayTools.h"

#include "vtkViewManagerSetupRelay.h"

// PCLModules
#include <Converters/Cc2Vtk.h>

// CV_CORE_LIB
#include <CVGeom.h>
#include <CVTools.h>
#include <ecvGLMatrix.h>

// CV_DB_LIB
#include <ecvBBox.h>
#include <ecvCameraSensor.h>
#include <ecvGenericMesh.h>
#include <ecvHObjectCaster.h>
#include <ecvImage.h>
#include <ecvMaterialSet.h>
#include <ecvMesh.h>
#include <ecvPointCloud.h>
#include <ecvPolyline.h>
#include <ecvScalarField.h>
#include <ecvSensor.h>

// LOCAL
#include <ecvDisplayCoordinates.h>
#include <ecvRepresentationManager.h>
#include <ecvViewManager.h>

#include "VTKExtensions/InteractionStyle/vtkCustomInteractorStyle.h"
#include "vtkGLView.h"

// VtkRendering
#include <VtkRendering/Core/ActorMap.h>
#include <VtkRendering/Core/VtkRenderingUtils.h>

// VTK
#include <VTKExtensions/Views/vtkPVLODActor.h>
#include <vtkCellArray.h>
#include <vtkDoubleArray.h>
#include <vtkFieldData.h>
#include <vtkGenericOpenGLRenderWindow.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkStringArray.h>
#include <vtkTextActor.h>
#include <vtkTextProperty.h>
#include <vtkUnsignedCharArray.h>

// SYSTEM
#include <assert.h>

// Qt
#include <QImage>
#include <algorithm>

namespace Visualization {

void VtkDisplayTools::registerVisualizer(QMainWindow* win, bool stereoMode) {
    registerViewManagerTypedRelay();
    installDisplayToolsRelay();

    this->m_vtkWidget = new QVTKWidgetCustom(win, this, stereoMode);
    this->m_vtkWidget->connectSignalsTo(this);
    SetMainScreen(this->m_vtkWidget);
    SetCurrentScreen(this->m_vtkWidget);

    if (!m_visualizer3D) {
        auto renderer = vtkSmartPointer<vtkRenderer>::New();
        auto renderWindow =
                vtkSmartPointer<vtkGenericOpenGLRenderWindow>::New();
        // NOTE: depth peeling is enabled lazily in setMeshOpacity when an
        // actor becomes translucent.  Enabling it here globally would break
        // vtkGridAxesActor3D rendering (it inherits vtkProp3D, not vtkActor,
        // and may not participate correctly in the depth-peeling pipeline).
        renderWindow->AddRenderer(renderer);
        auto interactorStyle =
                vtkSmartPointer<VTKExtensions::vtkCustomInteractorStyle>::New();
        m_visualizer3D.reset(new VtkVis(renderer, renderWindow, interactorStyle,
                                        "3Dviewer", false));
        // m_visualizer3D.reset(new VtkVis(interactorStyle,
        // "3Dviewer", false)); // deprecated!
    }

    getQVtkWidget()->SetRenderWindow(m_visualizer3D->getRenderWindow());
    m_visualizer3D->setupInteractor(getQVtkWidget()->GetInteractor(),
                                    getQVtkWidget()->GetRenderWindow());
    getQVtkWidget()->initVtk(m_visualizer3D->getRenderWindowInteractor(),
                             false);
    getQVtkWidget()->setCustomInteractorStyle(
            m_visualizer3D->get3DInteractorStyle());
    m_visualizer3D->initialize();

    if (ecvDisplayTools::USE_2D) {
        if (!m_visualizer2D) {
            m_visualizer2D.reset(new ImageVis("2Dviewer", false));
        }

        m_visualizer2D->setRender(getQVtkWidget()->getVtkRender());
        m_visualizer2D->setupInteractor(getQVtkWidget()->GetInteractor(),
                                        getQVtkWidget()->GetRenderWindow());
    } else {
        m_visualizer2D = nullptr;
    }

    ecvRepresentationManager::instance().setActorCleanupCallback(
            [this](ccHObject* entity, ecvGenericGLDisplay* view) {
                if (!entity) return;
                if (view) {
                    auto* glView = dynamic_cast<vtkGLView*>(view);
                    if (glView && !glView->getVisualizer3D()) return;
                }
                std::string viewID = CVTools::FromQString(entity->getViewId());
                VtkVis* vis = resolveVisualizer(view);
                if (vis && vis->contains(viewID)) {
                    vis->removePointCloud(viewID);
                    vis->removePolygonMesh(viewID);
                    vis->removeShape(viewID);
                }
            });
}

VtkDisplayTools::~VtkDisplayTools() {
    ecvRepresentationManager::instance().setActorCleanupCallback(nullptr);

    if (m_engineOwnedWidget) {
        delete m_engineOwnedWidget;
        m_engineOwnedWidget = nullptr;
    }
    m_vtkWidget = nullptr;
}

void VtkDisplayTools::switchActiveView(VtkVisPtr vis,
                                       QVTKWidgetCustom* widget) {
    if (!vis || !widget) return;

    if (m_visualizer3D && m_visualizer3D != vis) {
        disconnect(m_visualizer3D.get(),
                   &ecvGenericVisualizer3D::interactorPointPickedEvent, this,
                   &ecvDisplayTools::onPointPicking);
    }
    connect(vis.get(), &ecvGenericVisualizer3D::interactorPointPickedEvent,
            this, &ecvDisplayTools::onPointPicking, Qt::UniqueConnection);

    QVTKWidgetCustom* oldWidget = m_vtkWidget;
    m_visualizer3D = vis;
    m_vtkWidget = widget;
    SetCurrentScreen(widget);
    SetMainScreen(widget);

    if (m_visualizer2D && widget) {
        m_visualizer2D->setRender(widget->getVtkRender());
        m_visualizer2D->setupInteractor(widget->GetInteractor(),
                                        widget->GetRenderWindow());
    }

    if (oldWidget && oldWidget != widget) {
        // Per-view widgets (owned by an vtkGLView) must NOT be hidden or
        // detached — they stay visible inside their own layout cell.
        // Only hide/detach the legacy engine-owned singleton widget.
        if (!oldWidget->ownerView()) {
            oldWidget->hide();
            oldWidget->setParent(nullptr);
            if (!m_engineOwnedWidget) {
                m_engineOwnedWidget = oldWidget;
            }
        }
    }
}

// Phase M4: ScopedHotZoneRender deleted. vtkGLView now calls
// DrawClickableItems(xStart, yStart, hotZone, items, display) directly.

VtkVis* VtkDisplayTools::resolveVisualizer(ecvGenericGLDisplay* display) const {
    // Phase M3: prefer per-view pipeline from vtkGLView, fall back to engine.
    if (display) {
        auto* glView = dynamic_cast<vtkGLView*>(display);
        if (glView && glView->getVisualizer3D()) {
            return dynamic_cast<VtkVis*>(glView->getVisualizer3D());
        }
    }
    return m_visualizer3D.get();
}

VtkVis* VtkDisplayTools::findVisByActorId(const std::string& viewId) const {
    const auto& views = ecvViewManager::instance().getAllViews();
    for (auto* view : views) {
        auto* glView = dynamic_cast<vtkGLView*>(view);
        if (glView && glView->getVisualizer3D()) {
            auto* vis = dynamic_cast<VtkVis*>(glView->getVisualizer3D());
            if (vis && vis->contains(viewId)) {
                return vis;
            }
        }
    }
    if (m_visualizer3D && m_visualizer3D->contains(viewId)) {
        return m_visualizer3D.get();
    }
    return nullptr;
}

VtkVis* VtkDisplayTools::findVisByActorIdOrActive(
        const std::string& viewId) const {
    VtkVis* vis = findVisByActorId(viewId);
    if (vis) return vis;
    auto* activeView = ecvViewManager::instance().getActiveView();
    if (activeView) {
        auto* glView = dynamic_cast<vtkGLView*>(activeView);
        if (glView && glView->getVisualizer3D()) {
            return dynamic_cast<VtkVis*>(glView->getVisualizer3D());
        }
    }
    return m_visualizer3D.get();
}

void VtkDisplayTools::drawPointCloud(const CC_DRAW_CONTEXT& context,
                                     ccPointCloud* ecvCloud) {
    VtkVis* vis = resolveVisualizer(context.display);
    std::string viewID = CVTools::FromQString(context.viewID);
    int viewport = context.defaultViewPort;
    bool firstShow = !vis->contains(viewID);
    bool hasRedrawn = false;

    // Create local context to pass entity's redraw state
    // This ensures updateShadingMode() in VtkVis updates normals/colors when
    // needed
    CC_DRAW_CONTEXT localContext = context;
    if (ecvCloud->isRedraw()) {
        localContext.forceRedraw = true;
    }

    if (ecvCloud->isRedraw() || firstShow) {
        bool needFullRebuild =
                firstShow || checkEntityNeedUpdate(vis, viewID, ecvCloud);
        bool sfTriggered = false;

        // SF hiding: O(1) check using the cached display range stored in the
        // polydata's field data ("_SFDispRange"). Avoids an O(n) visible-point
        // counting loop on every draw call.
        if (!needFullRebuild && localContext.drawParam.showSF) {
            ccScalarField* sf = ecvCloud->getCurrentDisplayedScalarField();
            if (sf) {
                const auto& disp = sf->displayRange();
                bool narrowed =
                        (disp.start() > disp.min() || disp.stop() < disp.max());
                vtkActor* actor = vis->getActorById(viewID);
                vtkPolyData* pd = nullptr;
                if (actor && actor->GetMapper()) {
                    pd = vtkPolyData::SafeDownCast(
                            actor->GetMapper()->GetInputDataObject(0, 0));
                }
                if (pd) {
                    auto* cached = vtkDoubleArray::SafeDownCast(
                            pd->GetFieldData()->GetAbstractArray(
                                    "_SFDispRange"));
                    if (narrowed) {
                        // Range is narrowed; rebuild if the range changed
                        // since the last build (or no cache exists).
                        if (!cached || cached->GetValue(0) != disp.start() ||
                            cached->GetValue(1) != disp.stop()) {
                            needFullRebuild = true;
                            sfTriggered = true;
                        }
                    } else if (cached) {
                        // Range restored to full but polydata was built with a
                        // narrowed range — need to rebuild with all points.
                        needFullRebuild = true;
                        sfTriggered = true;
                    }
                }
            }
        }

        if (needFullRebuild) {
            vis->drawPointCloud(localContext, ecvCloud, sfTriggered);
            vis->updateNormals(localContext, ecvCloud);
            hasRedrawn = true;
        } else {
            vis->resetScalarColor(viewID, true, viewport);
            if (!updateEntityColor(localContext, ecvCloud)) {
                vis->drawPointCloud(localContext, ecvCloud);
                vis->updateNormals(localContext, ecvCloud);
                hasRedrawn = true;
            } else {
                if (localContext.drawParam.showNorms) {
                    vis->updateNormals(localContext, ecvCloud);
                } else {
                    vis->updateNormals(localContext,
                                       static_cast<ccPointCloud*>(nullptr));
                }
            }
        }
    }

    if (vis->contains(viewID)) {
        vis->setPointSize(context.defaultPointSize, viewID, viewport);

        if (firstShow || hasRedrawn) {
            vis->setCurrentSourceObject(ecvCloud, viewID);
        }

        // Sync ALL scalar fields to VTK on first show or force redraw
        // This ensures all SFs are available for selection/extraction (Find
        // Data)
        if (firstShow || localContext.forceRedraw) {
            unsigned numSFs = ecvCloud->getNumberOfScalarFields();
            for (unsigned i = 0; i < numSFs; ++i) {
                vis->addScalarFieldToVTK(viewID, ecvCloud, static_cast<int>(i),
                                         viewport);
            }

            // For point clouds without scalar fields, we still need to set
            // DatasetName This ensures they appear in the Data Producer combo
            // (ParaView style)
            if (numSFs == 0) {
                vtkActor* actor = vis->getActorById(viewID);
                if (actor && actor->GetMapper()) {
                    vtkPolyData* polyData = vtkPolyData::SafeDownCast(
                            actor->GetMapper()->GetInput());
                    if (polyData) {
                        vtkFieldData* fieldData = polyData->GetFieldData();
                        if (fieldData) {
                            vtkStringArray* datasetNameArray =
                                    vtkStringArray::SafeDownCast(
                                            fieldData->GetAbstractArray(
                                                    "DatasetName"));
                            if (!datasetNameArray) {
                                // DatasetName not yet added, create it
                                QString cloudName = ecvCloud->getName();
                                if (!cloudName.isEmpty()) {
                                    vtkSmartPointer<vtkStringArray>
                                            newDatasetNameArray =
                                                    vtkSmartPointer<
                                                            vtkStringArray>::
                                                            New();
                                    newDatasetNameArray->SetName("DatasetName");
                                    newDatasetNameArray->SetNumberOfTuples(1);
                                    newDatasetNameArray->SetValue(
                                            0, cloudName.toStdString());
                                    fieldData->AddArray(newDatasetNameArray);

                                    CVLog::PrintDebug(
                                            QString("[VtkDisplayTools] Added "
                                                    "DatasetName for point "
                                                    "cloud "
                                                    "without SFs: '%1'")
                                                    .arg(cloudName));
                                }
                            }
                        }
                    }
                }
            }
        }
        // Also ensure current SF is updated for tooltip display
        else if (context.drawParam.showSF && ecvCloud->sfShown()) {
            int sfIdx = ecvCloud->getCurrentDisplayedScalarFieldIndex();
            if (sfIdx >= 0) {
                // Add/update scalar field to VTK for tooltip display
                // Extract values directly from ccPointCloud (not from PCL
                // cloud) Note: addScalarFieldToVTK has internal check to avoid
                // unnecessary updates
                vis->addScalarFieldToVTK(viewID, ecvCloud, sfIdx, viewport);
            }
        }

        if ((!context.drawParam.showColors && !context.drawParam.showSF) ||
            ecvCloud->isColorOverridden()) {
            ecvColor::Rgbf pointUniqueColor =
                    ecvTools::TransFormRGB(context.pointsCurrentCol);
            vis->setPointCloudUniqueColor(pointUniqueColor.r,
                                          pointUniqueColor.g,
                                          pointUniqueColor.b, viewID, viewport);
        }

        vis->setPointGaussianRendering(ecvCloud->pointGaussianEnabled(),
                                       ecvCloud->pointGaussianRadius(),
                                       ecvCloud->pointGaussianShaderPreset(),
                                       ecvCloud->pointGaussianEmissive(),
                                       viewID, viewport);

        vis->setPointCloudOpacity(context.opacity, viewID, viewport);
    }
}

void VtkDisplayTools::updateMeshTextures(const CC_DRAW_CONTEXT& context,
                                         const ccGenericMesh* mesh) {
    VtkVis* vis = resolveVisualizer(context.display);
    std::string viewID = CVTools::FromQString(context.viewID);
    bool firstShow = !vis->contains(viewID);
    if (firstShow) {
        drawMesh(const_cast<CC_DRAW_CONTEXT&>(context),
                 const_cast<ccGenericMesh*>(mesh));
    } else {
        bool applyMaterials = (mesh->hasMaterials() && mesh->materialsShown());
        bool showTextures = (mesh->hasTextures() && mesh->materialsShown());
        if (applyMaterials || showTextures) {
            const ccMaterialSet* materials = mesh->getMaterialSet();
            assert(materials);
            if (!vis->updateTexture(context, materials)) {
                CVLog::Warning(QString("Update mesh texture failed!"));
            }
        } else {
            CVLog::Warning(
                    QString("Mesh texture has not been shown, please toggle it "
                            "to be shown!"));
        }
    }
}

void VtkDisplayTools::drawMesh(CC_DRAW_CONTEXT& context, ccGenericMesh* mesh) {
    VtkVis* vis = resolveVisualizer(context.display);
    std::string viewID = CVTools::FromQString(context.viewID);
    int viewport = context.defaultViewPort;
    context.visFiltering = true;
    bool firstShow = !vis->contains(viewID);

    // Set forceRedraw based on entity's redraw state
    // This ensures updateShadingMode() in VtkVis updates normals/colors when
    // needed
    if (mesh->isRedraw()) {
        context.forceRedraw = true;
    }

    if (mesh->isRedraw() || firstShow) {
        CVLog::PrintDebug(
                "[VtkDisplayTools::drawMesh] Entering render block "
                "(isRedraw=%d || firstShow=%d)",
                mesh->isRedraw(), firstShow);

        ccPointCloud* ecvCloud = ccHObjectCaster::ToPointCloud(mesh);
        if (!ecvCloud) {
            CVLog::Warning(
                    "[VtkDisplayTools::drawMesh] Failed to get point cloud "
                    "from mesh!");
            return;
        }

        // materials & textures
        bool applyMaterials = (mesh->hasMaterials() && mesh->materialsShown());
        bool lodEnabled = false;
        bool showTextures =
                (mesh->hasTextures() && mesh->materialsShown() && !lodEnabled);

        CVLog::PrintDebug(
                "[VtkDisplayTools::drawMesh] applyMaterials=%d, "
                "showTextures=%d",
                applyMaterials, showTextures);

        if (firstShow || checkEntityNeedUpdate(vis, viewID, ecvCloud)) {
            if (applyMaterials || showTextures) {
                if (!vis->addTextureMeshFromCCMesh(mesh, viewID, viewport)) {
                    CVLog::Warning(
                            "[VtkDisplayTools::drawMesh] Failed to add texture "
                            "mesh, falling back to regular mesh");
                    vis->drawMesh(context, mesh);
                }
            } else {
                vis->drawMesh(context, mesh);

                vtkActor* actor = vis->getActorById(viewID);
                if (actor && actor->GetMapper() &&
                    mesh->getName().length() > 0) {
                    vtkPolyData* polyData = vtkPolyData::SafeDownCast(
                            actor->GetMapper()->GetInput());
                    if (polyData) {
                        QString meshName = mesh->getName();
                        vtkSmartPointer<vtkStringArray> datasetNameArray =
                                vtkSmartPointer<vtkStringArray>::New();
                        datasetNameArray->SetName("DatasetName");
                        datasetNameArray->SetNumberOfTuples(1);
                        datasetNameArray->SetValue(0, meshName.toStdString());
                        polyData->GetFieldData()->AddArray(datasetNameArray);
                    }
                }
            }
        } else {
            CVLog::PrintDebug(
                    "[VtkDisplayTools::drawMesh] Update path: updating "
                    "properties only");
            vis->resetScalarColor(viewID, true, viewport);
            if (!updateEntityColor(context, ecvCloud)) {
                if (applyMaterials || showTextures) {
                    const ccMaterialSet* materials = mesh->getMaterialSet();
                    if (materials) {
                        if (!vis->updateTexture(context, materials)) {
                            CVLog::Warning(
                                    "[VtkDisplayTools::drawMesh] Update "
                                    "texture failed!");
                        }
                    }
                } else {
                    // Geometry is unchanged (checkEntityNeedUpdate == false)
                    // and no colors/materials to update. Use non-destructive
                    // path to avoid recreating the actor (which would reset
                    // properties set by active filter tools).
                    context.visFiltering = false;
                    vis->drawMesh(context, mesh);
                }
            }
        }
        vis->transformEntities(context);
    }

    if (vis->contains(viewID)) {
        vis->setMeshRenderingMode(context.meshRenderingMode, viewID, viewport);

        ccMesh* ccMeshObj = dynamic_cast<ccMesh*>(mesh);
        if (ccMeshObj && firstShow) {
            vis->setCurrentSourceObject(ccMeshObj, viewID);
        }

        if ((!context.drawParam.showColors && !context.drawParam.showSF) ||
            mesh->isColorOverridden()) {
            ecvColor::Rgbf meshColor =
                    ecvTools::TransFormRGB(context.defaultMeshColor);
            vis->setPointCloudUniqueColor(meshColor.r, meshColor.g, meshColor.b,
                                          viewID, viewport);
        }
        vis->setMeshOpacity(context.opacity, viewID, viewport);
        vis->setMeshStippling(mesh->stipplingEnabled(), viewID, viewport);
        vis->setPointGaussianRendering(
                mesh->pointGaussianEnabled(), mesh->pointGaussianRadius(),
                mesh->pointGaussianShaderPreset(),
                mesh->pointGaussianEmissive(), viewID, viewport);
    }
}

void VtkDisplayTools::drawPolygon(const CC_DRAW_CONTEXT& context,
                                  ccPolyline* polyline) {
    VtkVis* vis = resolveVisualizer(context.display);
    std::string viewID = CVTools::FromQString(context.viewID);
    bool firstShow = !vis->contains(viewID);
    int viewport = context.defaultViewPort;

    if (polyline->isRedraw() || firstShow) {
        vis->drawPolyline(context, polyline, polyline->isClosed());
    }

    if (vis->contains(viewID)) {
        ecvColor::Rgbf polygonColor =
                ecvTools::TransFormRGB(context.defaultPolylineColor);
        vis->setShapeUniqueColor(polygonColor.r, polygonColor.g, polygonColor.b,
                                 viewID, viewport);
        vis->setLineWidth(context.currentLineWidth, viewID, viewport);
        vis->setLightMode(viewID, viewport);
    }
}

void VtkDisplayTools::drawLines(const CC_DRAW_CONTEXT& context,
                                cloudViewer::geometry::LineSet* lineset) {
    VtkVis* vis = resolveVisualizer(context.display);
    std::string viewID = CVTools::FromQString(context.viewID);
    bool firstShow = !vis->contains(viewID);
    int viewport = context.defaultViewPort;

    if (lineset->isRedraw() || firstShow) {
        vis->drawLineSet(context, lineset);
        vis->transformEntities(context);
    }

    if (vis->contains(viewID)) {
        if (lineset->isColorOverridden() || !lineset->HasColors()) {
            ecvColor::Rgbf polygonColor =
                    ecvTools::TransFormRGB(context.defaultPolylineColor);
            vis->setShapeUniqueColor(polygonColor.r, polygonColor.g,
                                     polygonColor.b, viewID, viewport);
        }

        vis->setLineWidth(context.currentLineWidth, viewID, viewport);
        vis->setLightMode(viewID, viewport);
    }
}

void VtkDisplayTools::drawImage(const CC_DRAW_CONTEXT& context,
                                ccImage* image) {
    if (!image || !m_visualizer2D) return;

    std::string viewID = CVTools::FromQString(context.viewID);
    bool firstShow = !m_visualizer2D->contains(viewID);

    bool imageExists = !firstShow;

    // Note: isRedraw() might be true even if only opacity changed,
    // because ccHObject::draw() always sets setRedraw(true) at the end.
    // So we check if image already exists - if it does and only opacity
    // changed, we can just update opacity without reloading image data.
    bool needsImageReload = firstShow || (image->isRedraw() && !imageExists);

    double opacity = image->getAlpha();
    // Only reload image data if image data has changed or first time showing
    if (needsImageReload) {
        const QImage& qimage = image->data();
        if (qimage.isNull()) {
            CVLog::Warning(
                    "[VtkDisplayTools::drawImage] Failed to get image data!");
            return;
        }

        CVLog::PrintDebug(
                "[VtkDisplayTools::drawImage] Reloading image data: %d x %d, "
                "opacity: %f, redraw: %d, firstShow: %d, isEnabled: %d",
                image->getW(), image->getH(), image->getAlpha(),
                image->isRedraw(), firstShow, image->isEnabled());

        // ParaView-style: Use addQImage with vtkQImageToImageSource for
        // efficient conversion This avoids manual format conversion and memory
        // copying
        m_visualizer2D->addQImage(qimage, viewID, opacity);
    } else {
        // Only update opacity if image already exists and data hasn't changed
        CVLog::PrintDebug(
                "[VtkDisplayTools::drawImage] Updating opacity only for image "
                "%s, "
                "opacity: %f",
                viewID.c_str(), opacity);
        m_visualizer2D->changeOpacity(opacity, viewID);
    }
}

void VtkDisplayTools::drawSensor(const CC_DRAW_CONTEXT& context,
                                 ccSensor* sensor) {
    if (!sensor) {
        return;
    }

    VtkVis* vis = resolveVisualizer(context.display);
    std::string viewID = CVTools::FromQString(context.viewID);
    bool firstShow = !vis->contains(viewID);
    int viewport = context.defaultViewPort;

    if (sensor->isRedraw() || firstShow) {
        vis->drawSensor(context, sensor);
        vis->transformEntities(context);
    }

    if (vis->contains(viewID)) {
        vis->setLineWidth(context.currentLineWidth, viewID, viewport);
    }
}

bool VtkDisplayTools::updateEntityColor(const CC_DRAW_CONTEXT& context,
                                        ccHObject* ent) {
#ifdef _DEBUG
    CVTools::TimeStart();
#endif  // _DEBUG
    ccPointCloud* cloud = ccHObjectCaster::ToPointCloud(ent);
    if (!cloud) {
        return false;
    }

    VtkVis* vis = resolveVisualizer(context.display);
    std::string viewID = CVTools::FromQString(context.viewID);
    vtkActor* modelActor = vis->getActorById(viewID);
    if (!modelActor) {
        return false;
    }

    // Get the current poly data (mapper may be vtkDataSetMapper or
    // vtkPolyDataMapper; use vtkAlgorithm API for safety).
    vtkMapper* mapper = modelActor->GetMapper();
    if (!mapper) return false;
    vtkSmartPointer<vtkPolyData> polydata =
            vtkPolyData::SafeDownCast(mapper->GetInputDataObject(0, 0));
    if (!polydata) return false;

    unsigned old_points_num =
            static_cast<unsigned>(polydata->GetNumberOfPoints());
    unsigned new_points_num = cloud->size();
    if (cloud->isVisibilityTableInstantiated()) {
        new_points_num = 0;
        for (unsigned i = 0; i < cloud->size(); ++i) {
            if (cloud->getTheVisibilityArray().at(i) == POINT_VISIBLE)
                ++new_points_num;
        }
    }

    // SF display range can hide points when narrowed.  The color-only fast
    // path cannot change the point set, so force a full polydata rebuild
    // whenever the visible count differs.
    if (context.drawParam.showSF) {
        ccScalarField* sf = cloud->getCurrentDisplayedScalarField();
        if (sf) {
            const auto& disp = sf->displayRange();
            bool narrowed =
                    (disp.start() > disp.min() || disp.stop() < disp.max());
            if (narrowed) {
                unsigned sf_visible = 0;
                const bool use_vis = cloud->isVisibilityTableInstantiated();
                for (unsigned i = 0; i < cloud->size(); ++i) {
                    if (use_vis &&
                        cloud->getTheVisibilityArray().at(i) != POINT_VISIBLE)
                        continue;
                    if (disp.isInRange(sf->getValue(i))) ++sf_visible;
                }
                new_points_num = sf_visible;
            }
        }
    }

    if (old_points_num != new_points_num) {
        return false;
    }

    if (!context.drawParam.showColors && !context.drawParam.showSF) {
        return false;
    }

    bool has_colors = false;
    double minmax[2];
    vtkSmartPointer<vtkDataArray> scalars;
    if (Converters::Cc2Vtk::GetVtkScalars(cloud, scalars,
                                          context.drawParam.showSF)) {
        polydata->GetPointData()->SetScalars(scalars);
        scalars->GetRange(minmax);
        has_colors = true;
    }

    if (has_colors) {
        mapper->SetScalarRange(minmax);
        mapper->SetInputDataObject(polydata);
    }

    CVLog::PrintDebug(QString("updateEntityColor: finish cost %1 s")
                              .arg(CVTools::TimeOff()));

    return (true);
}

void VtkDisplayTools::draw(const CC_DRAW_CONTEXT& context,
                           const ccHObject* obj) {
    if (obj->isA(CV_TYPES::POINT_CLOUD)) {
        ccPointCloud* ecvCloud =
                ccHObjectCaster::ToPointCloud(const_cast<ccHObject*>(obj));
        if (!ecvCloud) return;

        drawPointCloud(context, ecvCloud);
    } else if (obj->isKindOf(CV_TYPES::MESH) ||
               obj->isKindOf(CV_TYPES::SUB_MESH)) {
        ccGenericMesh* tempMesh =
                ccHObjectCaster::ToGenericMesh(const_cast<ccHObject*>(obj));
        if (!tempMesh) {
            CVLog::Warning(
                    "[VtkDisplayTools::draw] Failed to cast to ccGenericMesh!");
            return;
        }
        drawMesh(const_cast<CC_DRAW_CONTEXT&>(context), tempMesh);
    } else if (obj->isA(CV_TYPES::POLY_LINE)) {
        ccPolyline* tempPolyline =
                ccHObjectCaster::ToPolyline(const_cast<ccHObject*>(obj));
        if (!tempPolyline) return;
        drawPolygon(context, tempPolyline);
    } else if (obj->isA(CV_TYPES::LINESET)) {
        cloudViewer::geometry::LineSet* lineset =
                ccHObjectCaster::ToLineSet(const_cast<ccHObject*>(obj));
        if (!lineset) return;
        drawLines(context, lineset);
    } else if (obj->isA(CV_TYPES::IMAGE)) {
        ccImage* image = ccHObjectCaster::ToImage(const_cast<ccHObject*>(obj));
        if (!image) return;
        drawImage(context, image);
    } else if (obj->isKindOf(CV_TYPES::SENSOR)) {
        ccSensor* sensor =
                ccHObjectCaster::ToSensor(const_cast<ccHObject*>(obj));
        if (!sensor) return;
        drawSensor(context, sensor);
    } else {
        return;
    }

    VtkVis* vis = resolveVisualizer(context.display);
    if (vis) {
        vis->resetCameraClippingRange(context.defaultViewPort);
    }
}

bool VtkDisplayTools::checkEntityNeedUpdate(VtkVis* vis,
                                            std::string& viewID,
                                            const ccHObject* obj) {
    bool firstShow = !vis->contains(viewID);
    if (firstShow) return true;

    ccPointCloud* cloud =
            ccHObjectCaster::ToPointCloud(const_cast<ccHObject*>(obj));
    if (!cloud) {
        return true;
    }

    vtkActor* modelActor = vis->getActorById(viewID);
    if (!modelActor) {
        return true;
    }

    // Get the current poly data (mapper may be vtkDataSetMapper or
    // vtkPolyDataMapper; use vtkAlgorithm API for safety).
    vtkMapper* mapper = modelActor->GetMapper();
    if (!mapper) return true;
    vtkSmartPointer<vtkPolyData> polydata =
            vtkPolyData::SafeDownCast(mapper->GetInputDataObject(0, 0));
    if (!polydata) {
        return true;
    }

    unsigned old_points_num =
            static_cast<unsigned>(polydata->GetNumberOfPoints());
    unsigned new_points_num = cloud->size();
    if (cloud->isVisibilityTableInstantiated()) {
        new_points_num = 0;
        for (unsigned i = 0; i < cloud->size(); ++i) {
            if (cloud->getTheVisibilityArray().at(i) == POINT_VISIBLE)
                ++new_points_num;
        }
    }
    if (old_points_num != new_points_num) {
        return true;
    }

    double bounds[6];
    polydata->GetBounds(bounds);
    CCVector3 minCorner(bounds[0], bounds[2], bounds[4]);
    CCVector3 maxCorner(bounds[1], bounds[3], bounds[5]);
    ccBBox oldBBox(minCorner, maxCorner);
    ccBBox newBBox = cloud->getOwnBB();

    if (abs((oldBBox.minCorner() - newBBox.minCorner()).normd()) >
        EPSILON_VALUE) {
        return true;
    }

    if (abs((oldBBox.maxCorner() - newBBox.maxCorner()).normd()) >
        EPSILON_VALUE) {
        return true;
    }

    return false;
}

void VtkDisplayTools::drawBBox(const CC_DRAW_CONTEXT& context,
                               const ccBBox* bbox) {
    VtkVis* vis = resolveVisualizer(context.display);
    ecvColor::Rgbf colf = ecvTools::TransFormRGB(context.bbDefaultCol);
    int viewport = context.defaultViewPort;
    if (vis) {
        std::string bboxID = CVTools::FromQString(context.viewID);
        if (!vis->contains(bboxID)) {
            vis->addCube(bbox->minCorner().x, bbox->maxCorner().x,
                         bbox->minCorner().y, bbox->maxCorner().y,
                         bbox->minCorner().z, bbox->maxCorner().z, colf.r,
                         colf.g, colf.b, bboxID, viewport);
            vis->setLineWidth(context.defaultLineWidth, bboxID, viewport);
            vis->setLightMode(bboxID, viewport);
        }

        const int representation =
                (context.meshRenderingMode ==
                 MESH_RENDERING_MODE::ECV_SURFACE_MODE)
                        ? VtkVis::PCL_VISUALIZER_REPRESENTATION_SURFACE
                        : VtkVis::PCL_VISUALIZER_REPRESENTATION_WIREFRAME;
        vis->setShapeRenderingProperties(VtkVis::PCL_VISUALIZER_REPRESENTATION,
                                         representation, bboxID, viewport);

        vis->setShapeUniqueColor(colf.r, colf.g, colf.b, bboxID, viewport);
        vis->setLineWidth(context.defaultLineWidth, bboxID, viewport);

        if (context.opacity >= 0.0 && context.opacity <= 1.0) {
            vis->setShapeRenderingProperties(VtkVis::PCL_VISUALIZER_OPACITY,
                                             context.opacity, bboxID, viewport);
        }
    }
}

void VtkDisplayTools::drawBBoxBatch(const CC_DRAW_CONTEXT& context,
                                    const std::vector<ccBBox>& boxes) {
    VtkVis* vis = resolveVisualizer(context.display);
    if (!vis || boxes.empty()) return;

    std::string batchID = CVTools::FromQString(context.viewID);
    int viewport = context.defaultViewPort;

    if (vis->contains(batchID)) {
        vis->removeShape(batchID);
    }

    const bool isSurface = (context.meshRenderingMode ==
                            MESH_RENDERING_MODE::ECV_SURFACE_MODE);
    const bool isPoints = (context.opacity == 1.0 && isSurface &&
                           context.viewID.contains("points"));

    auto points = vtkSmartPointer<vtkPoints>::New();
    auto cells = vtkSmartPointer<vtkCellArray>::New();

    if (isPoints) {
        points->SetNumberOfPoints(static_cast<vtkIdType>(boxes.size()));
        for (size_t i = 0; i < boxes.size(); ++i) {
            CCVector3 center = boxes[i].getCenter();
            points->SetPoint(static_cast<vtkIdType>(i), center.x, center.y,
                             center.z);
            vtkIdType pid = static_cast<vtkIdType>(i);
            cells->InsertNextCell(1, &pid);
        }
    } else {
        const int vertsPerBox = 8;
        const int linesPerBox = 12;
        const int quadsPerBox = 6;
        vtkIdType numBoxes = static_cast<vtkIdType>(boxes.size());

        if (isSurface) {
            points->SetNumberOfPoints(numBoxes * vertsPerBox);
            cells->Allocate(numBoxes * quadsPerBox);
        } else {
            points->SetNumberOfPoints(numBoxes * vertsPerBox);
            cells->Allocate(numBoxes * linesPerBox);
        }

        for (vtkIdType b = 0; b < numBoxes; ++b) {
            const auto& mn = boxes[b].minCorner();
            const auto& mx = boxes[b].maxCorner();
            vtkIdType base = b * vertsPerBox;
            points->SetPoint(base + 0, mn.x, mn.y, mn.z);
            points->SetPoint(base + 1, mx.x, mn.y, mn.z);
            points->SetPoint(base + 2, mx.x, mx.y, mn.z);
            points->SetPoint(base + 3, mn.x, mx.y, mn.z);
            points->SetPoint(base + 4, mn.x, mn.y, mx.z);
            points->SetPoint(base + 5, mx.x, mn.y, mx.z);
            points->SetPoint(base + 6, mx.x, mx.y, mx.z);
            points->SetPoint(base + 7, mn.x, mx.y, mx.z);

            if (isSurface) {
                vtkIdType q0[4] = {base, base + 3, base + 2, base + 1};
                vtkIdType q1[4] = {base + 4, base + 5, base + 6, base + 7};
                vtkIdType q2[4] = {base, base + 1, base + 5, base + 4};
                vtkIdType q3[4] = {base + 2, base + 3, base + 7, base + 6};
                vtkIdType q4[4] = {base, base + 4, base + 7, base + 3};
                vtkIdType q5[4] = {base + 1, base + 2, base + 6, base + 5};
                cells->InsertNextCell(4, q0);
                cells->InsertNextCell(4, q1);
                cells->InsertNextCell(4, q2);
                cells->InsertNextCell(4, q3);
                cells->InsertNextCell(4, q4);
                cells->InsertNextCell(4, q5);
            } else {
                vtkIdType e[2];
                auto addEdge = [&](vtkIdType a, vtkIdType b) {
                    e[0] = base + a;
                    e[1] = base + b;
                    cells->InsertNextCell(2, e);
                };
                addEdge(0, 1);
                addEdge(1, 2);
                addEdge(2, 3);
                addEdge(3, 0);
                addEdge(4, 5);
                addEdge(5, 6);
                addEdge(6, 7);
                addEdge(7, 4);
                addEdge(0, 4);
                addEdge(1, 5);
                addEdge(2, 6);
                addEdge(3, 7);
            }
        }
    }

    auto polydata = vtkSmartPointer<vtkPolyData>::New();
    polydata->SetPoints(points);
    if (isPoints)
        polydata->SetVerts(cells);
    else if (isSurface)
        polydata->SetPolys(cells);
    else
        polydata->SetLines(cells);

    vtkSmartPointer<vtkPVLODActor> actor;
    VtkRendering::CreateActorFromVTKDataSet(polydata, actor, false);
    actor->PickableOff();

    ecvColor::Rgbf colf = ecvTools::TransFormRGB(context.bbDefaultCol);
    actor->GetProperty()->SetColor(colf.r, colf.g, colf.b);

    if (isPoints) {
        actor->GetProperty()->SetRepresentationToPoints();
        actor->GetProperty()->SetPointSize(3.0);
    } else if (isSurface) {
        actor->GetProperty()->SetRepresentationToSurface();
    } else {
        actor->GetProperty()->SetRepresentationToWireframe();
        actor->GetProperty()->SetLineWidth(
                static_cast<float>(context.defaultLineWidth));
    }

    if (context.opacity >= 0.0 && context.opacity <= 1.0) {
        actor->GetProperty()->SetOpacity(context.opacity);
    }

    vis->addActorToRenderer(actor, viewport);
    (*vis->getShapeActorMap())[batchID] = actor;
}

void VtkDisplayTools::drawOrientedBBox(const CC_DRAW_CONTEXT& context,
                                       const ecvOrientedBBox* obb) {
    VtkVis* vis = resolveVisualizer(context.display);
    int viewport = context.defaultViewPort;
    if (vis) {
        std::string bboxID = CVTools::FromQString(context.viewID);
        if (vis->contains(bboxID)) {
            vis->removeShape(bboxID);
        }

        vis->addOrientedCube(*obb, bboxID, viewport);
        vis->setLineWidth(context.defaultLineWidth, bboxID, viewport);
        vis->setLightMode(bboxID, viewport);
    }
}

bool VtkDisplayTools::orientationMarkerShown() {
    return m_visualizer3D->pclMarkerAxesShown();
}

void VtkDisplayTools::toggleOrientationMarker(bool state) {
    if (state) {
        m_visualizer3D->showPclMarkerAxes(
                m_visualizer3D->getRenderWindowInteractor());
    } else {
        m_visualizer3D->hidePclMarkerAxes();
    }
}

void VtkDisplayTools::removeEntities(const CC_DRAW_CONTEXT& context) {
    VtkVis* vis = resolveVisualizer(context.display);
    if (context.removeEntityType == ENTITY_TYPE::ECV_ALL) {
        if (vis) {
            vis->removeEntities(context);
            vis->resetCameraClippingRange(context.defaultViewPort);
        }
        if (m_visualizer2D) {
            m_visualizer2D->removeAllLayers();
        }
        if (context.display) {
            auto* glView = dynamic_cast<vtkGLView*>(context.display);
            if (glView) {
                auto imgVis = glView->getImageVis();
                if (imgVis) {
                    imgVis->removeAllLayers();
                }
            }
        }
    } else if (context.removeEntityType == ENTITY_TYPE::ECV_IMAGE ||
               context.removeEntityType == ENTITY_TYPE::ECV_LINES_2D ||
               context.removeEntityType == ENTITY_TYPE::ECV_CIRCLE_2D ||
               context.removeEntityType == ENTITY_TYPE::ECV_RECTANGLE_2D ||
               context.removeEntityType == ENTITY_TYPE::ECV_TRIANGLE_2D ||
               context.removeEntityType == ENTITY_TYPE::ECV_MARK_POINT) {
        std::string viewId = CVTools::FromQString(context.removeViewID);
        if (m_visualizer2D && m_visualizer2D->contains(viewId)) {
            m_visualizer2D->removeLayer(viewId);
        }
        if (context.display) {
            auto* glView = dynamic_cast<vtkGLView*>(context.display);
            if (glView) {
                auto imgVis = glView->getImageVis();
                if (imgVis && imgVis->contains(viewId)) {
                    imgVis->removeLayer(viewId);
                }
            }
        }
    } else {
        if (context.removeEntityType == ENTITY_TYPE::ECV_TEXT2D ||
            context.removeEntityType == ENTITY_TYPE::ECV_POLYLINE_2D) {
            std::string viewId = CVTools::FromQString(context.removeViewID);
            if (m_visualizer2D && m_visualizer2D->contains(viewId)) {
                m_visualizer2D->removeLayer(viewId);
            }
            bool isSecondaryView =
                    context.display &&
                    context.display != static_cast<ecvDisplayTools*>(this);
            if (context.display) {
                auto* glView = dynamic_cast<vtkGLView*>(context.display);
                if (glView) {
                    auto imgVis = glView->getImageVis();
                    if (imgVis && imgVis->contains(viewId)) {
                        imgVis->removeLayer(viewId);
                    }
                    // For secondary views, text is rendered via vtkTextActor
                    // with composite IDs "groupID#text". Remove all actors
                    // whose ID starts with the group prefix.
                    if (isSecondaryView) {
                        auto* subVis = glView->getVisualizer3D();
                        if (subVis) {
                            std::string prefix = viewId + "#";
                            subVis->removeBySubstring(prefix,
                                                      context.defaultViewPort);
                        }
                    }
                }
            }
        }
        if (context.removeEntityType == ENTITY_TYPE::ECV_2DLABLE ||
            context.removeEntityType == ENTITY_TYPE::ECV_2DLABLE_VIEWPORT) {
            std::string viewId = CVTools::FromQString(context.removeViewID);
            if (m_visualizer2D) {
                m_visualizer2D->removeBySubstring(viewId);
            }
            if (context.display) {
                auto* glView = dynamic_cast<vtkGLView*>(context.display);
                if (glView) {
                    auto imgVis = glView->getImageVis();
                    if (imgVis) {
                        imgVis->removeBySubstring(viewId);
                    }
                }
            }
        }

        if (vis && vis->removeEntities(context)) {
            vis->resetCameraClippingRange(context.defaultViewPort);
        }
    }

    // Multi-window: propagate 3D scene entity removal to secondary views.
    // Only propagate when context.display is NULL (global removal request).
    // When context.display targets a specific view, the removal is
    // intentionally scoped to that view only (e.g., per-view label markers). 2D
    // overlays (text, rectangles, images, etc.) are per-view and must NOT be
    // propagated — each view's ClearBubbleView handles its own cleanup.
    const bool is2DOverlay =
            context.removeEntityType == ENTITY_TYPE::ECV_TEXT2D ||
            context.removeEntityType == ENTITY_TYPE::ECV_RECTANGLE_2D ||
            context.removeEntityType == ENTITY_TYPE::ECV_MARK_POINT ||
            context.removeEntityType == ENTITY_TYPE::ECV_IMAGE ||
            context.removeEntityType == ENTITY_TYPE::ECV_CIRCLE_2D ||
            context.removeEntityType == ENTITY_TYPE::ECV_POLYLINE_2D ||
            context.removeEntityType == ENTITY_TYPE::ECV_LINES_2D ||
            context.removeEntityType == ENTITY_TYPE::ECV_TRIANGLE_2D;

    const bool targetedRemoval = (context.display != nullptr);

    if (!is2DOverlay && !targetedRemoval) {
        const auto& views = ecvViewManager::instance().getAllViews();
        std::string removeId = CVTools::FromQString(context.removeViewID);
        for (auto* view : views) {
            if (!view || view == this) continue;
            auto* glView = dynamic_cast<vtkGLView*>(view);
            if (!glView) continue;

            auto* viewVis = dynamic_cast<VtkVis*>(glView->getVisualizer3D());
            if (viewVis) {
                viewVis->removeEntities(context);
            }

            auto imgVis = glView->getImageVis();
            if (imgVis) {
                if (context.removeEntityType == ENTITY_TYPE::ECV_ALL) {
                    imgVis->removeAllLayers();
                } else if (!removeId.empty() && imgVis->contains(removeId)) {
                    imgVis->removeLayer(removeId);
                }
            }
        }
    }
}

bool VtkDisplayTools::hideShowEntities(const CC_DRAW_CONTEXT& context) {
    VtkVis* vis = resolveVisualizer(context.display);
    std::string viewId = CVTools::FromQString(context.viewID);

    if (context.hideShowEntityType == ENTITY_TYPE::ECV_2DLABLE ||
        context.hideShowEntityType == ENTITY_TYPE::ECV_2DLABLE_VIEWPORT) {
        vis->hideShowActorsBySubstring(context.visible, viewId,
                                       context.defaultViewPort);
        if (m_visualizer2D) {
            m_visualizer2D->hideShowActorsBySubstring(context.visible, viewId);
        }
        return true;
    } else if (context.hideShowEntityType == ENTITY_TYPE::ECV_IMAGE ||
               context.hideShowEntityType == ENTITY_TYPE::ECV_LINES_2D ||
               context.hideShowEntityType == ENTITY_TYPE::ECV_CIRCLE_2D ||
               context.hideShowEntityType == ENTITY_TYPE::ECV_TRIANGLE_2D ||
               context.hideShowEntityType == ENTITY_TYPE::ECV_RECTANGLE_2D ||
               context.hideShowEntityType == ENTITY_TYPE::ECV_MARK_POINT) {
        if (!m_visualizer2D || !m_visualizer2D->contains(viewId)) return false;

        m_visualizer2D->hideShowActors(context.visible, viewId);
        return true;
    } else if (context.hideShowEntityType == ENTITY_TYPE::ECV_CAPTION) {
        if (vis->containWidget(viewId)) {
            vis->hideShowWidgets(context.visible, viewId,
                                 context.defaultViewPort);
        }
    } else {
        bool found = false;

        if (context.hideShowEntityType == ENTITY_TYPE::ECV_TEXT2D ||
            context.hideShowEntityType == ENTITY_TYPE::ECV_POLYLINE_2D) {
            if (m_visualizer2D && m_visualizer2D->contains(viewId)) {
                m_visualizer2D->hideShowActors(context.visible, viewId);
                found = true;
            }
        }

        if (vis->contains(viewId)) {
            vis->hideShowActors(context.visible, viewId,
                                context.defaultViewPort);
            found = true;
        }

        std::string normalViewId =
                CVTools::FromQString(context.viewID + "-normal");
        if (vis->contains(normalViewId)) {
            vis->hideShowActors(context.visible, normalViewId,
                                context.defaultViewPort);
            found = true;
        }

        if (found) {
            vis->resetCameraClippingRange(context.defaultViewPort);
        }

        return found;
    }

    return true;
}

void VtkDisplayTools::drawWidgets(const WIDGETS_PARAMETER& param) {
    VtkVis* vis = resolveVisualizer(param.context.display);
    int viewport = param.viewport;
    std::string viewID = CVTools::FromQString(param.viewID);
    switch (param.type) {
        case WIDGETS_TYPE::WIDGET_COORDINATE:
            break;
        case WIDGETS_TYPE::WIDGET_BBOX:
            break;
        case WIDGETS_TYPE::WIDGET_T2D: {
            bool isSecondaryT2D = param.context.display &&
                                  param.context.display !=
                                          static_cast<ecvDisplayTools*>(this);
            if (isSecondaryT2D) {
                Visualization::ImageVis* txtVis2D = nullptr;
                auto* glView = dynamic_cast<vtkGLView*>(param.context.display);
                if (glView && glView->getImageVis()) {
                    txtVis2D = glView->getImageVis().get();
                }
                if (txtVis2D) {
                    std::string text = CVTools::FromQString(param.text);
                    txtVis2D->addText(param.rect.x(), param.rect.y(), text,
                                      param.color.r, param.color.g,
                                      param.color.b, viewID, param.color.a,
                                      param.fontSize);
                }
            } else {
                Visualization::ImageVis* txtVis2D =
                        m_visualizer2D ? m_visualizer2D.get() : nullptr;
                if (txtVis2D) {
                    std::string text = CVTools::FromQString(param.text);
                    txtVis2D->addText(param.rect.x(), param.rect.y(), text,
                                      param.color.r, param.color.g,
                                      param.color.b, viewID, param.color.a,
                                      param.fontSize);
                } else {
                    CC_DRAW_CONTEXT context = param.context;
                    ecvDisplayTools::GetContext(context);
                    ecvTextParam tParam;
                    tParam.display3D = false;
                    tParam.font = ecvDisplayTools::GetLabelDisplayFont();
                    tParam.opacity = param.color.a;
                    tParam.text = param.text;
                    tParam.textPos =
                            CCVector3d(param.rect.x(), param.rect.y(), 0.0);
                    context.textDefaultCol =
                            ecvColor::FromRgbafToRgb(param.color);
                    context.textParam = tParam;
                    context.viewID = param.viewID;
                    if (vis) vis->displayText(context);
                }
            }
        } break;

        case WIDGETS_TYPE::WIDGET_LINE_3D:
            if (param.lineWidget.valid && !vis->contains(viewID)) {
                unsigned char lineWidth =
                        (unsigned char)param.lineWidget.lineWidth;
                ecvColor::Rgbf lineColor =
                        ecvTools::TransFormRGB(param.lineWidget.lineColor);
                vis->addLine(
                        param.lineWidget.lineSt.x, param.lineWidget.lineSt.y,
                        param.lineWidget.lineSt.z, param.lineWidget.lineEd.x,
                        param.lineWidget.lineEd.y, param.lineWidget.lineEd.z,
                        lineColor.r, lineColor.g, lineColor.b, viewID,
                        viewport);
                vis->setLineWidth(lineWidth, viewID, viewport);
            }
            break;
        case WIDGETS_TYPE::WIDGET_POLYLINE: {
            ccPolyline* poly =
                    param.entity ? ccHObjectCaster::ToPolyline(param.entity)
                                 : nullptr;
            if (!poly || poly->size() < 2) break;

            CC_DRAW_CONTEXT ctx;
            ecvDisplayTools::GetContext(ctx);
            ctx.display = param.context.display;
            if (!ctx.display) {
                ctx.display = ecvViewManager::instance().getEffectiveView();
            }
            ctx.defaultViewPort = viewport;
            ctx.viewID = poly->getViewId();
            ctx.drawingFlags = poly->is2DMode()
                                       ? (CC_DRAW_2D | CC_DRAW_FOREGROUND)
                                       : (CC_DRAW_3D | CC_DRAW_FOREGROUND);
            if (poly->isColorOverridden()) {
                ctx.defaultPolylineColor = poly->getTempColor();
            } else if (poly->colorsShown()) {
                ctx.defaultPolylineColor = poly->getColor();
            }
            if (poly->getWidth() != 0) {
                ctx.currentLineWidth = poly->getWidth();
            }
            drawPolygon(ctx, poly);
        } break;
        case WIDGETS_TYPE::WIDGET_SPHERE:
            if (!vis->contains(viewID)) {
                vis->addSphere(param.center.x, param.center.y, param.center.z,
                               param.radius, param.color.r, param.color.g,
                               param.color.b, viewID, viewport);
            }
            break;

        case WIDGETS_TYPE::WIDGET_POINT:
            if (!vis->contains(viewID)) {
                float pxSize = param.pointSize > 0 ? param.pointSize : 10.0f;
                vis->addPointSprite(param.center.x, param.center.y,
                                    param.center.z, pxSize, param.color.r,
                                    param.color.g, param.color.b, viewID,
                                    viewport);
            }
            break;

        case WIDGETS_TYPE::WIDGET_SCALAR_BAR:
            if (!vis->updateScalarBar(param.context)) {
                vis->addScalarBar(param.context);
            }
            break;
        case WIDGETS_TYPE::WIDGET_CAPTION:
            if (!vis->updateCaption(CVTools::FromQString(param.text), param.pos,
                                    param.center, param.color.r, param.color.g,
                                    param.color.b, param.color.a,
                                    param.fontSize, viewID, viewport)) {
                vis->addCaption(CVTools::FromQString(param.text), param.pos,
                                param.center, param.color.r, param.color.g,
                                param.color.b, param.color.a, param.fontSize,
                                viewID, param.handleEnabled, viewport);
            }
            break;
        case WIDGETS_TYPE::WIDGET_LINE_2D:
            if (m_visualizer2D) {
                m_visualizer2D->addLine(param.p1.x(), param.p1.y(),
                                        param.p2.x(), param.p2.y(),
                                        param.color.r, param.color.g,
                                        param.color.b, viewID, param.color.a);
            }
            break;
        case WIDGETS_TYPE::WIDGET_POLYLINE_2D:
            if (m_visualizer2D) {
                if (param.entity &&
                    param.entity->isKindOf(CV_TYPES::POLY_LINE)) {
                    ccPolyline* poly =
                            ccHObjectCaster::ToPolyline(param.entity);
                    if (poly->size() <= 1) {
                        return;
                    }

                    if (!poly->is2DMode()) {
                        CVLog::Warning(
                                "[VtkDisplayTools::drawWidgets] draw mode is "
                                "incompatible with entity mode!");
                        return;
                    }

                    viewID = CVTools::FromQString(poly->getViewId());

                    ecvColor::Rgbf color = ecvColor::FromRgb(ecvColor::green);
                    if (poly->isColorOverridden()) {
                        color = ecvColor::FromRgb(poly->getTempColor());
                    } else if (poly->colorsShown()) {
                        color = ecvColor::FromRgb(poly->getColor());
                    }

                    for (unsigned i = 1; i < poly->size(); ++i) {
                        const CCVector3* p1 = poly->getPoint(i - 1);
                        const CCVector3* p2 = poly->getPoint(i);

                        m_visualizer2D->addLine(static_cast<int>(p1->x),
                                                static_cast<int>(p1->y),
                                                static_cast<int>(p2->x),
                                                static_cast<int>(p2->y),
                                                color.r, color.g, color.b,
                                                viewID, param.opacity);
                    }

                    if (poly->isClosed()) {
                        m_visualizer2D->addLine(
                                static_cast<int>(
                                        poly->getPoint(poly->size() - 1)->x),
                                static_cast<int>(
                                        poly->getPoint(poly->size() - 1)->y),
                                static_cast<int>(poly->getPoint(0)->x),
                                static_cast<int>(poly->getPoint(0)->y), color.r,
                                color.g, color.b, viewID, param.opacity);
                    }
                }
            } else {
                if (param.entity &&
                    param.entity->isKindOf(CV_TYPES::POLY_LINE)) {
                    ccPolyline* poly =
                            ccHObjectCaster::ToPolyline(param.entity);
                    if (poly->size() <= 1) {
                        return;
                    }

                    ecvDisplayTools::DrawWidgets(
                            WIDGETS_PARAMETER(poly,
                                              WIDGETS_TYPE::WIDGET_POLYLINE),
                            true);
                }
            }
            break;
        case WIDGETS_TYPE::WIDGET_TRIANGLE_2D:
            if (m_visualizer2D) {
                // edge 1
                m_visualizer2D->addLine(param.p1.x(), param.p1.y(),
                                        param.p2.x(), param.p2.y(),
                                        param.color.r, param.color.g,
                                        param.color.b, viewID, param.color.a);

                // edge 2
                m_visualizer2D->addLine(param.p2.x(), param.p2.y(),
                                        param.p3.x(), param.p3.y(),
                                        param.color.r, param.color.g,
                                        param.color.b, viewID, param.color.a);
                if (param.p4.x() >= 0 && param.p4.y() >= 0) {
                    // edge 3
                    m_visualizer2D->addLine(
                            param.p3.x(), param.p3.y(), param.p4.x(),
                            param.p4.y(), param.color.r, param.color.g,
                            param.color.b, viewID, param.color.a);
                    // edge 4
                    m_visualizer2D->addLine(
                            param.p4.x(), param.p4.y(), param.p1.x(),
                            param.p1.y(), param.color.r, param.color.g,
                            param.color.b, viewID, param.color.a);
                } else {
                    // edge 3
                    m_visualizer2D->addLine(
                            param.p3.x(), param.p3.y(), param.p1.x(),
                            param.p1.y(), param.color.r, param.color.g,
                            param.color.b, viewID, param.color.a);
                }
            }
            break;
        case WIDGETS_TYPE::WIDGET_POINTS_2D: {
            bool isSecondaryPt = param.context.display &&
                                 param.context.display !=
                                         static_cast<ecvDisplayTools*>(this);
            Visualization::ImageVis* ptVis2D = nullptr;
            if (isSecondaryPt) {
                auto* glView = dynamic_cast<vtkGLView*>(param.context.display);
                if (glView && glView->getImageVis()) {
                    ptVis2D = glView->getImageVis().get();
                }
            } else {
                ptVis2D = m_visualizer2D ? m_visualizer2D.get() : nullptr;
            }
            if (ptVis2D) {
                Eigen::Array<unsigned char, 3, 1> color(
                        param.color.r, param.color.g, param.color.b);
                ptVis2D->markPoint(param.rect.x(), param.rect.y(), color, color,
                                   param.radius, viewID, param.color.a);
            }
        } break;
        case WIDGETS_TYPE::WIDGET_RECTANGLE_2D: {
            bool isSecondaryRect = param.context.display &&
                                   param.context.display !=
                                           static_cast<ecvDisplayTools*>(this);
            Visualization::ImageVis* rectVis2D = nullptr;
            if (isSecondaryRect) {
                auto* glView = dynamic_cast<vtkGLView*>(param.context.display);
                if (glView && glView->getImageVis()) {
                    rectVis2D = glView->getImageVis().get();
                }
            } else {
                rectVis2D = m_visualizer2D ? m_visualizer2D.get() : nullptr;
            }

            if (rectVis2D) {
                int minX = std::max(param.rect.x(), 0);
                int maxX = std::min(minX + param.rect.width(),
                                    rectVis2D->getSize()[0]);
                int minY = std::max(param.rect.y(), 0);
                int maxY = std::min(minY + param.rect.height(),
                                    rectVis2D->getSize()[1]);
                if (param.filled) {
                    rectVis2D->addFilledRectangle(minX, maxX, minY, maxY,
                                                  param.color.r, param.color.g,
                                                  param.color.b, viewID,
                                                  param.color.a);
                } else {
                    rectVis2D->addRectangle(minX, maxX, minY, maxY,
                                            param.color.r, param.color.g,
                                            param.color.b, viewID,
                                            param.color.a);
                }
            }
        } break;
        case WIDGETS_TYPE::WIDGET_CIRCLE_2D:
            if (m_visualizer2D) {
                m_visualizer2D->addCircle(param.rect.x(), param.rect.y(),
                                          param.radius, param.color.r,
                                          param.color.g, param.color.b, viewID,
                                          param.color.a);
            }
            break;
        case WIDGETS_TYPE::WIDGET_IMAGE: {
            bool isSecondaryImg = param.context.display &&
                                  param.context.display !=
                                          static_cast<ecvDisplayTools*>(this);
            Visualization::ImageVis* imgVis2D = nullptr;
            if (isSecondaryImg) {
                auto* glView = dynamic_cast<vtkGLView*>(param.context.display);
                if (glView && glView->getImageVis()) {
                    imgVis2D = glView->getImageVis().get();
                }
            } else {
                imgVis2D = m_visualizer2D ? m_visualizer2D.get() : nullptr;
            }
            if (imgVis2D && !param.image.isNull()) {
                imgVis2D->addImageOverlay(param.image, param.rect.x(),
                                          param.rect.y(), param.rect.width(),
                                          param.rect.height(), viewID,
                                          param.opacity);
            }
        } break;
        default:
            break;
    }
}

void VtkDisplayTools::displayText(const CC_DRAW_CONTEXT& context) {
    VtkVis* vis = resolveVisualizer(context.display);
    bool isSecondaryView =
            context.display &&
            context.display != static_cast<ecvDisplayTools*>(this);

    if (isSecondaryView && vis) {
        // For secondary views (e.g. Comparative sub-views), use vtkTextActor
        // via VtkVis instead of ImageVis (vtkContext2D). The vtkContext2D text
        // path shares a font texture atlas (vtkFreeTypeTools singleton) across
        // render windows, which causes garbled text when multiple windows
        // exist. vtkTextActor manages textures per-renderer and avoids this
        // issue.
        ecvTextParam textParam = context.textParam;
        std::string text = CVTools::FromQString(textParam.text);
        std::string groupID = CVTools::FromQString(context.viewID);
        std::string actorID = groupID + "#" + text;
        int xPos = static_cast<int>(textParam.textPos.x);
        int yPos = static_cast<int>(textParam.textPos.y);
        ecvColor::Rgbf textColor =
                ecvTools::TransFormRGB(context.textDefaultCol);
        int viewport = context.defaultViewPort;

        if (!vis->updateText(text, xPos, yPos, actorID)) {
            vis->addText(text, xPos, yPos, textParam.font.pointSize(),
                         textColor.r, textColor.g, textColor.b, actorID,
                         viewport);
        }
        auto smap = vis->getShapeActorMap();
        auto it = smap->find(actorID);
        if (it != smap->end()) {
            auto textActor = vtkTextActor::SafeDownCast(it->second);
            if (textActor) {
                auto tp = textActor->GetTextProperty();
                if (textParam.bkgAlpha > 0) {
                    tp->SetBackgroundColor(textParam.bkgColor[0],
                                           textParam.bkgColor[1],
                                           textParam.bkgColor[2]);
                    tp->SetBackgroundOpacity(textParam.bkgAlpha);
                } else {
                    tp->SetBackgroundColor(0.15, 0.15, 0.15);
                    tp->SetBackgroundOpacity(0.7);
                }
                tp->ShadowOn();
            }
        }
    } else {
        Visualization::ImageVis* txtVis2D =
                m_visualizer2D ? m_visualizer2D.get() : nullptr;
        if (txtVis2D) {
            ecvTextParam textParam = context.textParam;
            std::string viewID = CVTools::FromQString(context.viewID);
            std::string text = CVTools::FromQString(textParam.text);
            ecvColor::Rgbf textColor =
                    ecvTools::TransFormRGB(context.textDefaultCol);
            // RenderText / glViewport use physical pixels; ImageVis
            // vtkContext2D shares the same render-window coordinate space.
            txtVis2D->addText(static_cast<unsigned int>(textParam.textPos.x),
                              static_cast<unsigned int>(textParam.textPos.y),
                              text, textColor.r, textColor.g, textColor.b,
                              viewID, textParam.opacity,
                              textParam.font.pointSize(),
                              textParam.font.bold());
        } else if (vis) {
            vis->displayText(context);
        }
    }
}

void VtkDisplayTools::toggle2Dviewer(bool state) {
    // ParaView-style: toggle interaction mode on the 3D viewer, not swap
    // the interactor on the 2D visualizer.  'state == true' means "switch
    // to 2D"; 'false' means "back to 3D".
    if (m_visualizer3D) {
        m_visualizer3D->setInteractionMode(state ? VtkVis::INTERACTION_MODE_2D
                                                 : VtkVis::INTERACTION_MODE_3D);
    }
}

QString VtkDisplayTools::pick2DLabel(int x, int y) {
    if (m_visualizer2D) {
        return m_visualizer2D->pickItem(x, y).c_str();
    }

    return QString();
}

QString VtkDisplayTools::pick3DItem(int x, int y) {
    if (m_visualizer3D) {
        return m_visualizer3D->pickItem(x, y).c_str();
    }

    return QString();
}

QString VtkDisplayTools::pickObject(double x, double y) {
    if (m_visualizer3D) {
        vtkActor* pickedActor = m_visualizer3D->pickActor(x, y);
        if (pickedActor) {
            if (pickedActor) {
                return m_visualizer3D->getIdByActor(pickedActor).c_str();
            } else {
                return "-1";
            }
        }
    }
    return "-1";
}

QImage VtkDisplayTools::renderToImage(int zoomFactor,
                                      bool renderOverlayItems,
                                      bool silent,
                                      int viewport) {
    if (m_visualizer3D) {
        return m_visualizer3D->renderToImage(zoomFactor, renderOverlayItems,
                                             silent, viewport);
    } else {
        if (!silent)
            CVLog::Error(
                    "[VtkDisplayTools::renderToImage] VtkVis Initialization "
                    "failed! (not enough memory?)");
        return QImage();
    }
}

double VtkDisplayTools::getParallelScale(int viewport) {
    if (m_visualizer3D) {
        return m_visualizer3D->getParallelScale(viewport);
    }

    return -1;
}

void VtkDisplayTools::setParallelScale(double scale, int viewport) {
    if (m_visualizer3D) {
        m_visualizer3D->setParallelScale(scale, viewport);
    }
}

void VtkDisplayTools::getProjectionMatrix(double* projArray, int viewport) {
    Eigen::Matrix4d projMat;
    m_visualizer3D->getProjectionTransformMatrix(projMat, viewport);
    double* tempArray = projMat.data();
    projArray[0] = tempArray[0];
    projArray[1] = tempArray[1];
    projArray[2] = tempArray[2];
    projArray[3] = tempArray[3];
    projArray[4] = tempArray[4];
    projArray[5] = tempArray[5];
    projArray[6] = tempArray[6];
    projArray[7] = tempArray[7];
    projArray[8] = tempArray[8];
    projArray[9] = tempArray[9];
    projArray[10] = tempArray[10];
    projArray[11] = tempArray[11];
    projArray[12] = tempArray[12];
    projArray[13] = tempArray[13];
    projArray[14] = tempArray[14];
    projArray[15] = tempArray[15];
}

void VtkDisplayTools::getViewMatrix(double* ViewArray, int viewport) {
    Eigen::Matrix4d viewMat;
    m_visualizer3D->getModelViewTransformMatrix(viewMat, viewport);
    double* tempArray = viewMat.data();
    ViewArray[0] = tempArray[0];
    ViewArray[1] = tempArray[1];
    ViewArray[2] = tempArray[2];
    ViewArray[3] = tempArray[3];
    ViewArray[4] = tempArray[4];
    ViewArray[5] = tempArray[5];
    ViewArray[6] = tempArray[6];
    ViewArray[7] = tempArray[7];
    ViewArray[8] = tempArray[8];
    ViewArray[9] = tempArray[9];
    ViewArray[10] = tempArray[10];
    ViewArray[11] = tempArray[11];
    ViewArray[12] = tempArray[12];
    ViewArray[13] = tempArray[13];
    ViewArray[14] = tempArray[14];
    ViewArray[15] = tempArray[15];
}

void VtkDisplayTools::setViewMatrix(const ccGLMatrixd& viewMat, int viewport) {
    if (m_visualizer3D) {
        m_visualizer3D->setModelViewMatrix(viewMat, viewport);
    }
}

void VtkDisplayTools::changeOpacity(double opacity,
                                    const std::string& viewID,
                                    int viewport) {
    if (m_visualizer2D) {
        m_visualizer2D->changeOpacity(opacity, viewID);
    }
}

void VtkDisplayTools::changeEntityProperties(PROPERTY_PARAM& param) {
    std::string viewId = CVTools::FromQString(param.viewId);
    int viewport = param.viewport;

    // Route to the VtkVis that actually owns the entity's actors.
    VtkVis* vis = findVisByActorIdOrActive(viewId);
    if (!vis) return;

    switch (param.property) {
        case PROPERTY_MODE::ECV_POINTSSIZE_PROPERTY: {
            vis->setPointSize(param.pointSize, viewId, viewport);
        } break;
        case PROPERTY_MODE::ECV_LINEWITH_PROPERTY: {
            vis->setLineWidth(static_cast<unsigned char>(param.lineWidth),
                              viewId, viewport);
        } break;
        case PROPERTY_MODE::ECV_COLOR_PROPERTY: {
            ecvColor::Rgbf colf = ecvTools::TransFormRGB(param.color);
            switch (param.entityType) {
                case ENTITY_TYPE::ECV_POINT_CLOUD:
                case ENTITY_TYPE::ECV_MESH: {
                    vis->setPointCloudUniqueColor(colf.r, colf.g, colf.b,
                                                  viewId, viewport);
                } break;
                case ENTITY_TYPE::ECV_SHAPE:
                case ENTITY_TYPE::ECV_LINES_3D: {
                    vis->setShapeUniqueColor(colf.r, colf.g, colf.b, viewId,
                                             viewport);
                } break;
                default:
                    break;
            }
        } break;
        case PROPERTY_MODE::ECV_OPACITY_PROPERTY: {
            switch (param.entityType) {
                case ENTITY_TYPE::ECV_POINT_CLOUD: {
                    vis->setPointCloudOpacity(param.opacity, viewId, viewport);
                } break;
                case ENTITY_TYPE::ECV_MESH: {
                    vis->setMeshOpacity(param.opacity, viewId, viewport);
                } break;
                case ENTITY_TYPE::ECV_SHAPE:
                case ENTITY_TYPE::ECV_LINES_3D: {
                    vis->setShapeOpacity(param.opacity, viewId, viewport);
                } break;
                default:
                    break;
            }
        } break;
        case PROPERTY_MODE::ECV_SHADING_PROPERTY: {
            switch (param.entityType) {
                case ENTITY_TYPE::ECV_POINT_CLOUD:
                case ENTITY_TYPE::ECV_MESH: {
                    vis->setMeshShadingMode(param.shadingMode, viewId,
                                            viewport);
                } break;
                case ENTITY_TYPE::ECV_SHAPE:
                case ENTITY_TYPE::ECV_LINES_3D: {
                    vis->setShapeShadingMode(param.shadingMode, viewId,
                                             viewport);
                } break;
                default:
                    break;
            }
        } break;

        default:
            break;
    }
}

void VtkDisplayTools::transformCameraView(const ccGLMatrixd& viewMat) {
    getQVtkWidget()->transformCameraView(viewMat.data());
}

void VtkDisplayTools::transformCameraProjection(const ccGLMatrixd& projMat) {
    getQVtkWidget()->transformCameraProjection(projMat.data());
}

// ============================================================================
// View Properties Implementation (ParaView-compatible)
// ============================================================================

void VtkDisplayTools::ToggleCameraOrientationWidget(bool show) {
    auto* activeView = ecvViewManager::instance().getActiveView();
    auto* glView = activeView ? dynamic_cast<vtkGLView*>(activeView) : nullptr;
    auto* vis = glView ? dynamic_cast<VtkVis*>(glView->getVisualizer3D())
                       : m_visualizer3D.get();
    if (!vis) return;
    vis->ToggleCameraOrientationWidget(show);
}

bool VtkDisplayTools::IsCameraOrientationWidgetShown() const {
    if (!m_visualizer3D) {
        return false;
    }

    // Delegate to VtkVis
    return m_visualizer3D->IsCameraOrientationWidgetShown();
}

// Override base class virtual methods (ecvDisplayTools interface)
void VtkDisplayTools::toggleCameraOrientationWidget(bool show) {
    ToggleCameraOrientationWidget(show);
}

bool VtkDisplayTools::isCameraOrientationWidgetShown() const {
    return IsCameraOrientationWidgetShown();
}

void VtkDisplayTools::setLightIntensity(double intensity) {
    auto* activeView = ecvViewManager::instance().getActiveView();
    auto* glView = activeView ? dynamic_cast<vtkGLView*>(activeView) : nullptr;
    auto* vis = glView ? dynamic_cast<VtkVis*>(glView->getVisualizer3D())
                       : m_visualizer3D.get();
    if (!vis) return;
    vis->setLightIntensity(intensity);
}

double VtkDisplayTools::getLightIntensity() const {
    if (!m_visualizer3D) {
        return 1.0;  // Default intensity
    }

    // Delegate to VtkVis
    return m_visualizer3D->getLightIntensity();
}

void VtkDisplayTools::setObjectLightIntensity(const QString& viewID,
                                              double intensity) {
    std::string id = CVTools::FromQString(viewID);
    VtkVis* vis = findVisByActorIdOrActive(id);
    if (!vis) return;
    vis->setObjectLightIntensity(id, intensity);
}

double VtkDisplayTools::getObjectLightIntensity(const QString& viewID) const {
    std::string id = CVTools::FromQString(viewID);
    VtkVis* vis = findVisByActorId(id);
    if (!vis) {
        return 1.0;
    }
    return vis->getObjectLightIntensity(id);
}

// ============================================================================
// Axes Grid Properties (ParaView-compatible)
// ============================================================================

// Unified struct-based interface (Simplified - direct pass-through)
void VtkDisplayTools::setDataAxesGridProperties(const QString& viewID,
                                                const AxesGridProperties& props,
                                                int viewport) {
    std::string id = CVTools::FromQString(viewID);
    VtkVis* vis = findVisByActorIdOrActive(id);
    if (!vis) return;
    vis->SetDataAxesGridProperties(id, props);
}

void VtkDisplayTools::getDataAxesGridProperties(const QString& viewID,
                                                AxesGridProperties& props,
                                                int viewport) const {
    std::string id = CVTools::FromQString(viewID);
    VtkVis* vis = findVisByActorId(id);
    if (!vis) {
        props = AxesGridProperties();
        return;
    }
    vis->GetDataAxesGridProperties(id, props);
}

}  // namespace Visualization