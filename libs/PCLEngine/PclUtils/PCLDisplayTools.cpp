// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "PCLDisplayTools.h"

// PCLModules
#include "PCLConv.h"
#include "cc2sm.h"
#include "sm2cc.h"

// CV_CORE_LIB
#include <CVGeom.h>
#include <CVTools.h>
#include <ecvGLMatrix.h>

// ECV_DB_LIB
#include <ecvBBox.h>
#include <ecvCameraSensor.h>
#include <ecvGenericMesh.h>
#include <ecvHObjectCaster.h>
#include <ecvImage.h>
#include <ecvMaterialSet.h>
#include <ecvPointCloud.h>
#include <ecvPolyline.h>
#include <ecvScalarField.h>
#include <ecvSensor.h>

// LOCAL
#include "VTKExtensions/InteractionStyle/vtkCustomInteractorStyle.h"

// PCL
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

// VTK
#include <vtkGenericOpenGLRenderWindow.h>

// SYSTEM
#include <assert.h>

void PCLDisplayTools::registerVisualizer(QMainWindow* win, bool stereoMode) {
    this->m_vtkWidget = new QVTKWidgetCustom(win, this, stereoMode);
    SetMainScreen(this->m_vtkWidget);
    SetCurrentScreen(this->m_vtkWidget);

    if (!m_visualizer3D) {
        auto renderer = vtkSmartPointer<vtkRenderer>::New();
        auto renderWindow =
                vtkSmartPointer<vtkGenericOpenGLRenderWindow>::New();
        renderWindow->AddRenderer(renderer);
        auto interactorStyle =
                vtkSmartPointer<VTKExtensions::vtkCustomInteractorStyle>::New();
        m_visualizer3D.reset(new PclUtils::PCLVis(
                renderer, renderWindow, interactorStyle, "3Dviewer", false));
        // m_visualizer3D.reset(new PclUtils::PCLVis(interactorStyle,
        // "3Dviewer", false)); // deprecated!
    }

    getQVtkWidget()->SetRenderWindow(m_visualizer3D->getRenderWindow());
    m_visualizer3D->setupInteractor(getQVtkWidget()->GetInteractor(),
                                    getQVtkWidget()->GetRenderWindow());
    getQVtkWidget()->initVtk(m_visualizer3D->getRenderWindowInteractor(),
                             false);
    m_visualizer3D->initialize();

    if (ecvDisplayTools::USE_2D) {
        if (!m_visualizer2D) {
            m_visualizer2D.reset(new PclUtils::ImageVis("2Dviewer", false));
        }

        m_visualizer2D->setRender(getQVtkWidget()->getVtkRender());
        m_visualizer2D->setupInteractor(getQVtkWidget()->GetInteractor(),
                                        getQVtkWidget()->GetRenderWindow());
    } else {
        m_visualizer2D = nullptr;
    }
}

PCLDisplayTools::~PCLDisplayTools() {
    if (this->m_vtkWidget) {
        delete this->m_vtkWidget;
        this->m_vtkWidget = nullptr;
    }
}

void PCLDisplayTools::drawPointCloud(const CC_DRAW_CONTEXT& context,
                                     ccPointCloud* ecvCloud) {
    std::string viewID = CVTools::FromQString(context.viewID);
    int viewport = context.defaultViewPort;
    bool firstShow = !m_visualizer3D->contains(viewID);

    if (ecvCloud->isRedraw() || firstShow) {
        if (firstShow || checkEntityNeedUpdate(viewID, ecvCloud)) {
            PCLCloud::Ptr pclCloud =
                    cc2smReader(ecvCloud, true)
                            .getAsSM(!context.drawParam.showSF);
            if (!pclCloud) {
                return;
            }
            m_visualizer3D->draw(context, pclCloud);
            m_visualizer3D->updateNormals(context, pclCloud);
        } else {
            m_visualizer3D->resetScalarColor(viewID, true, viewport);
            if (!updateEntityColor(context, ecvCloud)) {
                PCLCloud::Ptr pclCloud =
                        cc2smReader(ecvCloud, true)
                                .getAsSM(!context.drawParam.showSF);
                if (!pclCloud) {
                    return;
                }
                m_visualizer3D->draw(context, pclCloud);
                m_visualizer3D->updateNormals(context, pclCloud);
            } else {
                if (context.drawParam.showNorms) {
                    PCLCloud::Ptr pointNormals =
                            cc2smReader(ecvCloud).getPointNormals();
                    m_visualizer3D->updateNormals(context, pointNormals);
                } else {
                    m_visualizer3D->updateNormals(context, nullptr);
                }
            }
        }
    }

    if (m_visualizer3D->contains(viewID)) {
        m_visualizer3D->setPointSize(context.defaultPointSize, viewID,
                                     viewport);

        if ((!context.drawParam.showColors && !context.drawParam.showSF) ||
            ecvCloud->isColorOverridden()) {
            ecvColor::Rgbf pointUniqueColor =
                    ecvTools::TransFormRGB(context.pointsCurrentCol);
            m_visualizer3D->setPointCloudUniqueColor(
                    pointUniqueColor.r, pointUniqueColor.g, pointUniqueColor.b,
                    viewID, viewport);
        }
    }
}

void PCLDisplayTools::updateMeshTextures(const CC_DRAW_CONTEXT& context,
                                         const ccGenericMesh* mesh) {
    std::string viewID = CVTools::FromQString(context.viewID);
    bool firstShow = !m_visualizer3D->contains(viewID);
    if (firstShow) {
        drawMesh(const_cast<CC_DRAW_CONTEXT&>(context),
                 const_cast<ccGenericMesh*>(mesh));
    } else {
        // materials & textures
        bool applyMaterials = (mesh->hasMaterials() && mesh->materialsShown());
        bool showTextures = (mesh->hasTextures() && mesh->materialsShown());
        if (applyMaterials || showTextures) {
            // materials
            const ccMaterialSet* materials = mesh->getMaterialSet();
            assert(materials);
            std::vector<pcl::TexMaterial> tex_materials;
            for (std::size_t newMatlIndex = 0; newMatlIndex < materials->size();
                 ++newMatlIndex) {
                PCLMaterial pcl_material;
                cc2smReader::ConVertToPCLMaterial(materials->at(newMatlIndex),
                                                  pcl_material);
                tex_materials.push_back(pcl_material);
            }
            if (!m_visualizer3D->updateTexture(context, tex_materials)) {
                CVLog::Warning(QString("Update mesh texture failed!"));
            }
        } else {
            CVLog::Warning(
                    QString("Mesh texture has not been shown, please toggle it "
                            "to be shown!"));
        }
    }
}

void PCLDisplayTools::drawMesh(CC_DRAW_CONTEXT& context, ccGenericMesh* mesh) {
    std::string viewID = CVTools::FromQString(context.viewID);
    int viewport = context.defaultViewPort;
    context.visFiltering = true;
    bool firstShow = !m_visualizer3D->contains(viewID);

    if (mesh->isRedraw() || firstShow) {
        ccPointCloud* ecvCloud = ccHObjectCaster::ToPointCloud(mesh);
        if (!ecvCloud) return;

        // materials & textures
        bool applyMaterials = (mesh->hasMaterials() && mesh->materialsShown());
        bool lodEnabled = false;
        bool showTextures =
                (mesh->hasTextures() && mesh->materialsShown() && !lodEnabled);

        if (firstShow || checkEntityNeedUpdate(viewID, ecvCloud)) {
            if (applyMaterials || showTextures) {
                PCLTextureMesh::Ptr textureMesh =
                        cc2smReader(ecvCloud, true).getPclTextureMesh(mesh);
                if (textureMesh) {
                    m_visualizer3D->draw(context, textureMesh);
                } else {
                    PCLMesh::Ptr pclMesh =
                            cc2smReader(ecvCloud, true).getPclMesh(mesh);
                    if (!pclMesh) return;
                    m_visualizer3D->draw(context, pclMesh);
                }

            } else {
                PCLMesh::Ptr pclMesh =
                        cc2smReader(ecvCloud, true).getPclMesh(mesh);
                if (!pclMesh) return;
                m_visualizer3D->draw(context, pclMesh);
            }
        } else {
            m_visualizer3D->resetScalarColor(viewID, true, viewport);
            if (!updateEntityColor(context, ecvCloud)) {
                if (applyMaterials || showTextures) {
                    PCLTextureMesh::Ptr textureMesh =
                            cc2smReader(ecvCloud, true).getPclTextureMesh(mesh);
                    if (textureMesh) {
                        m_visualizer3D->draw(context, textureMesh);
                    } else {
                        PCLMesh::Ptr pclMesh =
                                cc2smReader(ecvCloud, true).getPclMesh(mesh);
                        if (!pclMesh) return;
                        m_visualizer3D->draw(context, pclMesh);
                    }
                } else {
                    PCLMesh::Ptr pclMesh =
                            cc2smReader(ecvCloud, true).getPclMesh(mesh);
                    if (!pclMesh) return;
                    m_visualizer3D->draw(context, pclMesh);
                }
            }
        }
        m_visualizer3D->transformEntities(context);
    }

    if (m_visualizer3D->contains(viewID)) {
        m_visualizer3D->setMeshRenderingMode(context.meshRenderingMode, viewID,
                                             viewport);

        if ((!context.drawParam.showColors && !context.drawParam.showSF) ||
            mesh->isColorOverridden()) {
            ecvColor::Rgbf meshColor =
                    ecvTools::TransFormRGB(context.defaultMeshColor);
            m_visualizer3D->setPointCloudUniqueColor(
                    meshColor.r, meshColor.g, meshColor.b, viewID, viewport);
        }
        m_visualizer3D->setPointCloudOpacity(context.opacity, viewID, viewport);
    }
}

void PCLDisplayTools::drawPolygon(const CC_DRAW_CONTEXT& context,
                                  ccPolyline* polyline) {
    std::string viewID = CVTools::FromQString(context.viewID);
    bool firstShow = !m_visualizer3D->contains(viewID);
    int viewport = context.defaultViewPort;

    if (polyline->isRedraw() || firstShow) {
        PCLPolygon::Ptr pclPolygon = cc2smReader(true).getPclPolygon(polyline);
        if (!pclPolygon) return;
        m_visualizer3D->draw(context, pclPolygon, polyline->isClosed());
    }

    if (m_visualizer3D->contains(viewID)) {
        ecvColor::Rgbf polygonColor =
                ecvTools::TransFormRGB(context.defaultPolylineColor);
        m_visualizer3D->setShapeUniqueColor(polygonColor.r, polygonColor.g,
                                            polygonColor.b, viewID, viewport);
        m_visualizer3D->setLineWidth(context.currentLineWidth, viewID,
                                     viewport);
        m_visualizer3D->setLightMode(viewID, viewport);
    }
}

void PCLDisplayTools::drawLines(const CC_DRAW_CONTEXT& context,
                                cloudViewer::geometry::LineSet* lineset) {
    std::string viewID = CVTools::FromQString(context.viewID);
    bool firstShow = !m_visualizer3D->contains(viewID);
    int viewport = context.defaultViewPort;

    if (lineset->isRedraw() || firstShow) {
        m_visualizer3D->draw(context, lineset);
        m_visualizer3D->transformEntities(context);
    }

    if (m_visualizer3D->contains(viewID)) {
        if (lineset->isColorOverridden() || !lineset->HasColors()) {
            ecvColor::Rgbf polygonColor =
                    ecvTools::TransFormRGB(context.defaultPolylineColor);
            m_visualizer3D->setShapeUniqueColor(polygonColor.r, polygonColor.g,
                                                polygonColor.b, viewID,
                                                viewport);
        }

        m_visualizer3D->setLineWidth(context.currentLineWidth, viewID,
                                     viewport);
        m_visualizer3D->setLightMode(viewID, viewport);
    }
}

void PCLDisplayTools::drawImage(const CC_DRAW_CONTEXT& context,
                                ccImage* image) {
    Q_UNUSED(context);
    Q_UNUSED(image);

    if (!m_visualizer2D) return;

#if 1
    std::string viewID = CVTools::FromQString(context.viewID);
    bool firstShow = !m_visualizer2D->contains(viewID);

    if (image->isRedraw() || firstShow) {
        m_visualizer2D->showRGBImage(image->data().bits(), image->getW(),
                                     image->getH(), viewID,
                                     image->getOpacity());
        // m_visualizer2D->addRGBImage(image->data().bits(), 0, 0,
        // image->getW(),
        //                             image->getH(), viewID,
        //                             image->getOpacity());
    }
    m_visualizer2D->changeOpacity(viewID, image->getOpacity());
#else
    CVLog::Warning(QString("Image showing has not been supported!"));
#endif
}

void PCLDisplayTools::drawSensor(const CC_DRAW_CONTEXT& context,
                                 ccSensor* sensor) {
    if (!sensor) {
        return;
    }

    std::string viewID = CVTools::FromQString(context.viewID);
    bool firstShow = !m_visualizer3D->contains(viewID);
    int viewport = context.defaultViewPort;

    if (sensor->isRedraw() || firstShow) {
        m_visualizer3D->draw(context, sensor);
        m_visualizer3D->transformEntities(context);
    }

    if (m_visualizer3D->contains(viewID)) {
        m_visualizer3D->setLineWidth(context.currentLineWidth, viewID,
                                     viewport);
    }
}

bool PCLDisplayTools::updateEntityColor(const CC_DRAW_CONTEXT& context,
                                        ccHObject* ent) {
#ifdef _DEBUG
    CVTools::TimeStart();
#endif  // _DEBUG
    ccPointCloud* cloud = ccHObjectCaster::ToPointCloud(ent);
    if (!cloud) {
        return false;
    }

    std::string viewID = CVTools::FromQString(context.viewID);
    vtkActor* modelActor = m_visualizer3D->getActorById(viewID);
    if (!modelActor) {
        return false;
    }

    // Get the current poly data
    vtkSmartPointer<vtkPolyData> polydata =
            reinterpret_cast<vtkPolyDataMapper*>(modelActor->GetMapper())
                    ->GetInput();
    if (!polydata) return (false);

    // Get the colors from the handler
    bool has_colors = false;
    double minmax[2];
    vtkSmartPointer<vtkDataArray> scalars;
    cc2smReader converter(cloud, true);
    unsigned old_points_num =
            static_cast<unsigned>(polydata->GetNumberOfPoints());
    unsigned new_points_num = converter.getvisibilityNum();
    if (old_points_num != new_points_num) {
        return false;
    }

    if (!context.drawParam.showColors && !context.drawParam.showSF) {
        return false;
    }

    if (converter.getvtkScalars(scalars, context.drawParam.showSF)) {
        // Update the data
        polydata->GetPointData()->SetScalars(scalars);
        scalars->GetRange(minmax);
        has_colors = true;
    }

    if (has_colors) {
#if VTK_RENDERING_BACKEND_OPENGL_VERSION < 2
        modelActor->GetMapper()->ImmediateModeRenderingOff();
#endif
        modelActor->GetMapper()->SetScalarRange(minmax);
        // Update the mapper
#if VTK_MAJOR_VERSION < 6
        reinterpret_cast<vtkPolyDataMapper*>(modelActor->GetMapper())
                ->SetInput(polydata);
#else
        reinterpret_cast<vtkPolyDataMapper*>(modelActor->GetMapper())
                ->SetInputData(polydata);
#endif
    }

#ifdef _DEBUG
    CVLog::Print(QString("updateEntityColor: finish cost %1 s")
                         .arg(CVTools::TimeOff()));
#endif  // _DEBUG

    return (true);
}

void PCLDisplayTools::draw(const CC_DRAW_CONTEXT& context,
                           const ccHObject* obj) {
    if (obj->isA(CV_TYPES::POINT_CLOUD)) {
        // the cloud to draw
        ccPointCloud* ecvCloud =
                ccHObjectCaster::ToPointCloud(const_cast<ccHObject*>(obj));
        if (!ecvCloud) return;

        drawPointCloud(context, ecvCloud);
    } else if (obj->isKindOf(CV_TYPES::MESH) ||
               obj->isKindOf(CV_TYPES::SUB_MESH)) {
        // the mesh to draw
        ccGenericMesh* tempMesh =
                ccHObjectCaster::ToGenericMesh(const_cast<ccHObject*>(obj));
        if (!tempMesh) return;
        drawMesh(const_cast<CC_DRAW_CONTEXT&>(context), tempMesh);
    } else if (obj->isA(CV_TYPES::POLY_LINE)) {
        // the polyline to draw
        ccPolyline* tempPolyline =
                ccHObjectCaster::ToPolyline(const_cast<ccHObject*>(obj));
        if (!tempPolyline) return;
        drawPolygon(context, tempPolyline);
    } else if (obj->isA(CV_TYPES::LINESET)) {
        // the polyline to draw
        cloudViewer::geometry::LineSet* lineset =
                ccHObjectCaster::ToLineSet(const_cast<ccHObject*>(obj));
        if (!lineset) return;
        drawLines(context, lineset);
    } else if (obj->isA(CV_TYPES::IMAGE)) {
        // the image to draw
        ccImage* image = ccHObjectCaster::ToImage(const_cast<ccHObject*>(obj));
        if (!image) return;
        drawImage(context, image);
    } else if (obj->isKindOf(CV_TYPES::SENSOR))  //  must use isKindOf here
    {
        // the sensor to draw
        ccSensor* sensor =
                ccHObjectCaster::ToSensor(const_cast<ccHObject*>(obj));
        if (!sensor) return;
        drawSensor(context, sensor);
    } else {
        return;
    }

    if (m_visualizer3D) {
        m_visualizer3D->resetCameraClippingRange(context.defaultViewPort);
    }
}

bool PCLDisplayTools::checkEntityNeedUpdate(std::string& viewID,
                                            const ccHObject* obj) {
    bool firstShow = !m_visualizer3D->contains(viewID);
    if (firstShow) return true;

    ccPointCloud* cloud =
            ccHObjectCaster::ToPointCloud(const_cast<ccHObject*>(obj));
    if (!cloud) {
        return true;
    }

    vtkActor* modelActor = m_visualizer3D->getActorById(viewID);
    if (!modelActor) {
        return true;
    }

    // Get the current poly data
    vtkSmartPointer<vtkPolyData> polydata =
            reinterpret_cast<vtkPolyDataMapper*>(modelActor->GetMapper())
                    ->GetInput();
    if (!polydata) {
        return true;
    }

    cc2smReader converter(cloud);
    unsigned old_points_num =
            static_cast<unsigned>(polydata->GetNumberOfPoints());
    unsigned new_points_num = converter.getvisibilityNum();
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

void PCLDisplayTools::drawBBox(const CC_DRAW_CONTEXT& context,
                               const ccBBox* bbox) {
    ecvColor::Rgbf colf = ecvTools::TransFormRGB(context.bbDefaultCol);
    int viewport = context.defaultViewPort;
    if (m_visualizer3D) {
        std::string bboxID = CVTools::FromQString(context.viewID);
        if (!m_visualizer3D->contains(bboxID)) {
            m_visualizer3D->addCube(bbox->minCorner().x, bbox->maxCorner().x,
                                    bbox->minCorner().y, bbox->maxCorner().y,
                                    bbox->minCorner().z, bbox->maxCorner().z,
                                    colf.r, colf.g, colf.b, bboxID, viewport);

            // m_visualizer3D->setMeshRenderingMode(context.meshRenderingMode,
            // bboxID, viewport);
            m_visualizer3D->setShapeRenderingProperties(
                    pcl::visualization::PCL_VISUALIZER_REPRESENTATION,
                    pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME,
                    bboxID, viewport);
            m_visualizer3D->setLineWidth(context.defaultLineWidth, bboxID,
                                         viewport);
            m_visualizer3D->setLightMode(bboxID, viewport);
        }
    }
}

void PCLDisplayTools::drawOrientedBBox(const CC_DRAW_CONTEXT& context,
                                       const ecvOrientedBBox* obb) {
    int viewport = context.defaultViewPort;
    if (m_visualizer3D) {
        std::string bboxID = CVTools::FromQString(context.viewID);
        if (m_visualizer3D->contains(bboxID)) {
            m_visualizer3D->removeShape(bboxID);
        }

        m_visualizer3D->addOrientedCube(*obb, bboxID, viewport);
        m_visualizer3D->setLineWidth(context.defaultLineWidth, bboxID,
                                     viewport);
        m_visualizer3D->setLightMode(bboxID, viewport);
    }
}

bool PCLDisplayTools::orientationMarkerShown() {
    return m_visualizer3D->pclMarkerAxesShown();
}

void PCLDisplayTools::toggleOrientationMarker(bool state) {
    if (state) {
        m_visualizer3D->showPclMarkerAxes(
                m_visualizer3D->getRenderWindowInteractor());
    } else {
        m_visualizer3D->hidePclMarkerAxes();
    }
}

void PCLDisplayTools::removeEntities(const CC_DRAW_CONTEXT& context) {
    if (context.removeEntityType == ENTITY_TYPE::ECV_IMAGE ||
        context.removeEntityType == ENTITY_TYPE::ECV_LINES_2D ||
        context.removeEntityType == ENTITY_TYPE::ECV_CIRCLE_2D ||
        context.removeEntityType == ENTITY_TYPE::ECV_RECTANGLE_2D ||
        context.removeEntityType == ENTITY_TYPE::ECV_TRIANGLE_2D ||
        context.removeEntityType == ENTITY_TYPE::ECV_MARK_POINT) {
        if (!m_visualizer2D) return;
        std::string viewId = CVTools::FromQString(context.removeViewID);
        if (m_visualizer2D->contains(viewId)) {
            m_visualizer2D->removeLayer(viewId);
        }
    } else {
        if (context.removeEntityType == ENTITY_TYPE::ECV_TEXT2D ||
            context.removeEntityType == ENTITY_TYPE::ECV_POLYLINE_2D) {
            if (m_visualizer2D) {
                std::string viewId = CVTools::FromQString(context.removeViewID);
                if (m_visualizer2D->contains(viewId)) {
                    m_visualizer2D->removeLayer(viewId);
                }
            }
        }

        if (m_visualizer3D && m_visualizer3D->removeEntities(context)) {
            m_visualizer3D->resetCameraClippingRange(context.defaultViewPort);
        }
    }
}

bool PCLDisplayTools::hideShowEntities(const CC_DRAW_CONTEXT& context) {
    std::string viewId = CVTools::FromQString(context.viewID);

    if (context.hideShowEntityType == ENTITY_TYPE::ECV_IMAGE ||
        context.removeEntityType == ENTITY_TYPE::ECV_LINES_2D ||
        context.removeEntityType == ENTITY_TYPE::ECV_CIRCLE_2D ||
        context.removeEntityType == ENTITY_TYPE::ECV_TRIANGLE_2D ||
        context.removeEntityType == ENTITY_TYPE::ECV_RECTANGLE_2D ||
        context.removeEntityType == ENTITY_TYPE::ECV_MARK_POINT) {
        if (!m_visualizer2D || !m_visualizer2D->contains(viewId)) return false;

        m_visualizer2D->hideShowActors(context.visible, viewId);
        return true;
    } else if (context.removeEntityType == ENTITY_TYPE::ECV_CAPTION) {
        if (m_visualizer3D->containWidget(viewId)) {
            m_visualizer3D->hideShowWidgets(context.visible, viewId,
                                            context.defaultViewPort);
        }
    } else {
        if (context.removeEntityType == ENTITY_TYPE::ECV_TEXT2D ||
            context.removeEntityType == ENTITY_TYPE::ECV_POLYLINE_2D) {
            if (m_visualizer2D && m_visualizer2D->contains(viewId)) {
                m_visualizer2D->hideShowActors(context.visible, viewId);
            }
        }

        if (!m_visualizer3D->contains(viewId)) {
            return false;
        }

        m_visualizer3D->hideShowActors(context.visible, viewId,
                                       context.defaultViewPort);

        // for normals case
        std::string normalViewId =
                CVTools::FromQString(context.viewID + "-normal");
        if (m_visualizer3D->contains(normalViewId)) {
            m_visualizer3D->hideShowActors(context.visible, normalViewId,
                                           context.defaultViewPort);
        }

        m_visualizer3D->resetCameraClippingRange(context.defaultViewPort);
    }

    return true;
}

void PCLDisplayTools::drawWidgets(const WIDGETS_PARAMETER& param) {
    //	ccHObject * entity = param.entity;
    int viewport = param.viewport;
    std::string viewID = CVTools::FromQString(param.viewID);
    switch (param.type) {
        case WIDGETS_TYPE::WIDGET_COORDINATE:
            break;
        case WIDGETS_TYPE::WIDGET_BBOX:
            break;
        case WIDGETS_TYPE::WIDGET_T2D:
            if (m_visualizer2D) {
                std::string text = CVTools::FromQString(param.text);
                m_visualizer2D->addText(param.rect.x(), param.rect.y(), text,
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
                context.textDefaultCol = ecvColor::FromRgbafToRgb(param.color);
                context.textParam = tParam;
                context.viewID = tParam.text;
                m_visualizer3D->displayText(context);
            }

        case WIDGETS_TYPE::WIDGET_LINE_3D:
            if (param.lineWidget.valid && !m_visualizer3D->contains(viewID)) {
                PointT lineSt;
                lineSt.x = param.lineWidget.lineSt.x;
                lineSt.y = param.lineWidget.lineSt.y;
                lineSt.z = param.lineWidget.lineSt.z;
                PointT lineEd;
                lineEd.x = param.lineWidget.lineEd.x;
                lineEd.y = param.lineWidget.lineEd.y;
                lineEd.z = param.lineWidget.lineEd.z;
                unsigned char lineWidth =
                        (unsigned char)param.lineWidget.lineWidth;
                ecvColor::Rgbf lineColor =
                        ecvTools::TransFormRGB(param.lineWidget.lineColor);
                m_visualizer3D->addLine(lineSt, lineEd, lineColor.r,
                                        lineColor.g, lineColor.b, viewID,
                                        viewport);
                m_visualizer3D->setLineWidth(lineWidth, viewID, viewport);
            }
            break;
        case WIDGETS_TYPE::WIDGET_SPHERE:
            if (!m_visualizer3D->contains(viewID)) {
                PointT center;
                center.x = param.center.x;
                center.y = param.center.y;
                center.z = param.center.z;
                m_visualizer3D->addSphere(center, param.radius, param.color.r,
                                          param.color.g, param.color.b, viewID,
                                          viewport);
            }
            break;

        case WIDGETS_TYPE::WIDGET_SCALAR_BAR:
            if (!m_visualizer3D->updateScalarBar(param.context)) {
                m_visualizer3D->addScalarBar(param.context);
            }
            break;
        case WIDGETS_TYPE::WIDGET_CAPTION:
            if (!m_visualizer3D->updateCaption(
                        CVTools::FromQString(param.text), param.pos,
                        param.center, param.color.r, param.color.g,
                        param.color.b, param.color.a, param.fontSize, viewID,
                        viewport)) {
                m_visualizer3D->addCaption(
                        CVTools::FromQString(param.text), param.pos,
                        param.center, param.color.r, param.color.g,
                        param.color.b, param.color.a, param.fontSize, viewID,
                        param.handleEnabled, viewport);
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
                                "[PCLDisplayTools::drawWidgets] draw mode is "
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
        case WIDGETS_TYPE::WIDGET_POINTS_2D:
            if (m_visualizer2D) {
                pcl::visualization::Vector3ub color =
                        pcl::visualization::Vector3ub(
                                param.color.r, param.color.g, param.color.b);
                m_visualizer2D->markPoint(param.rect.x(), param.rect.y(), color,
                                          color, param.radius, viewID,
                                          param.color.a);
            }
            break;
        case WIDGETS_TYPE::WIDGET_RECTANGLE_2D:
            if (m_visualizer2D) {
                int minX = std::max(param.rect.x(), 0);
                int maxX = std::min(minX + param.rect.width(),
                                    m_visualizer2D->getSize()[0]);
                int minY = std::max(param.rect.y(), 0);
                int maxY = std::min(minY + param.rect.height(),
                                    m_visualizer2D->getSize()[1]);

                if (param.filled) {
                    m_visualizer2D->addFilledRectangle(
                            minX, maxX, minY, maxY, param.color.r,
                            param.color.g, param.color.b, viewID,
                            param.color.a);
                } else {
                    m_visualizer2D->addRectangle(minX, maxX, minY, maxY,
                                                 param.color.r, param.color.g,
                                                 param.color.b, viewID,
                                                 param.color.a);
                }
            }
            break;
        case WIDGETS_TYPE::WIDGET_CIRCLE_2D:
            if (m_visualizer2D) {
                m_visualizer2D->addCircle(param.rect.x(), param.rect.y(),
                                          param.radius, param.color.r,
                                          param.color.g, param.color.b, viewID,
                                          param.color.a);
            }
            break;
        case WIDGETS_TYPE::WIDGET_IMAGE:
            if (m_visualizer2D) {
                if (param.image.isNull()) return;

                m_visualizer2D->addRGBImage(param.image.bits(), param.rect.x(),
                                            param.rect.y(), param.image.width(),
                                            param.image.height(), viewID,
                                            param.opacity);
            }
            break;
        default:
            break;
    }
}

void PCLDisplayTools::displayText(const CC_DRAW_CONTEXT& context) {
    if (m_visualizer2D) {
        ecvTextParam textParam = context.textParam;
        std::string viewID = CVTools::FromQString(context.viewID);
        std::string text = CVTools::FromQString(textParam.text);

        ecvColor::Rgbf textColor =
                ecvTools::TransFormRGB(context.textDefaultCol);
        {
            m_visualizer2D->addText(
                    textParam.textPos.x, textParam.textPos.y, text, textColor.r,
                    textColor.g, textColor.b, viewID, textParam.opacity,
                    textParam.font.pointSize(), textParam.font.bold());
        }
    } else {
        m_visualizer3D->displayText(context);
    }
}

void PCLDisplayTools::toggle2Dviewer(bool state) {
    if (m_visualizer2D) {
        m_visualizer2D->enable2Dviewer(state);
    }
}

QString PCLDisplayTools::pick2DLabel(int x, int y) {
    if (m_visualizer2D) {
        return m_visualizer2D->pickItem(x, y).c_str();
    }

    return QString();
}

QString PCLDisplayTools::pick3DItem(int x, int y) {
    if (m_visualizer3D) {
        return m_visualizer3D->pickItem(x, y).c_str();
    }

    return QString();
}

QString PCLDisplayTools::pickObject(double x, double y) {
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

QImage PCLDisplayTools::renderToImage(int zoomFactor,
                                      bool renderOverlayItems,
                                      bool silent,
                                      int viewport) {
    if (m_visualizer3D) {
        return m_visualizer3D->renderToImage(zoomFactor, renderOverlayItems,
                                             silent, viewport);
    } else {
        if (!silent)
            CVLog::Error(
                    "[PCLDisplayTools::renderToImage] PCLVis Initialization "
                    "failed! (not enough memory?)");
        return QImage();
    }
}

double PCLDisplayTools::getParallelScale(int viewport) {
    if (m_visualizer3D) {
        return m_visualizer3D->getParallelScale(viewport);
    }

    return -1;
}

void PCLDisplayTools::setParallelScale(double scale, int viewport) {
    if (m_visualizer3D) {
        m_visualizer3D->setParallelScale(scale, viewport);
    }
}

void PCLDisplayTools::getProjectionMatrix(double* projArray, int viewport) {
    Eigen::Matrix4d projMat;
    m_visualizer3D->getCamera(viewport).computeProjectionMatrix(projMat);
    // m_visualizer3D->getProjectionTransformMatrix(projMat);
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

void PCLDisplayTools::getViewMatrix(double* ViewArray, int viewport) {
    Eigen::Matrix4d viewMat;
    m_visualizer3D->getCamera(viewport).computeViewMatrix(viewMat);
    // m_visualizer3D->getModelViewTransformMatrix(viewMat);
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

void PCLDisplayTools::setViewMatrix(const ccGLMatrixd& viewMat, int viewport) {
    if (m_visualizer3D) {
        m_visualizer3D->setModelViewMatrix(viewMat, viewport);
    }
}

void PCLDisplayTools::changeEntityProperties(PROPERTY_PARAM& param) {
    std::string viewId = CVTools::FromQString(param.viewId);
    int viewport = param.viewport;
    switch (param.property) {
        case PROPERTY_MODE::ECV_POINTSSIZE_PROPERTY: {
            m_visualizer3D->setPointSize(param.pointSize, viewId, viewport);
        } break;
        case PROPERTY_MODE::ECV_LINEWITH_PROPERTY: {
            m_visualizer3D->setLineWidth(
                    static_cast<unsigned char>(param.lineWidth), viewId,
                    viewport);
        } break;
        case PROPERTY_MODE::ECV_COLOR_PROPERTY: {
            ecvColor::Rgbf colf = ecvTools::TransFormRGB(param.color);
            switch (param.entityType) {
                case ENTITY_TYPE::ECV_POINT_CLOUD:
                case ENTITY_TYPE::ECV_MESH: {
                    m_visualizer3D->setPointCloudUniqueColor(
                            colf.r, colf.g, colf.b, viewId, viewport);
                } break;
                case ENTITY_TYPE::ECV_SHAPE:
                case ENTITY_TYPE::ECV_LINES_3D: {
                    m_visualizer3D->setShapeUniqueColor(colf.r, colf.g, colf.b,
                                                        viewId, viewport);
                } break;
                default:
                    break;
            }
        } break;
        case PROPERTY_MODE::ECV_OPACITY_PROPERTY: {
            switch (param.entityType) {
                case ENTITY_TYPE::ECV_POINT_CLOUD:
                case ENTITY_TYPE::ECV_MESH: {
                    m_visualizer3D->setPointCloudOpacity(param.opacity, viewId,
                                                         viewport);
                } break;
                case ENTITY_TYPE::ECV_SHAPE:
                case ENTITY_TYPE::ECV_LINES_3D: {
                    m_visualizer3D->setShapeOpacity(param.opacity, viewId,
                                                    viewport);
                } break;
                default:
                    break;
            }
        } break;
        case PROPERTY_MODE::ECV_SHADING_PROPERTY: {
            switch (param.entityType) {
                case ENTITY_TYPE::ECV_POINT_CLOUD:
                case ENTITY_TYPE::ECV_MESH: {
                    m_visualizer3D->setMeshShadingMode(param.shadingMode,
                                                       viewId, viewport);
                }
                case ENTITY_TYPE::ECV_SHAPE:
                case ENTITY_TYPE::ECV_LINES_3D: {
                    m_visualizer3D->setShapeShadingMode(param.shadingMode,
                                                        viewId, viewport);
                } break;
                default:
                    break;
            }
        } break;

        default:
            break;
    }
}

void PCLDisplayTools::transformCameraView(const ccGLMatrixd& viewMat) {
    getQVtkWidget()->transformCameraView(viewMat.data());
}

void PCLDisplayTools::transformCameraProjection(const ccGLMatrixd& projMat) {
    getQVtkWidget()->transformCameraProjection(projMat.data());
}
