// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "PclTransformTool.h"

// Local
#include "PclUtils/PCLVis.h"
#include "PclUtils/vtk2cc.h"
#include "VTKExtensions/CallbackTools/vtkCallbackTools.h"
#include "VTKExtensions/Widgets/CustomVtkBoxWidget.h"
#include "VtkUtils/vtkutils.h"

// CV_CORE_LIB
#include <CVGeom.h>
#include <CVLog.h>
#include <CVTools.h>

// CV_DB_LIB
#include <ecvDisplayTools.h>
#include <ecvMesh.h>
#include <ecvPointCloud.h>
#include <ecvPolyline.h>

// VTK
#include <vtkActor.h>
#include <vtkAssembly.h>
#include <vtkBoxRepresentation.h>
#include <vtkMatrix4x4.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkTransform.h>
#include <vtkTransformPolyDataFilter.h>
#include <vtkWidgetRepresentation.h>

using namespace PclUtils;

PclTransformTool::PclTransformTool(ecvGenericVisualizer3D* viewer)
    : ecvGenericTransformTool(),
      m_boxWidgetTransformer(nullptr),
      m_viewer(nullptr) {
    setVisualizer(viewer);
    m_modelActors.clear();
}

PclTransformTool::~PclTransformTool() {}

void PclTransformTool::setVisualizer(ecvGenericVisualizer3D* viewer) {
    if (viewer) {
        m_viewer = reinterpret_cast<PCLVis*>(viewer);
        if (!m_viewer) {
            CVLog::Warning(
                    "[PclTransformTool::setVisualizer] viewer is Null!");
        }
    } else {
        CVLog::Warning("[PclTransformTool::setVisualizer] viewer is Null!");
    }
}

bool PclTransformTool::setInputData(ccHObject* entity, int viewport) {
    if (!m_viewer || !ecvGenericTransformTool::setInputData(entity)) {
        return false;
    }

    m_modelActors.clear();

    if (!getAssociatedEntity()) {
        return false;
    }

    addActors();

    return true;
}

void PclTransformTool::addActors() {
    assert(m_modelActors.empty());
    if (!getAssociatedEntity()) {
        return;
    }

    unsigned n = getAssociatedEntity()->getChildrenNumber();
    for (unsigned i = 0; i < n; ++i) {
        ccHObject* ent = getAssociatedEntity()->getChild(i);
        std::string id = CVTools::FromQString(ent->getViewId());
        if (id == "") {
            continue;
        }

        vtkActor* actor = m_viewer->getActorById(id);
        if (!actor) {
            continue;
        }
        m_modelActors.push_back(actor);
    }
}

void PclTransformTool::showInteractor(bool state) {
    if (!m_boxWidgetTransformer) {
        CVLog::Warning(
                QString("[PclTransformTool::showInteractor] boxWidget has not "
                        "been initialized!"));
        return;
    }

    if (state) {
        m_boxWidgetTransformer->On();
    } else {
        m_boxWidgetTransformer->Off();
    }
}

bool PclTransformTool::start() {
    if (m_modelActors.empty()) {
        return false;
    }

    VtkUtils::vtkInitOnce(m_boxWidgetTransformer);
    m_boxWidgetTransformer->SetInteractor(
            m_viewer->getRenderWindowInteractor());
    m_boxWidgetTransformer->HandlesOff();         // default off
    m_boxWidgetTransformer->SetPlaceFactor(1.1);  // default 0.5

    VTK_CREATE(vtkAssembly, assembly);
    for (vtkActor* actor : m_modelActors) {
        assembly->AddPart(actor);
    }
    m_boxWidgetTransformer->PlaceWidget(assembly->GetBounds());

    VtkUtils::vtkInitOnce(m_originTrans);
    m_boxWidgetTransformer->GetTransform(m_originTrans);

    vtkSmartPointer<CallbackTools::vtkBoxCallback> boxCallback =
            vtkSmartPointer<CallbackTools::vtkBoxCallback>::New();
    boxCallback->SetActors(m_modelActors);
    boxCallback->attach(m_boxWidgetTransformer);
    // connect(boxCallback, &CallbackTools::vtkBoxCallback::uerTransform, this,
    // &PclTransformTool::onTransform);
    m_boxWidgetTransformer->On();

    return true;
}

void PclTransformTool::stop() {
    if (m_boxWidgetTransformer) {
        reset();
        // m_boxWidgetTransformer->RemoveObserver(vtkCommand::EndInteractionEvent);
        m_boxWidgetTransformer->Off();
    }

    if (!m_modelActors.empty()) {
        m_modelActors.clear();
    }
}

void PclTransformTool::reset() {
    if (m_originTrans && !m_modelActors.empty() && m_boxWidgetTransformer) {
        m_boxWidgetTransformer->SetTransform(m_originTrans);
        for (vtkActor* actor : this->m_modelActors) {
            if (actor) {
                actor->SetUserTransform(m_originTrans);
                actor->Modified();
            }
        }
        ecvDisplayTools::UpdateScreen();
    }
}

void PclTransformTool::clear() {}

void PclTransformTool::setTranlationMode(TranslationMOde mode) {
    if (!m_boxWidgetTransformer) {
        CVLog::Warning(
                QString("[PclTransformTool::setTranlationMode] boxWidget has "
                        "not been initialized!"));
        return;
    }

    switch (mode) {
        case ecvGenericTransformTool::T_X: {
            m_boxWidgetTransformer->SetTranslateXEnabled(true);
            m_boxWidgetTransformer->SetTranslateYEnabled(false);
            m_boxWidgetTransformer->SetTranslateZEnabled(false);
        } break;
        case ecvGenericTransformTool::T_Y: {
            m_boxWidgetTransformer->SetTranslateXEnabled(false);
            m_boxWidgetTransformer->SetTranslateYEnabled(true);
            m_boxWidgetTransformer->SetTranslateZEnabled(false);
        } break;
        case ecvGenericTransformTool::T_Z: {
            m_boxWidgetTransformer->SetTranslateXEnabled(false);
            m_boxWidgetTransformer->SetTranslateYEnabled(false);
            m_boxWidgetTransformer->SetTranslateZEnabled(true);
        } break;
        case ecvGenericTransformTool::T_XY: {
            m_boxWidgetTransformer->SetTranslateXEnabled(true);
            m_boxWidgetTransformer->SetTranslateYEnabled(true);
            m_boxWidgetTransformer->SetTranslateZEnabled(false);
        } break;
        case ecvGenericTransformTool::T_XZ: {
            m_boxWidgetTransformer->SetTranslateXEnabled(true);
            m_boxWidgetTransformer->SetTranslateYEnabled(false);
            m_boxWidgetTransformer->SetTranslateZEnabled(true);
        } break;
        case ecvGenericTransformTool::T_ZY: {
            m_boxWidgetTransformer->SetTranslateXEnabled(false);
            m_boxWidgetTransformer->SetTranslateYEnabled(true);
            m_boxWidgetTransformer->SetTranslateZEnabled(true);
        } break;
        case ecvGenericTransformTool::T_XYZ: {
            m_boxWidgetTransformer->SetTranslateXEnabled(true);
            m_boxWidgetTransformer->SetTranslateYEnabled(true);
            m_boxWidgetTransformer->SetTranslateZEnabled(true);
        } break;
        case ecvGenericTransformTool::T_NONE: {
            m_boxWidgetTransformer->SetTranslateXEnabled(false);
            m_boxWidgetTransformer->SetTranslateYEnabled(false);
            m_boxWidgetTransformer->SetTranslateZEnabled(false);
        } break;
        default:
            break;
    }
}

void PclTransformTool::setRotationMode(RotationMode mode) {
    if (!m_boxWidgetTransformer) {
        CVLog::Warning(
                QString("[PclTransformTool::setRotationMode] boxWidget has "
                        "not been initialized!"));
        return;
    }

    switch (mode) {
        case ecvGenericTransformTool::R_XYZ: {
            m_boxWidgetTransformer->SetRotateXEnabled(true);
            m_boxWidgetTransformer->SetRotateYEnabled(true);
            m_boxWidgetTransformer->SetRotateZEnabled(true);
        } break;
        case ecvGenericTransformTool::R_X: {
            m_boxWidgetTransformer->SetRotateXEnabled(true);
            m_boxWidgetTransformer->SetRotateYEnabled(false);
            m_boxWidgetTransformer->SetRotateZEnabled(false);
        } break;
        case ecvGenericTransformTool::R_Y: {
            m_boxWidgetTransformer->SetRotateXEnabled(false);
            m_boxWidgetTransformer->SetRotateYEnabled(true);
            m_boxWidgetTransformer->SetRotateZEnabled(false);
        } break;
        case ecvGenericTransformTool::R_Z: {
            m_boxWidgetTransformer->SetRotateXEnabled(false);
            m_boxWidgetTransformer->SetRotateYEnabled(false);
            m_boxWidgetTransformer->SetRotateZEnabled(true);
        } break;
        default:
            break;
    }
}

void PclTransformTool::setScaleEnabled(bool state) {
    if (!m_boxWidgetTransformer) {
        CVLog::Warning(
                QString("[PclTransformTool::setScaleEnabled] boxWidget has "
                        "not been initialized!"));
        return;
    }

    m_boxWidgetTransformer->SetScaleEnabled(state);
}

void PclTransformTool::setShearEnabled(bool state) {
    if (!m_boxWidgetTransformer) {
        CVLog::Warning(
                QString("[PclTransformTool::setShearEnabled] boxWidget has "
                        "not been initialized!"));
        return;
    }

    state ? m_boxWidgetTransformer->HandlesOn()
          : m_boxWidgetTransformer->HandlesOff();
}

const ccGLMatrixd PclTransformTool::getFinalTransformation() {
    if (!m_boxWidgetTransformer) {
        return ccGLMatrixd();
    }

    vtkSmartPointer<vtkTransform> trans = vtkSmartPointer<vtkTransform>::New();
    m_boxWidgetTransformer->GetTransform(trans);

    vtkSmartPointer<vtkMatrix4x4> matrix44 =
            vtkSmartPointer<vtkMatrix4x4>::New();
    trans->GetTranspose(matrix44);
    return ccGLMatrixd(matrix44->GetData());
}

void PclTransformTool::getOutput(std::vector<ccHObject*>& out) {
    if (!m_boxWidgetTransformer) {
        return;
    }

    vtkSmartPointer<vtkTransform> trans = vtkSmartPointer<vtkTransform>::New();
    m_boxWidgetTransformer->GetTransform(trans);

    assert(m_modelActors.size() == getAssociatedEntity()->getChildrenNumber());
    int index = 0;
    for (vtkActor* actor : m_modelActors) {
        vtkPolyData* polydata =
                reinterpret_cast<vtkPolyDataMapper*>(actor->GetMapper())
                        ->GetInput();
        if (nullptr == polydata) {
            out.push_back(nullptr);
            index++;
            continue;
        }

        vtkSmartPointer<vtkPolyData> dataObject =
                vtkSmartPointer<vtkPolyData>::New();
        dataObject->DeepCopy(polydata);

        vtkSmartPointer<vtkTransformPolyDataFilter> transformFilter =
                vtkSmartPointer<vtkTransformPolyDataFilter>::New();
        transformFilter->SetInputData(dataObject);
        transformFilter->SetTransform(trans);
        transformFilter->Update();
        dataObject = transformFilter->GetOutput();

        ccHObject* result = nullptr;
        ccHObject* baseEntity = getAssociatedEntity()->getChild(index);
        assert(baseEntity);

        if (baseEntity->isKindOf(CV_TYPES::POINT_CLOUD)) {
            result = vtk2cc::ConvertToPointCloud(dataObject);
        } else if (baseEntity->isKindOf(CV_TYPES::MESH)) {
            result = vtk2cc::ConvertToMesh(dataObject);
        } else if (baseEntity->isKindOf(CV_TYPES::POLY_LINE)) {
            result = vtk2cc::ConvertToPolyline(dataObject);
        } else {
            CVLog::Warning(QString(
                    "only cloud, mesh and polyline are supported now!"));
        }

        if (result) {
            out.push_back(result);
        } else {
            out.push_back(nullptr);
        }

        index++;
    }
}

