// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// LOCAL
#include "../../qPCL.h"

// ECV_DB_LIB
#include <ecvGenericTransformTool.h>

// VTK
#include <vtkSmartPointer.h>

namespace PclUtils {
class PCLVis;
}

class vtkActor;
class vtkTransform;
class CustomVtkBoxWidget;
class ecvGenericVisualizer3D;

class QPCL_ENGINE_LIB_API QvtkTransformTool : public ecvGenericTransformTool {
    Q_OBJECT
public:
    explicit QvtkTransformTool(ecvGenericVisualizer3D* viewer);
    ~QvtkTransformTool();

    /**
     * @brief initialize
     */
    virtual void setVisualizer(
            ecvGenericVisualizer3D* viewer = nullptr) override;
    virtual bool setInputData(ccHObject* entity, int viewport = 0) override;

    void addActors();

    virtual void showInteractor(bool state) override;

    virtual bool start() override;
    virtual void stop() override;
    virtual void reset() override;
    virtual void clear() override;

    virtual void setTranlationMode(TranslationMOde mode);
    virtual void setRotationMode(RotationMode mode);
    virtual void setScaleEnabled(bool state);
    virtual void setShearEnabled(bool state);

    virtual const ccGLMatrixd getFinalTransformation() override;
    virtual void getOutput(std::vector<ccHObject*>& out) override;

private:
    std::vector<vtkActor*> m_modelActors;
    vtkSmartPointer<CustomVtkBoxWidget> m_boxWidgetTransformer;
    vtkSmartPointer<vtkTransform> m_originTrans;
    PclUtils::PCLVis* m_viewer;
};
