// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/**
 * @file VtkTransformTool.h
 * @brief VTK-based transform tool for 3D object translation, rotation, scale,
 * and shear (ecvGenericTransformTool implementation).
 */

// LOCAL
#include "qVTK.h"

// CV_DB_LIB
#include <ecvGenericTransformTool.h>

// VTK
#include <vtkSmartPointer.h>

namespace Visualization {
class VtkVis;
}

class vtkActor;
class vtkTransform;
class CustomVtkBoxWidget;
class ecvGenericVisualizer3D;

class QVTK_ENGINE_LIB_API VtkTransformTool : public ecvGenericTransformTool {
    Q_OBJECT
public:
    explicit VtkTransformTool(ecvGenericVisualizer3D* viewer);
    ~VtkTransformTool();

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
    Visualization::VtkVis* m_viewer;
};
