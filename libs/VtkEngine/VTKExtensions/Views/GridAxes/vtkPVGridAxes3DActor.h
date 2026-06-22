// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef vtkPVGridAxes3DActor_h
#define vtkPVGridAxes3DActor_h

#include "qVTK.h"
#include "vtkGridAxesActor3D.h"

class vtkMatrix4x4;

class QVTK_ENGINE_LIB_API vtkPVGridAxes3DActor : public vtkGridAxesActor3D {
public:
    static vtkPVGridAxes3DActor* New();
    vtkTypeMacro(vtkPVGridAxes3DActor, vtkGridAxesActor3D);
    void PrintSelf(ostream& os, vtkIndent indent) override;

    /**
     * Shallow copy from another vtkPVGridAxes3DActor.
     */
    void ShallowCopy(vtkProp* prop) override;

    ///@{
    /**
     * Specify transformation used by the data.
     */
    vtkSetVector3Macro(DataScale, double);
    vtkGetVector3Macro(DataScale, double);
    ///@}

    ///@{
    /**
     * Specify the scale used in the display transformation.
     */
    void SetDisplayTransformScale(double scaleX, double scaleY, double scaleZ);
    void SetDisplayTransformScale(double scale[3]) {
        this->SetDisplayTransformScale(scale[0], scale[1], scale[2]);
    }
    vtkGetVector3Macro(DisplayTransformScale, double);
    ///@}

    /**
     * If enabled, the scale will be applied to the data but the tick values
     * will still show the non-scaled model dimensions. By default, this option
     * is false.
     */
    void InvertDisplayTransformScaleForTickLabels(bool enabled);

    vtkSetVector3Macro(DataPosition, double);
    vtkGetVector3Macro(DataPosition, double);

    ///@{
    /**
     * Specify the scale factor used to proportionally
     * scale each axis. 1 means no change.
     */
    vtkSetMacro(DataBoundsScaleFactor, double);
    vtkGetMacro(DataBoundsScaleFactor, double);
    ///@}

    ///@{
    /**
     * Another way for specifying grid bounds except here the bounds are
     * considered to be pre-transformed. Using the DataPosition, DataScale, and
     * DataBoundsScaleFactor, the provided bounds will un-transformed before
     * calling this->SetGridBounds() and then position and scale are instead
     * passed on the vtkActor. This results in display scaling rather than data
     * scaling.
     */
    vtkSetVector6Macro(TransformedBounds, double);
    vtkGetVector6Macro(TransformedBounds, double);
    ///@}

    ///@{
    /**
     * If set to true, CustomTransformedBounds are used instead of
     * TransformedBounds. Default is false.
     */
    vtkSetMacro(UseCustomTransformedBounds, bool);
    vtkGetMacro(UseCustomTransformedBounds, bool);
    ///@}

    ///@{
    /**
     * Same as TransformedBounds, except used only when
     * UseCustomTransformedBounds is set.
     */
    vtkSetVector6Macro(CustomTransformedBounds, double);
    vtkGetVector6Macro(CustomTransformedBounds, double);
    ///@}

    vtkSetMacro(UseModelTransform, bool);
    vtkGetMacro(UseModelTransform, bool);
    vtkBooleanMacro(UseModelTransform, bool);
    vtkSetVector6Macro(ModelBounds, double);
    vtkGetVector6Macro(ModelBounds, double);
    void SetModelTransformMatrix(double* matrix);

    /**
     * Overridden to ensure that the transform information is passed on the
     * superclass.
     */
    double* GetBounds() override;

protected:
    vtkPVGridAxes3DActor();
    ~vtkPVGridAxes3DActor() override;

    void Update(vtkViewport* viewport) override;
    void UpdateGridBounds();
    void UpdateGridBoundsUsingDataBounds();
    void UpdateGridBoundsUsingModelTransform();

    double DataScale[3];
    double DataPosition[3];
    double DataBoundsScaleFactor;
    double TransformedBounds[6];

    double DisplayTransformScale[3];

    bool UseModelTransform;
    double ModelBounds[6];
    vtkNew<vtkMatrix4x4> ModelTransformMatrix;

    bool UseCustomTransformedBounds;
    double CustomTransformedBounds[6];

private:
    vtkPVGridAxes3DActor(const vtkPVGridAxes3DActor&) = delete;
    void operator=(const vtkPVGridAxes3DActor&) = delete;

    vtkTimeStamp BoundsUpdateTime;
};

#endif
