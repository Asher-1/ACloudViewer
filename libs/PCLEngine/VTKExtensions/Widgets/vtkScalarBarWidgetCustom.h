// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef vtkPVScalarBarRepresentationCustom_h
#define vtkPVScalarBarRepresentationCustom_h

#include "qPCL.h"
#include "vtkScalarBarRepresentation.h"
#include "vtkScalarBarWidget.h"

class QPCL_ENGINE_LIB_API vtkScalarBarWidgetCustom : public vtkScalarBarWidget {
public:
    static vtkScalarBarWidgetCustom* New();
    vtkTypeMacro(vtkScalarBarWidgetCustom, vtkBorderWidget);
    void PrintSelf(ostream& os, vtkIndent indent) override;

    /**
     * Specify an instance of vtkWidgetRepresentation used to represent this
     * widget in the scene. Note that the representation is a subclass of
     * vtkProp so it can be added to the renderer independent of the widget.
     */
    virtual void SetRepresentation(vtkScalarBarRepresentation* rep) override;

    /**
     * Return the representation as a vtkScalarBarRepresentation.
     */
    vtkScalarBarRepresentation* GetScalarBarRepresentation() {
        return reinterpret_cast<vtkScalarBarRepresentation*>(
                this->GetRepresentation());
    }

    //@{
    /**
     * Get the ScalarBar used by this Widget. One is created automatically.
     */
    virtual void SetScalarBarActor(vtkScalarBarActor* actor) override;
    virtual vtkScalarBarActor* GetScalarBarActor() override;

    /**
     * Create the default widget representation if one is not set.
     */
    void CreateDefaultRepresentation() override;

    void CreateDefaultScalarBarActor();

protected:
    vtkScalarBarWidgetCustom();
    ~vtkScalarBarWidgetCustom() override;

private:
    vtkScalarBarWidgetCustom(const vtkScalarBarWidgetCustom&) = delete;
    void operator=(const vtkScalarBarWidgetCustom&) = delete;
};

#endif  // vtkPVScalarBarRepresentationCustom_h
