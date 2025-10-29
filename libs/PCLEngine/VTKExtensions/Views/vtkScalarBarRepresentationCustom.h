// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "qPCL.h"
#include "vtkScalarBarRepresentation.h"

class QPCL_ENGINE_LIB_API vtkScalarBarRepresentationCustom
    : public vtkScalarBarRepresentation {
public:
    vtkTypeMacro(vtkScalarBarRepresentationCustom,
                 vtkScalarBarRepresentation) void PrintSelf(ostream& os,
                                                            vtkIndent indent)
            override;
    static vtkScalarBarRepresentationCustom* New();

    enum {
        AnyLocation = 0,
        LowerLeftCorner,
        LowerRightCorner,
        LowerCenter,
        UpperLeftCorner,
        UpperRightCorner,
        UpperCenter
    };

    //@{
    /**
     * Set the scalar bar position, by enumeration (
     * AnyLocation = 0,
     * LowerLeftCorner,
     * LowerRightCorner,
     * LowerCenter,
     * UpperLeftCorner,
     * UpperRightCorner,
     * UpperCenter)
     * related to the render window.
     */
    vtkSetMacro(WindowLocation, int);
    vtkGetMacro(WindowLocation, int);
    //@}

    /**
     * Override to obtain viewport size and potentially adjust placement
     * of the representation.
     */
    int RenderOverlay(vtkViewport*) override;

protected:
    vtkScalarBarRepresentationCustom();
    ~vtkScalarBarRepresentationCustom() override;

    int WindowLocation;

private:
    vtkScalarBarRepresentationCustom(const vtkScalarBarRepresentationCustom&) =
            delete;
    void operator=(const vtkScalarBarRepresentationCustom&) = delete;
};
