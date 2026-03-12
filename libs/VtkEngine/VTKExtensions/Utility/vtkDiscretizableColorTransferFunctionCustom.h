// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/**
 * @file vtkDiscretizableColorTransferFunctionCustom.h
 * @brief Extended color transfer function with full annotation set and active
 * subset.
 */

#include "qVTK.h"
#include "vtkDiscretizableColorTransferFunction.h"

class vtkAbstractArray;
class vtkDoubleArray;
class vtkVariantArray;

/**
 * @class vtkDiscretizableColorTransferFunctionCustom
 * @brief Color transfer function with full annotation set and active-value
 * filtering.
 */
class QVTK_ENGINE_LIB_API vtkDiscretizableColorTransferFunctionCustom
    : public vtkDiscretizableColorTransferFunction {
public:
    static vtkDiscretizableColorTransferFunctionCustom* New();
    vtkTypeMacro(vtkDiscretizableColorTransferFunctionCustom,
                 vtkDiscretizableColorTransferFunction);
    void PrintSelf(ostream& os, vtkIndent indent) override;

    //@{
    /**
     * Parallel API to API for annotated values to set/get the full list of
     * annotations. A subset of the full list will be used.
     */
    /// @param values Array of annotation values
    /// @param annotations Array of annotation strings
    virtual void SetAnnotationsInFullSet(vtkAbstractArray* values,
                                         vtkStringArray* annotations);
    vtkGetObjectMacro(AnnotatedValuesInFullSet, vtkAbstractArray);
    vtkGetObjectMacro(AnnotationsInFullSet, vtkStringArray);
    virtual vtkIdType SetAnnotationInFullSet(vtkVariant value,
                                             std::string annotation);
    virtual vtkIdType SetAnnotationInFullSet(std::string value,
                                             std::string annotation);
    virtual void ResetAnnotationsInFullSet();
    //@}

    /**
     * Add colors to use when \a IndexedLookup is true.
     * \a SetIndexedColor() will automatically call
     * SetNumberOfIndexedColors(index+1) if the current number of indexed colors
     * is not sufficient for the specified index and all will be initialized to
     * the RGBA/RGB values passed to this call.
     */
    void SetIndexedColorRGB(unsigned int index, const double rgb[3]) {
        this->SetIndexedColor(index, rgb[0], rgb[1], rgb[2]);
    }
    void SetIndexedColorRGBA(unsigned int index, const double rgba[4]) {
        this->SetIndexedColor(index, rgba[0], rgba[1], rgba[2]);
    }

    void ResetActiveAnnotatedValues();
    /// @param value Active annotation value
    void SetActiveAnnotatedValue(std::string value);

    /// @param n Number of indexed colors
    void SetNumberOfIndexedColorsInFullSet(int n);
    int GetNumberOfIndexedColorsInFullSet();
    /// @param index Color index
    /// @param r Red component
    /// @param g Green component
    /// @param b Blue component
    void SetIndexedColorInFullSet(unsigned int index,
                                  double r,
                                  double g,
                                  double b);
    /// @param index Color index
    /// @param rgb Output array [r,g,b]
    void GetIndexedColorInFullSet(unsigned int index, double rgb[3]);

    /// @param n Number of indexed opacities
    void SetNumberOfIndexedOpacitiesInFullSet(int n);
    int GetNumberOfIndexedOpacitiesInFullSet();
    /// @param index Opacity index
    /// @param alpha Opacity value
    void SetIndexedOpacityInFullSet(unsigned int index, double alpha);
    /// @param index Opacity index
    /// @param alpha Output opacity value
    void GetIndexedOpacityInFullSet(unsigned int index, double* alpha);

    //@{
    /**
     * Set whether to use restrict annotations to only the values
     * designated as active. Off by default.
     */
    vtkSetMacro(UseActiveValues, bool);
    vtkGetMacro(UseActiveValues, bool);
    vtkBooleanMacro(UseActiveValues, bool);
    //@}

    /**
     * Override to set only the active annotations
     */
    void Build() override;

protected:
    vtkDiscretizableColorTransferFunctionCustom();
    ~vtkDiscretizableColorTransferFunctionCustom() override;

private:
    vtkDiscretizableColorTransferFunctionCustom(
            const vtkDiscretizableColorTransferFunctionCustom&) = delete;
    void operator=(const vtkDiscretizableColorTransferFunctionCustom&) = delete;

    //@{
    /**
     * All annotations.
     */
    vtkAbstractArray* AnnotatedValuesInFullSet;
    vtkStringArray* AnnotationsInFullSet;
    //@}

    vtkDoubleArray* IndexedColorsInFullSet;
    vtkDoubleArray* IndexedOpacitiesInFullSet;

    /**
     * Set of active annotations.
     */
    vtkVariantArray* ActiveAnnotatedValues;

    /**
     * Set whether only "active" annotations should be display. If off, show all
     * annotations.
     */
    bool UseActiveValues;

    /**
     * Build time for this subclass.
     */
    vtkTimeStamp BuildTime;
};
