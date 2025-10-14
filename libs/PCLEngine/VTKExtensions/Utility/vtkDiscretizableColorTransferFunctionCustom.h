// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef vtkDiscretizableColorTransferFunctionCustom_h
#define vtkDiscretizableColorTransferFunctionCustom_h

#include "qPCL.h"
#include "vtkDiscretizableColorTransferFunction.h"

class vtkAbstractArray;
class vtkDoubleArray;
class vtkVariantArray;

class QPCL_ENGINE_LIB_API vtkDiscretizableColorTransferFunctionCustom
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
    void SetActiveAnnotatedValue(std::string value);

    void SetNumberOfIndexedColorsInFullSet(int n);
    int GetNumberOfIndexedColorsInFullSet();
    void SetIndexedColorInFullSet(unsigned int index,
                                  double r,
                                  double g,
                                  double b);
    void GetIndexedColorInFullSet(unsigned int index, double rgb[3]);

    void SetNumberOfIndexedOpacitiesInFullSet(int n);
    int GetNumberOfIndexedOpacitiesInFullSet();
    void SetIndexedOpacityInFullSet(unsigned int index, double alpha);
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

#endif  // vtkDiscretizableColorTransferFunctionCustom_h
