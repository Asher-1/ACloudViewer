// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Custom handle representation that supports custom translation axis
// This backports VTK 9.3+ functionality to VTK 9.2 and earlier

#ifndef cvCustomAxisHandleRepresentation_h
#define cvCustomAxisHandleRepresentation_h

#include <vtkPointHandleRepresentation3D.h>
#include <vtkVersion.h>

/**
 * @brief Handle representation with custom translation axis support
 *
 * This class extends vtkPointHandleRepresentation3D to add support for
 * custom translation axes, which is available in VTK 9.3+ but not in
 * earlier versions. It follows ParaView's implementation approach.
 */
class cvCustomAxisHandleRepresentation : public vtkPointHandleRepresentation3D {
public:
    static cvCustomAxisHandleRepresentation* New();
    vtkTypeMacro(cvCustomAxisHandleRepresentation,
                 vtkPointHandleRepresentation3D);
    void PrintSelf(ostream& os, vtkIndent indent) override;

#if !((VTK_MAJOR_VERSION > 9) || \
      (VTK_MAJOR_VERSION == 9 && VTK_MINOR_VERSION >= 3))
    // These methods are only needed for VTK < 9.3
    // In VTK 9.3+, the base class already has these

    /**
     * @brief Enable custom translation axis mode
     */
    void SetCustomTranslationAxisOn();

    /**
     * @brief Disable custom translation axis mode
     */
    void SetCustomTranslationAxisOff();

    /**
     * @brief Set the custom translation axis vector
     * @param axis The axis vector (will be normalized internally)
     */
    void SetCustomTranslationAxis(double axis[3]);
    void SetCustomTranslationAxis(double x, double y, double z);

    /**
     * @brief Get the custom translation axis
     */
    vtkGetVector3Macro(CustomTranslationAxis, double);

    /**
     * @brief Check if custom translation axis is enabled
     */
    bool IsCustomAxisEnabled() const { return this->CustomAxisEnabled; }

    /**
     * @brief Override GetTranslationVector to support custom axis
     */
    void GetTranslationVector(const double* p1,
                              const double* p2,
                              double* v) const override;

    /**
     * @brief Override Translate to support custom axis
     */
    void Translate(const double* p1, const double* p2) override;
    void Translate(const double* v) override;

protected:
    cvCustomAxisHandleRepresentation();
    ~cvCustomAxisHandleRepresentation() override;

    bool CustomAxisEnabled;
    double CustomTranslationAxis[3];

#else
    // For VTK 9.3+, just forward to base class methods
    void SetCustomTranslationAxisOn() {
        this->Superclass::SetCustomTranslationAxisOn();
    }

    void SetCustomTranslationAxisOff() {
        this->Superclass::SetTranslationAxisOff();
    }

    void SetCustomTranslationAxis(double axis[3]) {
        this->Superclass::SetCustomTranslationAxis(axis);
    }

    void SetCustomTranslationAxis(double x, double y, double z) {
        double axis[3] = {x, y, z};
        this->Superclass::SetCustomTranslationAxis(axis);
    }

protected:
    cvCustomAxisHandleRepresentation() {}
    ~cvCustomAxisHandleRepresentation() override {}
#endif

private:
    cvCustomAxisHandleRepresentation(const cvCustomAxisHandleRepresentation&) =
            delete;
    void operator=(const cvCustomAxisHandleRepresentation&) = delete;
};

#endif  // cvCustomAxisHandleRepresentation_h
