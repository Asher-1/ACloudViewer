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
    void PrintSelf(ostream& os, vtkIndent indent);

#if !((VTK_MAJOR_VERSION > 9) || \
      (VTK_MAJOR_VERSION == 9 && VTK_MINOR_VERSION >= 3))
    // ========================================================================
    // VTK < 9.3 specific - backport VTK 9.3+ functionality
    // Implement all VTK 9.3+ interfaces for compatibility
    // ========================================================================

    /**
     * @brief Enable custom translation axis mode (backport from VTK 9.3+)
     */
    void SetCustomTranslationAxisOn();

    /**
     * @brief Disable custom translation axis mode (backport from VTK 9.3+)
     */
    void SetCustomTranslationAxisOff();

    /**
     * @brief Set the custom translation axis vector (backport from VTK 9.3+)
     * @param axis The axis vector (will be normalized internally)
     */
    void SetCustomTranslationAxis(double axis[3]);
    void SetCustomTranslationAxis(double x, double y, double z);

    /**
     * @brief Override SetTranslationAxisOff to also turn off custom axis
     * This ensures ParaView-style unified release works correctly
     */
    void SetTranslationAxisOff();

    /**
     * @brief Get the custom translation axis (backport from VTK 9.3+)
     */
    vtkGetVector3Macro(CustomTranslationAxis, double);

    /**
     * @brief Check if custom translation axis is enabled
     */
    bool IsCustomAxisEnabled() const { return this->CustomAxisEnabled; }

protected:
    cvCustomAxisHandleRepresentation();
    ~cvCustomAxisHandleRepresentation();

    /**
     * @brief Override GetTranslationVector to support custom axis
     * Note: Not marked override for VTK 9.2 compatibility
     */
    void GetTranslationVector(const double* p1,
                              const double* p2,
                              double* v) const;

    /**
     * @brief Override Translate to support custom axis
     * Note: Not marked override for VTK 9.2 compatibility
     */
    void Translate(const double* p1, const double* p2);
    void Translate(const double* v);

    /**
     * @brief Override DetermineConstraintAxis to prevent automatic axis
     * determination when custom axis is enabled
     * Note: Not marked override for VTK 9.2 compatibility
     */
    int DetermineConstraintAxis(int constraint,
                                double* x,
                                double* startPickPoint);

    bool CustomAxisEnabled;
    double CustomTranslationAxis[3];

#else
    // ========================================================================
    // VTK 9.3+ specific - minimal implementation
    // Base class already provides complete custom axis support
    // No need to override anything, just inherit base class functionality
    // ========================================================================

protected:
    cvCustomAxisHandleRepresentation() {}
    ~cvCustomAxisHandleRepresentation() {}
#endif

private:
    cvCustomAxisHandleRepresentation(const cvCustomAxisHandleRepresentation&) =
            delete;
    void operator=(const cvCustomAxisHandleRepresentation&) = delete;
};

#endif  // cvCustomAxisHandleRepresentation_h
