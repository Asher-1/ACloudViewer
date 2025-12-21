// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <vtkOrientedGlyphContourRepresentation.h>
#include <vtkSmartPointer.h>
#include <vtkTextActor.h>  // Full include required for vtkSmartPointer template instantiation

/**
 * @brief Extended contour representation with instance label support
 *
 * Extends vtkOrientedGlyphContourRepresentation to add label suffix
 * functionality, allowing multiple contour instances to be identified
 * by their labels displayed in the VTK view.
 */
class cvConstrainedContourRepresentation
    : public vtkOrientedGlyphContourRepresentation {
public:
    static cvConstrainedContourRepresentation* New();
    vtkTypeMacro(cvConstrainedContourRepresentation,
                 vtkOrientedGlyphContourRepresentation);

    /**
     * @brief Set the label suffix to identify this contour instance
     * @param suffix The suffix string (e.g., "#1", "#2")
     */
    void SetLabelSuffix(const char* suffix);

    /**
     * @brief Get the current label suffix
     */
    const char* GetLabelSuffix() const;

    /**
     * @brief Show or hide the instance label
     */
    void SetShowLabel(int show);
    int GetShowLabel() const { return this->ShowLabel; }

    /**
     * @brief Build the representation (override to update label position)
     */
    void BuildRepresentation() override;

    /**
     * @brief Set the renderer (override to add label actor)
     */
    void SetRenderer(vtkRenderer* ren) override;

    /**
     * @brief Set visibility (override to control label visibility)
     */
    void SetVisibility(vtkTypeBool visible) override;

    /**
     * @brief Get the label actor (for property access)
     */
    vtkTextActor* GetLabelActor() { return this->LabelActor; }

protected:
    cvConstrainedContourRepresentation();
    ~cvConstrainedContourRepresentation() override;

    /**
     * @brief Update the label text and position
     */
    void UpdateLabel();

private:
    cvConstrainedContourRepresentation(
            const cvConstrainedContourRepresentation&) = delete;
    void operator=(const cvConstrainedContourRepresentation&) = delete;

    vtkSmartPointer<vtkTextActor> LabelActor;
    char* LabelSuffix;
    int ShowLabel;
};
