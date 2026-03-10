// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/**
 * @file vtkUndoSet.h
 * @brief Collection of undo elements representing a single logical change
 */

#include "qVTK.h"  // needed for export macro
#include "vtkObject.h"

class vtkCollection;
class vtkPVXMLElement;
class vtkUndoElement;

/**
 * @class vtkUndoSet
 * @brief Container of vtkUndoElement objects for grouped undo/redo
 */
class QVTK_ENGINE_LIB_API vtkUndoSet : public vtkObject {
public:
    static vtkUndoSet* New();
    vtkTypeMacro(vtkUndoSet, vtkObject);
    void PrintSelf(ostream& os, vtkIndent indent) override;

    /**
     * Perform an Undo.
     */
    virtual int Undo();

    /**
     * Perform a Redo.
     */
    virtual int Redo();

    /**
     * Add an element to this set. If the newly added element, \c elem, and
     * the most recently added element are both \c Mergeable, then an
     * attempt is made to merge the new element with the previous one. On
     * successful merging, the new element is discarded, otherwise
     * it is appended to the set.
     * @param elem Element to add
     * @return Index at which the element got added/merged
     */
    int AddElement(vtkUndoElement* elem);

    /**
     * Remove an element at a particular index.
     * @param index Index of element to remove
     */
    void RemoveElement(int index);

    /**
     * Get an element at a particular index
     * @param index Index of element
     * @return Element at index or nullptr
     */
    vtkUndoElement* GetElement(int index);

    /**
     * Remove all elemments.
     */
    void RemoveAllElements();

    /**
     * Get number of elements in the set.
     * @return Count of elements
     */
    int GetNumberOfElements();

protected:
    vtkUndoSet();
    ~vtkUndoSet() override;

    vtkCollection* Collection;
    vtkCollection* TmpWorkingCollection;

private:
    vtkUndoSet(const vtkUndoSet&) = delete;
    void operator=(const vtkUndoSet&) = delete;
};
