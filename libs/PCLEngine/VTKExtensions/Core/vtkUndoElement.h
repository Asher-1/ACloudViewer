// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef vtkUndoElement_h
#define vtkUndoElement_h

#include "qPCL.h"  // needed for export macro
#include "vtkObject.h"
class vtkCollection;

class QPCL_ENGINE_LIB_API vtkUndoElement : public vtkObject {
public:
    vtkTypeMacro(vtkUndoElement, vtkObject);
    void PrintSelf(ostream& os, vtkIndent indent) override;

    /**
     * Undo the operation encapsulated by this element.
     * \return the status of the operation, 1 on success, 0 otherwise.
     */
    virtual int Undo() = 0;

    /**
     * Redo the operation encaspsulated by this element.
     * \return the status of the operation, 1 on success, 0 otherwise.
     */
    virtual int Redo() = 0;

    //@{
    /**
     * Returns if this undo element can be merged with other
     * undo elements.
     * When an undo element is added to a vtkUndoSet unsing AddElement,
     * an attempt is made to \c "merge" the element with the
     * most recently added undo element, if any, if both the undo elements
     * are mergeable.
     */
    vtkGetMacro(Mergeable, bool);
    //@}

    /**
     * Called on the older element in the UndoSet to merge with the
     * element being added if  both the elements are \c mergeable.
     * Returns if the merge was successful.
     * Default implementation doesn't do anything.
     */
    virtual bool Merge(vtkUndoElement* vtkNotUsed(new_element)) {
        return false;
    }

    // Set the working context if run inside a UndoSet context, so object
    // that are cross referenced can leave long enough to be associated
    // to another object. Otherwise the undo of a Delete will create the object
    // again but as no-one is holding a reference to that newly created object
    // it will be automatically deleted. Therefore, we provide a collection
    // that will hold a reference during an undoset so the object has a chance
    // to be attached to the ProxyManager or any other object.
    virtual void SetUndoSetWorkingContext(vtkCollection* workCTX) {
        this->UndoSetWorkingContext = workCTX;
    }

protected:
    vtkUndoElement();
    ~vtkUndoElement() override;

    //@{
    /**
     * Subclasses must set this flag to enable merging of consecutive elements
     * in an UndoSet.
     */
    bool Mergeable;
    vtkSetMacro(Mergeable, bool);
    vtkCollection* UndoSetWorkingContext;
    //@}

private:
    vtkUndoElement(const vtkUndoElement&) = delete;
    void operator=(const vtkUndoElement&) = delete;
};

#endif
