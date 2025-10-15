// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <memory>  // for std::unique_ptr

#include "qPCL.h"  // needed for export macro
#include "vtkObject.h"

class QPCL_ENGINE_LIB_API vtkStringList : public vtkObject {
public:
    static vtkStringList* New();
    vtkTypeMacro(vtkStringList, vtkObject);
    void PrintSelf(ostream& os, vtkIndent indent) override;

    //@{
    /**
     * Add a simple string.
     */
    void AddString(const char* str);
    void AddUniqueString(const char* str);
    //@}

    /**
     * Add a command and format it any way you like.
     */
    void AddFormattedString(const char* EventString, ...);

    /**
     * Initialize to empty.
     */
    void RemoveAllItems();

    /**
     * Random access.
     */
    void SetString(int idx, const char* str);

    /**
     * Get the length of the list.
     */
    int GetLength() { return this->GetNumberOfStrings(); }

    /**
     * Get the index of a string.
     */
    int GetIndex(const char* str);

    /**
     * Get a command from its index.
     */
    const char* GetString(int idx);

    /**
     * Returns the number of strings.
     */
    int GetNumberOfStrings();

protected:
    vtkStringList();
    ~vtkStringList() override;

private:
    class vtkInternals;
    std::unique_ptr<vtkInternals> Internals;

    vtkStringList(const vtkStringList&) = delete;
    void operator=(const vtkStringList&) = delete;
};
