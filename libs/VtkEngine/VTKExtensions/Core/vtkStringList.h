// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/**
 * @file vtkStringList.h
 * @brief VTK object for storing and managing a list of strings
 */

#include <memory>  // for std::unique_ptr

#include "qVTK.h"  // needed for export macro
#include "vtkObject.h"

/**
 * @class vtkStringList
 * @brief Container for a list of strings with add/remove/unique operations
 */
class QVTK_ENGINE_LIB_API vtkStringList : public vtkObject {
public:
    static vtkStringList* New();
    vtkTypeMacro(vtkStringList, vtkObject);
    void PrintSelf(ostream& os, vtkIndent indent) override;

    //@{
    /**
     * Add a simple string.
     * @param str String to add
     */
    void AddString(const char* str);
    /**
     * Add a string if not already present.
     * @param str String to add
     */
    void AddUniqueString(const char* str);
    //@}

    /**
     * Add a command and format it any way you like.
     * @param EventString Format string (printf-style)
     * @param ... Format arguments
     */
    void AddFormattedString(const char* EventString, ...);

    /**
     * Initialize to empty.
     */
    void RemoveAllItems();

    /**
     * Random access.
     * @param idx Index of string to set
     * @param str New string value
     */
    void SetString(int idx, const char* str);

    /**
     * Get the length of the list.
     * @return Number of strings
     */
    int GetLength() { return this->GetNumberOfStrings(); }

    /**
     * Get the index of a string.
     * @param str String to find
     * @return Index or -1 if not found
     */
    int GetIndex(const char* str);

    /**
     * Get a command from its index.
     * @param idx Index of string
     * @return String at index or nullptr
     */
    const char* GetString(int idx);

    /**
     * Returns the number of strings.
     * @return Count of strings in list
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
