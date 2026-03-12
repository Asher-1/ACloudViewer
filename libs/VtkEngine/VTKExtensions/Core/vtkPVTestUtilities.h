// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/**
 * @file vtkPVTestUtilities.h
 * @brief Test utilities for data paths and command-line argument parsing
 */

#include "qVTK.h"  // needed for export macro
#include "vtkObject.h"

class vtkPolyData;
class vtkDataArray;

/**
 * @class vtkPVTestUtilities
 * @brief Helper for tests: data root, temp root, and path construction
 */
class QVTK_ENGINE_LIB_API vtkPVTestUtilities : public vtkObject {
public:
    // the usual vtk stuff
    vtkTypeMacro(vtkPVTestUtilities, vtkObject);
    void PrintSelf(ostream& os, vtkIndent indent) override;
    static vtkPVTestUtilities* New();

    /**
     * Initialize the object from command tail arguments.
     * @param argc Argument count
     * @param argv Argument vector (e.g. -D for data root, -T for temp root)
     */
    void Initialize(int argc, char** argv);
    /**
     * Given a path relative to the Data root (provided
     * in argv by -D option), construct a OS independent path
     * to the file specified by "name". "name" should not start
     * with a path separator and if path separators are needed
     * use '/'. Be sure to delete [] the return when you are
     * finished.
     */
    char* GetDataFilePath(const char* name) {
        return this->GetFilePath(this->DataRoot, name);
    }
    /**
     * Given a path relative to the working directory (provided
     * in argv by -T option), construct a OS independent path
     * to the file specified by "name". "name" should not start
     * with a path separator and if path separators are needed
     * use '/'. Be sure to delete [] the return when you are
     * finished.
     */
    char* GetTempFilePath(const char* name) {
        return this->GetFilePath(this->TempRoot, name);
    }

protected:
    vtkPVTestUtilities() { this->Initialize(0, 0); }
    ~vtkPVTestUtilities() override { this->Initialize(0, 0); }

private:
    vtkPVTestUtilities(const vtkPVTestUtilities&) = delete;
    void operator=(const vtkPVTestUtilities&) = delete;
    ///
    char GetPathSep();
    char* GetDataRoot();
    char* GetTempRoot();
    char* GetCommandTailArgument(const char* tag);
    char* GetFilePath(const char* base, const char* name);
    //
    int Argc;
    char** Argv;
    char* DataRoot;
    char* TempRoot;
};
