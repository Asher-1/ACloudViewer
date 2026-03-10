// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/**
 * @file vtkFileSequenceParser.h
 * @brief Parses file names to detect and extract sequence patterns
 */

#include "qVTK.h"  //needed for exports
#include "vtkObject.h"

namespace vtksys {
class RegularExpression;
}

/**
 * @class vtkFileSequenceParser
 * @brief Extracts base name and index from numbered file sequences
 */
class QVTK_ENGINE_LIB_API vtkFileSequenceParser : public vtkObject {
public:
    static vtkFileSequenceParser* New();
    vtkTypeMacro(vtkFileSequenceParser, vtkObject);
    void PrintSelf(ostream& os, vtkIndent indent) override;

    /**
     * Extract base file name and index from a sequence file name.
     * @param file File path (e.g. "data_001.vtk")
     * @return True if sequence detected; sets SequenceName and SequenceIndex
     */
    bool ParseFileSequence(const char* file);

    vtkGetStringMacro(SequenceName);
    vtkGetMacro(SequenceIndex, int);

protected:
    vtkFileSequenceParser();
    ~vtkFileSequenceParser() override;

    vtksys::RegularExpression* reg_ex;
    vtksys::RegularExpression* reg_ex2;
    vtksys::RegularExpression* reg_ex3;
    vtksys::RegularExpression* reg_ex4;
    vtksys::RegularExpression* reg_ex5;
    vtksys::RegularExpression* reg_ex_last;

    // Used internal so char * allocations are done automatically.
    vtkSetStringMacro(SequenceName);

    int SequenceIndex;
    char* SequenceName;

private:
    vtkFileSequenceParser(const vtkFileSequenceParser&) = delete;
    void operator=(const vtkFileSequenceParser&) = delete;
};
