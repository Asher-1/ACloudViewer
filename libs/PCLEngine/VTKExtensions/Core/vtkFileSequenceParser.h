// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "qPCL.h"  //needed for exports
#include "vtkObject.h"

namespace vtksys {
class RegularExpression;
}

class QPCL_ENGINE_LIB_API vtkFileSequenceParser : public vtkObject {
public:
    static vtkFileSequenceParser* New();
    vtkTypeMacro(vtkFileSequenceParser, vtkObject);
    void PrintSelf(ostream& os, vtkIndent indent) override;

    /**
     * Extract base file name sequence from the file.
     * Returns true if a sequence is detected and
     * sets SequenceName and SequenceIndex.
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
