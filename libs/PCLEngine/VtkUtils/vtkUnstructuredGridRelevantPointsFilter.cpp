/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkUnstructuredGridRelevantPointsFilter.C,v $
  Language:  C++
  Date:      $Date: 2000/05/17 16:05:12 $
  Version:   $Revision: 1.54 $


Copyright (c) 1993-2000 Ken Martin, Will Schroeder, Bill Lorensen
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

 * Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

 * Neither name of Ken Martin, Will Schroeder, or Bill Lorensen nor the names
   of any contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

 * Modified source versions must be plainly marked as such, and must not be
   misrepresented as being the original software.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

=========================================================================*/
#include "vtkUnstructuredGridRelevantPointsFilter.h"

#include <vtkCellArray.h>
#include <vtkCellData.h>
#include <vtkIdList.h>
#include <vtkMergePoints.h>
#include <vtkObjectFactory.h>
#include <vtkPointData.h>
#include <vtkUnstructuredGrid.h>

#include "vtkInformation.h"
#include "vtkInformationVector.h"

//------------------------------------------------------------------------------
// Modifications:
//   Kathleen Bonnell, Wed Mar  6 17:10:03 PST 2002
//   Replace 'New' method with Macro to match VTK 4.0 API.
//------------------------------------------------------------------------------

vtkStandardNewMacro(vtkUnstructuredGridRelevantPointsFilter);

//------------------------------------------------------------------------------
// Modifications:
//
//   Hank Childs, Sun Mar 13 14:12:38 PST 2005
//   Fix memory leak.
//
//   Hank Childs, Sun Mar 29 16:26:27 CDT 2009
//   Remove call to BuildLinks.
//
//------------------------------------------------------------------------------

int vtkUnstructuredGridRelevantPointsFilter::RequestData(
        vtkInformation *vtkNotUsed(request),
        vtkInformationVector **inputVector,
        vtkInformationVector *outputVector) {
    // get the info objects
    vtkInformation *inInfo = inputVector[0]->GetInformationObject(0);
    vtkInformation *outInfo = outputVector->GetInformationObject(0);

    // get the input and output
    vtkUnstructuredGrid *input = vtkUnstructuredGrid::SafeDownCast(
            inInfo->Get(vtkDataObject::DATA_OBJECT()));
    vtkUnstructuredGrid *output = vtkUnstructuredGrid::SafeDownCast(
            outInfo->Get(vtkDataObject::DATA_OBJECT()));

    vtkDebugMacro(<< "Beginning UnstructuredGrid Relevant Points Filter ");

    if (input == NULL) {
        vtkErrorMacro(<< "Input is NULL");
        return 0;
    }

    vtkPoints *inPts = input->GetPoints();
    int numInPts = input->GetNumberOfPoints();
    int numCells = input->GetNumberOfCells();
    output->Allocate(numCells);

    if ((numInPts < 1) || (inPts == NULL)) {
        vtkErrorMacro(<< "No data to Operate On!");
        return 0;
    }

    int *pointMap = new int[numInPts];
    for (int i = 0; i < numInPts; i++) {
        pointMap[i] = -1;
    }
    vtkCellArray *cells = input->GetCells();
    vtkIdType npts;
#ifdef VTK_CELL_ARRAY_V2
    const vtkIdType *pts;
#else   // VTK_CELL_ARRAY_V2
    vtkIdType *pts;
#endif  // VTK_CELL_ARRAY_V2

    cells->InitTraversal();
    int numOutPts = 0;
    while (cells->GetNextCell(npts, pts)) {
        for (vtkIdType j = 0; j < npts; j++) {
            vtkIdType pointId = pts[j];
            if (pointMap[pointId] == -1) pointMap[pointId] = numOutPts++;
        }
    }

    vtkPoints *newPts = vtkPoints::New();
    newPts->SetNumberOfPoints(numOutPts);
    vtkPointData *inputPD = input->GetPointData();
    vtkPointData *outputPD = output->GetPointData();
    outputPD->CopyAllocate(inputPD, numOutPts);

    for (int j = 0; j < numInPts; j++) {
        if (pointMap[j] != -1) {
            double pt[3];
            inPts->GetPoint(j, pt);
            newPts->SetPoint(pointMap[j], pt);
            outputPD->CopyData(inputPD, j, pointMap[j]);
        }
    }

    vtkCellData *inputCD = input->GetCellData();
    vtkCellData *outputCD = output->GetCellData();
    outputCD->PassData(inputCD);

    vtkIdList *cellIds = vtkIdList::New();

    output->SetPoints(newPts);

    // now work through cells, changing associated point id to coincide
    // with the new ones as specified in the pointmap;

    vtkIdList *oldIds = vtkIdList::New();
    vtkIdList *newIds = vtkIdList::New();
    int id, cellType;

    cells->InitTraversal();
    while (cells->GetNextCell(npts, pts)) {
        cellType = input->GetCellType(cells->GetTraversalCellId());
        newIds->SetNumberOfIds(npts);
        for (vtkIdType i = 0; i < npts; ++i) {
            vtkIdType pointId = pts[i];
            newIds->SetId(i, pointMap[pointId]);
        }
        output->InsertNextCell(cellType, newIds);
    }

    newPts->Delete();
    oldIds->Delete();
    newIds->Delete();
    cellIds->Delete();
    delete[] pointMap;

    return 1;
}

//------------------------------------------------------------------------------
void vtkUnstructuredGridRelevantPointsFilter::PrintSelf(ostream &os,
                                                        vtkIndent indent) {
    this->Superclass::PrintSelf(os, indent);
}
