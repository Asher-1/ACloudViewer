// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

/**
 * @file cvHardwareSelector.cpp
 * @brief Implementation of cvHardwareSelector
 *
 * This implementation is adapted from ParaView's vtkPVHardwareSelector.cxx
 * with ParaView-specific dependencies removed.
 *
 * Reference: ParaView/Remoting/Views/vtkPVHardwareSelector.cxx
 */

#include "cvHardwareSelector.h"

// CloudViewer
#include <CVLog.h>
#include <vtkActor.h>
#include <vtkActorCollection.h>
#include <vtkDataObject.h>
#include <vtkMapper.h>
#include <vtkObjectFactory.h>
#include <vtkProp.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkSelection.h>

// For debug output (optional)
// #define cvHardwareSelectorDEBUG
#ifdef cvHardwareSelectorDEBUG
#include <vtkImageImport.h>
#include <vtkNew.h>
#include <vtkPNMWriter.h>

#include <sstream>
#endif

//----------------------------------------------------------------------------
vtkStandardNewMacro(cvHardwareSelector);

//----------------------------------------------------------------------------
cvHardwareSelector::cvHardwareSelector() {
    // ParaView-style: Use process ID from data
    // Reference: vtkPVHardwareSelector constructor line 43
    this->SetUseProcessIdFromData(true);
    this->ProcessID = 0;
    this->UniqueId = 0;
    // ParaView code default is 0 (disabled), but settings XML sets it to 50
    this->PointPickingRadius = 50;
}

//----------------------------------------------------------------------------
cvHardwareSelector::~cvHardwareSelector() { this->PropMap.clear(); }

//----------------------------------------------------------------------------
bool cvHardwareSelector::PassRequired(int pass) {
    // Reference: vtkPVHardwareSelector::PassRequired() lines 63-84
    // Always require the process pass on first iteration
    if (pass == PROCESS_PASS && this->Iteration == 0) {
        return true;
    }

    return this->Superclass::PassRequired(pass);
}

//----------------------------------------------------------------------------
bool cvHardwareSelector::PrepareSelect() {
    // Reference: vtkPVHardwareSelector::PrepareSelect() lines 87-102
    bool needRender = this->NeedToRenderForSelection();

    if (needRender) {
        if (!this->Renderer) {
            CVLog::Error(
                    "[cvHardwareSelector::PrepareSelect] No renderer set!");
            return false;
        }

        int* size = this->Renderer->GetSize();
        int* origin = this->Renderer->GetOrigin();

        this->SetArea(origin[0], origin[1], origin[0] + size[0] - 1,
                      origin[1] + size[1] - 1);

        if (this->CaptureBuffers() == false) {
            CVLog::Error(
                    "[cvHardwareSelector::PrepareSelect] CaptureBuffers "
                    "failed!");
            this->CaptureTime.Modified();
            return false;
        }
        this->CaptureTime.Modified();
    }
    return true;
}

//----------------------------------------------------------------------------
vtkSelection* cvHardwareSelector::Select(int region[4]) {
    // Reference: vtkPVHardwareSelector::Select() lines 105-131
    if (!this->PrepareSelect()) {
        CVLog::Error("[cvHardwareSelector::Select] PrepareSelect failed!");
        return nullptr;
    }

    vtkSelection* sel =
            this->GenerateSelection(region[0], region[1], region[2], region[3]);

    // ParaView-style Point Picking Radius support
    // Reference: vtkPVHardwareSelector::Select() lines 113-129
    // Only applies to:
    // 1. Empty selection (no direct hit)
    // 2. Point selection (not cell)
    // 3. Single point click (not drag)
    // 4. PointPickingRadius > 0
    bool noNodes = (sel->GetNumberOfNodes() == 0);
    bool isPointSelection =
            (this->FieldAssociation == vtkDataObject::FIELD_ASSOCIATION_POINTS);
    bool isSingleClick = (region[0] == region[2] && region[1] == region[3]);
    bool hasRadius = (this->PointPickingRadius > 0);

    if (noNodes && isPointSelection && isSingleClick && hasRadius) {
        unsigned int pos[2];
        pos[0] = static_cast<unsigned int>(region[0]);
        pos[1] = static_cast<unsigned int>(region[1]);

        unsigned int out_pos[2];
        vtkHardwareSelector::PixelInformation info = this->GetPixelInformation(
                pos, this->PointPickingRadius, out_pos);

        if (info.Valid) {
            sel->Delete();
            return this->GenerateSelection(out_pos[0], out_pos[1], out_pos[0],
                                           out_pos[1]);
        }
    }
    return sel;
}

//----------------------------------------------------------------------------
vtkSelection* cvHardwareSelector::PolygonSelect(int* polygonPoints,
                                                vtkIdType count) {
    // Reference: vtkPVHardwareSelector::PolygonSelect() lines 134-141
    if (!this->PrepareSelect()) {
        return nullptr;
    }
    return this->GeneratePolygonSelection(polygonPoints, count);
}

//----------------------------------------------------------------------------
bool cvHardwareSelector::NeedToRenderForSelection() {
    // Reference: vtkPVHardwareSelector::NeedToRenderForSelection() lines
    // 144-150 We rely on external logic to ensure that the MTime for the
    // cvHardwareSelector is explicitly modified when some action happens that
    // would result in invalidation of captured buffers.
    return this->CaptureTime < this->GetMTime();
}

//----------------------------------------------------------------------------
int cvHardwareSelector::AssignUniqueId(vtkProp* prop) {
    // Reference: vtkPVHardwareSelector::AssignUniqueId() lines 153-159
    int id = this->UniqueId;
    this->UniqueId++;
    this->PropMap[prop] = id;
    return id;
}

//----------------------------------------------------------------------------
int cvHardwareSelector::GetPropID(int idx, vtkProp* prop) {
    // Reference: vtkPVHardwareSelector::GetPropID() lines 162-166
    // First try to find in PropMap (for explicitly assigned IDs via
    // AssignUniqueId)
    auto iter = this->PropMap.find(prop);
    if (iter != this->PropMap.end()) {
        return iter->second;
    }

    // CRITICAL FIX: Fall back to VTK's default behavior using idx
    // ParaView's vtkPVRenderView calls AssignUniqueId to populate PropMap,
    // but ACloudViewer doesn't have this logic. Without this fallback,
    // GetPropID returns -1, causing ACTOR_PASS to output 0 (invalid ID).
    // The idx parameter is the sequential index of the prop being rendered,
    // which VTK uses as the default prop ID.
    return idx;
}

//----------------------------------------------------------------------------
void cvHardwareSelector::PrintSelf(ostream& os, vtkIndent indent) {
    this->Superclass::PrintSelf(os, indent);
    os << indent << "PointPickingRadius: " << this->PointPickingRadius << "\n";
    os << indent << "UniqueId: " << this->UniqueId << "\n";
    os << indent << "PropMap size: " << this->PropMap.size() << "\n";
}

//----------------------------------------------------------------------------
void cvHardwareSelector::BeginSelection() {
    // Call parent which sets this->Renderer->SetSelector(this)
    this->Superclass::BeginSelection();
}

//----------------------------------------------------------------------------
void cvHardwareSelector::EndSelection() { this->Superclass::EndSelection(); }

//----------------------------------------------------------------------------
void cvHardwareSelector::BeginRenderProp(vtkRenderWindow* rw) {
    // Reference: vtkPVHardwareSelector::BeginRenderProp() lines 175-181
    // In ParaView, this sets ProcessID from vtkProcessModule
    // For ACloudViewer (single process), we keep ProcessID = 0
    this->Superclass::BeginRenderProp(rw);
}

//----------------------------------------------------------------------------
void cvHardwareSelector::SavePixelBuffer(int passNo) {
    // Reference: vtkPVHardwareSelector::SavePixelBuffer() lines 185-206
    this->Superclass::SavePixelBuffer(passNo);

#ifdef cvHardwareSelectorDEBUG
    vtkNew<vtkImageImport> ii;
    ii->SetImportVoidPointer(this->PixBuffer[passNo], 1);
    ii->SetDataScalarTypeToUnsignedChar();
    ii->SetNumberOfScalarComponents(3);
    ii->SetDataExtent(this->Area[0], this->Area[2], this->Area[1],
                      this->Area[3], 0, 0);
    ii->SetWholeExtent(this->Area[0], this->Area[2], this->Area[1],
                       this->Area[3], 0, 0);

    std::ostringstream fname;
    fname << "/tmp/cv-buffer-pass-" << passNo << ".pnm";
    vtkNew<vtkPNMWriter> pw;
    pw->SetInputConnection(ii->GetOutputPort());
    pw->SetFileName(fname.str().c_str());
    pw->Write();
#endif
}
