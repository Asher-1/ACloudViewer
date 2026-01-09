// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

/**
 * @file cvHardwareSelector.h
 * @class cvHardwareSelector
 * @brief vtkOpenGLHardwareSelector subclass with ParaView-style buffer caching
 *
 * This class is adapted from ParaView's vtkPVHardwareSelector to provide
 * optimized hardware selection with buffer caching. It avoids recapturing
 * buffers unless needed, improving selection performance.
 *
 * Reference: ParaView/Remoting/Views/vtkPVHardwareSelector.h
 */

#ifndef cvHardwareSelector_h
#define cvHardwareSelector_h

#include <vtkOpenGLHardwareSelector.h>
#include <vtkTimeStamp.h>

#include <map>

#include "qPCL.h"

class vtkProp;
class vtkRenderer;

/**
 * @brief Hardware selector with ParaView-style buffer caching
 *
 * This class extends vtkOpenGLHardwareSelector to add logic for reusing
 * captured buffers as much as possible, avoiding repeated selection-rendering.
 *
 * Key features (from ParaView):
 * - Buffer caching with CaptureTime tracking
 * - Point picking radius support
 * - Polygon selection support
 * - Unique prop ID assignment
 */
class QPCL_ENGINE_LIB_API cvHardwareSelector
    : public vtkOpenGLHardwareSelector {
public:
    static cvHardwareSelector* New();
    vtkTypeMacro(cvHardwareSelector, vtkOpenGLHardwareSelector);
    void PrintSelf(ostream& os, vtkIndent indent) override;

    /**
     * @brief Perform selection over the specified region
     *
     * This method avoids clearing captured buffers if they are still valid.
     * Reference: vtkPVHardwareSelector::Select()
     *
     * @param region Selection region [x1, y1, x2, y2]
     * @return Selection result (caller must Delete())
     */
    virtual vtkSelection* Select(int region[4]);

    /**
     * @brief Perform polygon selection
     *
     * Reference: vtkPVHardwareSelector::PolygonSelect()
     *
     * @param polygonPoints Array of polygon points
     * @param count Number of points
     * @return Selection result (caller must Delete())
     */
    vtkSelection* PolygonSelect(int* polygonPoints, vtkIdType count);

    /**
     * @brief Check if next Select() will need to render
     *
     * Returns true when the next call to Select() will result in renders
     * to capture the selection-buffers.
     *
     * Reference: vtkPVHardwareSelector::NeedToRenderForSelection()
     */
    virtual bool NeedToRenderForSelection();

    /**
     * @brief Invalidate the cached selection buffers
     *
     * Call this when the scene has changed and cached buffers are no longer
     * valid. Reference: vtkPVHardwareSelector::InvalidateCachedSelection()
     */
    void InvalidateCachedSelection() { this->Modified(); }

    /**
     * @brief Assign a unique ID to a prop
     *
     * Used for tracking props during selection.
     * Reference: vtkPVHardwareSelector::AssignUniqueId()
     *
     * @param prop The prop to assign an ID to
     * @return The assigned unique ID
     */
    int AssignUniqueId(vtkProp* prop);

    ///@{
    /**
     * @brief Point picking radius (in pixels)
     *
     * When selecting a single point and no hit is found at the exact pixel,
     * search in this radius for nearby points.
     *
     * Reference: vtkPVRenderViewSettings::GetPointPickingRadius()
     */
    vtkSetMacro(PointPickingRadius, unsigned int);
    vtkGetMacro(PointPickingRadius, unsigned int);
    ///@}

    // Fixes a -Woverloaded-virtual warning
    using vtkOpenGLHardwareSelector::BeginRenderProp;

    /**
     * @brief Begin render prop (sets ProcessID)
     */
    void BeginRenderProp(vtkRenderWindow* rw) override;

    /**
     * @brief Begin selection - sets selector on renderer
     */
    void BeginSelection() override;

    /**
     * @brief End selection - clears selector from renderer
     */
    void EndSelection() override;

protected:
    cvHardwareSelector();
    ~cvHardwareSelector() override;

    /**
     * @brief Return a unique ID for the prop
     *
     * Reference: vtkPVHardwareSelector::GetPropID()
     */
    int GetPropID(int idx, vtkProp* prop) override;

    /**
     * @brief Check if a pass is required
     *
     * Reference: vtkPVHardwareSelector::PassRequired()
     */
    bool PassRequired(int pass) override;

    /**
     * @brief Prepare for selection
     *
     * Captures buffers if needed.
     * Reference: vtkPVHardwareSelector::PrepareSelect()
     *
     * @return true if ready for selection, false if capture failed
     */
    bool PrepareSelect();

    /**
     * @brief Save pixel buffer (with optional debug output)
     */
    void SavePixelBuffer(int passNo) override;

    /// Time when buffers were last captured
    vtkTimeStamp CaptureTime;

    /// Counter for unique prop IDs
    int UniqueId;

    /// Point picking radius in pixels (default: 10)
    unsigned int PointPickingRadius;

private:
    cvHardwareSelector(const cvHardwareSelector&) = delete;
    void operator=(const cvHardwareSelector&) = delete;

    /// Map from prop pointer to unique ID
    std::map<void*, int> PropMap;
};

#endif  // cvHardwareSelector_h
