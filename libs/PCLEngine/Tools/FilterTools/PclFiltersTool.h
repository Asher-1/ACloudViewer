// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// LOCAL
#include "../../qPCL.h"
#include "PclUtils/PCLCloud.h"

// ECV_DB_LIB
#include <ecvGenericFiltersTool.h>

namespace PclUtils {
class PCLVis;
}

class vtkActor;
class cvGenericFilter;

class QPCL_ENGINE_LIB_API PclFiltersTool : public ecvGenericFiltersTool {
    Q_OBJECT
public:
    explicit PclFiltersTool(FilterType type = FilterType::CLIP_FILTER);
    explicit PclFiltersTool(ecvGenericVisualizer3D* viewer,
                            FilterType type = FilterType::CLIP_FILTER);
    virtual ~PclFiltersTool() override;

    /**
     * @brief initialize
     */
    void setVisualizer(ecvGenericVisualizer3D* viewer = nullptr);

public:  // implemented from ecvGenericFiltersTool interface
    virtual void showInteractor(bool state) override;
    virtual ccHObject* getOutput() const override;
    virtual void getOutput(
            std::vector<ccHObject*>& outputSlices,
            std::vector<ccPolyline*>& outputContours) const override;

    virtual void setNegative(bool state) override;

    virtual QWidget* getFilterWidget() override;

    virtual const ccBBox& getBox() override;

    //! Sets the box extents
    virtual void setBox(const ccBBox& box) override;

    //! Shifts the current interactor
    virtual void shift(const CCVector3& v) override;

    //! Manually sets the box parameters
    virtual void set(const ccBBox& extents,
                     const ccGLMatrix& transformation) override;
    virtual void get(ccBBox& extents, ccGLMatrix& transformation) override;

    virtual bool setInputData(ccHObject* entity, int viewport = 0) override;

    virtual void intersectMode() override;
    virtual void unionMode() override;
    virtual void trimMode() override;
    virtual void resetMode() override;

    /**
     * @brief clear state before load new cloud and annotation
     */
    virtual bool start() override;
    virtual void reset() override;
    virtual void restore() override;
    virtual void clear() override;

    virtual void showOutline(bool state) override;

protected:
    virtual void unregisterFilter() override;
    virtual void initialize(ecvGenericVisualizer3D* viewer) override;

protected slots:
    void areaPickingEventProcess(const std::vector<int>& new_selected_slice);

private:
    void setPointSize(const std::string& viewID, int viewport = 0);

private:
    bool m_intersectMode;
    bool m_unionMode;
    bool m_trimMode;

    PclUtils::PCLVis* m_viewer;

    cvGenericFilter* m_filter;

    /**
     * @brief state of each point to identity color or selectable
     */
    int* m_cloudLabel;
    std::vector<int> m_last_selected_slice;
};
