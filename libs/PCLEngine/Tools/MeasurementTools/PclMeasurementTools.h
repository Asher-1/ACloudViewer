// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// LOCAL
#include "PclUtils/PCLCloud.h"
#include "qPCL.h"

// ECV_DB_LIB
#include <ecvGenericMeasurementTools.h>
#include <ecvGenericVisualizer3D.h>

namespace PclUtils {
class PCLVis;
}

class cvGenericMeasurementTool;

class QPCL_ENGINE_LIB_API PclMeasurementTools
    : public ecvGenericMeasurementTools {
    Q_OBJECT
public:
    explicit PclMeasurementTools(
            MeasurementType type = MeasurementType::DISTANCE_WIDGET);
    explicit PclMeasurementTools(
            ecvGenericVisualizer3D* viewer,
            MeasurementType type = MeasurementType::DISTANCE_WIDGET);
    virtual ~PclMeasurementTools() override;

    void setVisualizer(ecvGenericVisualizer3D* viewer = nullptr);

public:  // implemented from ecvGenericMeasurementTools interface
    virtual bool setInputData(ccHObject* entity) override;
    virtual bool start() override;
    virtual void reset() override;
    virtual void clear() override;
    virtual QWidget* getMeasurementWidget() override;
    virtual ccHObject* getOutput() const override;

    virtual double getMeasurementValue() const override;
    virtual void getPoint1(double pos[3]) const override;
    virtual void getPoint2(double pos[3]) const override;
    virtual void getCenter(double pos[3]) const override;
    virtual void setPoint1(double pos[3]) override;
    virtual void setPoint2(double pos[3]) override;
    virtual void setCenter(double pos[3]) override;
    virtual void setupShortcuts(QWidget* win) override;

protected:
    virtual void initialize() override;

private:
    PclUtils::PCLVis* m_viewer;
    cvGenericMeasurementTool* m_tool;
};
