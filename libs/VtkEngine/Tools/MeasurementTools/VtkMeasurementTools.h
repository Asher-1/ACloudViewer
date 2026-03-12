// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/**
 * @file VtkMeasurementTools.h
 * @brief VTK-based measurement tools (distance, contour, protractor).
 *
 * Implements ecvGenericMeasurementTools using VTK widgets for
 * interactive 3D measurements in the VtkVis visualizer.
 */

// LOCAL
#include "qVTK.h"

// CV_DB_LIB
#include <ecvGenericMeasurementTools.h>
#include <ecvGenericVisualizer3D.h>

namespace Visualization {
class VtkVis;
}

class cvGenericMeasurementTool;

class QVTK_ENGINE_LIB_API VtkMeasurementTools
    : public ecvGenericMeasurementTools {
    Q_OBJECT
public:
    explicit VtkMeasurementTools(
            MeasurementType type = MeasurementType::DISTANCE_WIDGET);
    explicit VtkMeasurementTools(
            ecvGenericVisualizer3D* viewer,
            MeasurementType type = MeasurementType::DISTANCE_WIDGET);
    virtual ~VtkMeasurementTools() override;

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
    virtual void setColor(double r, double g, double b) override;
    //! Get measurement color (RGB values in range [0.0, 1.0])
    //! Returns false if not implemented, true if color is retrieved
    virtual bool getColor(double& r, double& g, double& b) const;
    virtual void lockInteraction() override;
    virtual void unlockInteraction() override;
    virtual void setInstanceLabel(const QString& label) override;
    virtual void setFontFamily(const QString& family) override;
    virtual void setFontSize(int size) override;
    virtual void setBold(bool bold) override;
    virtual void setItalic(bool italic) override;
    virtual void setShadow(bool shadow) override;
    virtual void setFontOpacity(double opacity) override;
    virtual void setFontColor(double r, double g, double b) override;

    //! Get font properties (for UI synchronization)
    QString getFontFamily() const;
    int getFontSize() const;
    void getFontColor(double& r, double& g, double& b) const;
    bool getFontBold() const;
    bool getFontItalic() const;
    bool getFontShadow() const;
    double getFontOpacity() const;
    QString getHorizontalJustification() const;
    QString getVerticalJustification() const;
    virtual void setHorizontalJustification(
            const QString& justification) override;
    virtual void setVerticalJustification(
            const QString& justification) override;
    virtual void setupShortcuts(QWidget* win) override;
    virtual void disableShortcuts() override;
    virtual void clearPickingCache() override;

protected:
    virtual void initialize() override;

private:
    Visualization::VtkVis* m_viewer;
    cvGenericMeasurementTool* m_tool;
};
