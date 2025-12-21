// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "eCV_db.h"

// QT
#include <QObject>

class ccHObject;
class QWidget;

//! Generic Measurement Tools interface
class ECV_DB_LIB_API ecvGenericMeasurementTools : public QObject {
    Q_OBJECT
public:
    enum MeasurementType { DISTANCE_WIDGET, CONTOUR_WIDGET, PROTRACTOR_WIDGET };

    //! Default constructor
    /**
            \param type MeasurementType type
    **/
    ecvGenericMeasurementTools(
            MeasurementType type = MeasurementType::DISTANCE_WIDGET);

    //! Destructor
    virtual ~ecvGenericMeasurementTools();

    //! Sets the input entity
    virtual bool setInputData(ccHObject* entity) = 0;

    //! Starts the measurement tool
    virtual bool start() = 0;

    //! Resets the measurement tool
    virtual void reset() = 0;

    //! Clears the measurement tool
    virtual void clear() = 0;

    //! Updates the display
    virtual void update();

    //! Returns the measurement widget
    virtual QWidget* getMeasurementWidget() = 0;

    //! Returns the output (if any)
    virtual ccHObject* getOutput() const = 0;

    //! Get measurement value (distance or angle)
    virtual double getMeasurementValue() const = 0;

    //! Get point 1 coordinates
    virtual void getPoint1(double pos[3]) const = 0;

    //! Get point 2 coordinates
    virtual void getPoint2(double pos[3]) const = 0;

    //! Get center point coordinates (for angle/protractor)
    virtual void getCenter(double pos[3]) const = 0;

    //! Set point 1 coordinates
    virtual void setPoint1(double pos[3]) = 0;

    //! Set point 2 coordinates
    virtual void setPoint2(double pos[3]) = 0;

    //! Set center point coordinates (for angle/protractor)
    virtual void setCenter(double pos[3]) = 0;

    //! Set measurement color (RGB values in range [0.0, 1.0])
    virtual void setColor(double r, double g, double b) = 0;

    //! Lock tool interaction (disable VTK widget and UI controls)
    virtual void lockInteraction() = 0;

    //! Unlock tool interaction (enable VTK widget and UI controls)
    virtual void unlockInteraction() = 0;

    //! Set instance label suffix (e.g., "#1", "#2") for display in 3D view
    virtual void setInstanceLabel(const QString& label) = 0;

    //! Set font family for measurement labels (e.g., "Arial", "Times New Roman")
    virtual void setFontFamily(const QString& family) = 0;

    //! Set font size for measurement labels
    virtual void setFontSize(int size) = 0;

    //! Set font bold state for measurement labels
    virtual void setBold(bool bold) = 0;

    //! Set font italic state for measurement labels
    virtual void setItalic(bool italic) = 0;

    //! Set font shadow state for measurement labels
    virtual void setShadow(bool shadow) = 0;

    //! Set font opacity for measurement labels (0.0 to 1.0)
    virtual void setFontOpacity(double opacity) = 0;

    //! Set font color for measurement labels (RGB values 0.0-1.0)
    virtual void setFontColor(double r, double g, double b) = 0;

    //! Set horizontal justification for measurement labels ("Left", "Center", "Right")
    virtual void setHorizontalJustification(const QString& justification) = 0;

    //! Set vertical justification for measurement labels ("Top", "Center", "Bottom")
    virtual void setVerticalJustification(const QString& justification) = 0;

    //! Setup keyboard shortcuts bound to the render window widget
    virtual void setupShortcuts(QWidget* win) { Q_UNUSED(win); }

    //! Disable keyboard shortcuts (called before tool destruction)
    virtual void disableShortcuts() {}

    //! Clear picking cache (called when scene/camera changes)
    virtual void clearPickingCache() {}

public:
    inline MeasurementType getMeasurementType() { return m_measurementType; }

signals:
    //! Signal sent when the measurement changes
    void measurementChanged();

    //! Signal sent when point picking is requested
    //! @param pointIndex: 1=point1, 2=point2, 3=center
    void pointPickingRequested(int pointIndex);

    //! Signal sent when point picking is cancelled
    void pointPickingCancelled();

protected:
    virtual void initialize() = 0;

    MeasurementType m_measurementType;
    ccHObject* m_associatedEntity;
};
