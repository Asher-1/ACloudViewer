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
    
    //! Setup keyboard shortcuts bound to the render window widget
    virtual void setupShortcuts(QWidget* win) { Q_UNUSED(win); }

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
