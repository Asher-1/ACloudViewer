// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef ECV_DEPTH_BUFFER_HEADER
#define ECV_DEPTH_BUFFER_HEADER

// local
#include "eCV_db.h"

// CV_CORE_LIB
#include <CVGeom.h>

// System
#include <vector>

//! Sensor "depth map"
/** Contains an array of depth values (along each scanned direction) and its
dimensions. This array corresponds roughly to what have been "seen" by the
sensor during acquisition (the 3D points are simply projected in the sensor
frame).
**/
class ECV_DB_LIB_API ccDepthBuffer {
public:
    //! Default constructor
    ccDepthBuffer();
    //! Destructor
    virtual ~ccDepthBuffer();

    //! Z-Buffer grid
    std::vector<PointCoordinateType> zBuff;
    //! Pitch step (may differ from the sensor's)
    PointCoordinateType deltaPhi;
    //! Yaw step (may differ from the sensor's)
    PointCoordinateType deltaTheta;
    //! Buffer width
    unsigned width;
    //! Buffer height
    unsigned height;

    //! Clears the buffer
    void clear();

    //! Applies a mean filter to fill small holes (= lack of information) of the
    //! depth map.
    /**	The depth buffer must have been created before (see
    GroundBasedLidarSensor::computeDepthBuffer). \return a negative value if an
    error occurs, 0 otherwise
    **/
    int fillHoles();
};

#endif  // ECV_DEPTH_BUFFER_HEADER
