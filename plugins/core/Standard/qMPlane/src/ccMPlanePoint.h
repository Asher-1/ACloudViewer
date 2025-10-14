// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef CC_MPLANE_POINT
#define CC_MPLANE_POINT

// ECV_DB_LIB
#include "ecv2DLabel.h"

class ccMPlanePoint {
public:
    explicit ccMPlanePoint(cc2DLabel* label)
        : m_label(label), m_distance(0.0) {}

    unsigned int getIndex() const;
    const CCVector3& getCoordinates() const;
    cc2DLabel* getLabel();
    QString getName() const;
    void setName(const QString& newName);
    float getDistance() const;
    void setDistance(float);

private:
    cc2DLabel* m_label;
    float m_distance;
};

#endif