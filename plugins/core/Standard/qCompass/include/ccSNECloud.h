// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <ccMeasurement.h>
#include <ecvPointCloud.h>

#include <QStringList>

/*
Class for representing/drawing lineations measured with qCompass.
*/
class ccSNECloud : public ccPointCloud, public ccMeasurement {
public:
    ccSNECloud();
    ccSNECloud(ccPointCloud* obj);
    virtual ~ccSNECloud();

    void updateMetadata();

    static bool isSNECloud(ccHObject* obj);

protected:
    virtual void drawMeOnly(CC_DRAW_CONTEXT& context) override;

private:
    void removeNormalActors();
    QStringList m_normalViewIds;
};
