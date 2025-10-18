// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// Local
#include "distanceMapGenerationTool.h"

// Qt
#include <QSharedPointer>
#include <QString>

class ccPolyline;
class ecvMainAppInterface;

//! DXF profiles (of a surface of revolution) exporter
/** Requires DXF lib support!
 **/
class DxfProfilesExporter {
public:
    //! Returns whether DXF support is enabled or not
    static bool IsEnabled();

    struct Parameters {
        QStringList profileTitles;
        QString legendTheoProfileTitle;
        QString legendRealProfileTitle;
        QString scaledDevUnits;
        double devLabelMultCoef;
        double devMagnifyCoef;
        int precision;
    };

    //! Exports vertical profiles (of a surface of revolution's map) as a DXF
    //! file
    static bool SaveVerticalProfiles(
            const QSharedPointer<DistanceMapGenerationTool::Map>& map,
            ccPolyline* profile,
            QString filename,
            unsigned angularStepCount,
            double heightStep,
            double heightShift,
            const Parameters& params,
            ecvMainAppInterface* app = 0);

    //! Exports horizontal profiles (of a surface of revolution's map) as a DXF
    //! file
    static bool SaveHorizontalProfiles(
            const QSharedPointer<DistanceMapGenerationTool::Map>& map,
            ccPolyline* profile,
            QString filename,
            unsigned heightStepCount,
            double heightShift,
            double angularStep_rad,
            double radToUnitConvFactor,
            QString angleUnit,
            const Parameters& params,
            ecvMainAppInterface* app = 0);
};
