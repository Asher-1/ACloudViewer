// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "HeightProfileFilter.h"

// qCC_db
#include <ecvPolyline.h>

// Qt
#include <QFile>

// Qt5/Qt6 Compatibility
#include <QtCompat.h>

HeightProfileFilter::HeightProfileFilter()
    : FileIOFilter({"_Height profile Filter",
                    21.0f,  // priority
                    QStringList(), "", QStringList(),
                    QStringList{"Height profile (*.csv)"}, Export}) {}

bool HeightProfileFilter::canSave(CV_CLASS_ENUM type,
                                  bool& multiple,
                                  bool& exclusive) const {
    if (type == CV_TYPES::POLY_LINE) {
        multiple = false;
        exclusive = true;
        return true;
    }
    return false;
}

CC_FILE_ERROR HeightProfileFilter::saveToFile(
        ccHObject* entity,
        const QString& filename,
        const SaveParameters& parameters) {
    if (!entity || filename.isEmpty()) {
        return CC_FERR_BAD_ARGUMENT;
    }

    // get the polyline
    if (!entity->isA(CV_TYPES::POLY_LINE)) {
        return CC_FERR_BAD_ENTITY_TYPE;
    }
    ccPolyline* poly = static_cast<ccPolyline*>(entity);
    unsigned vertCount = poly->size();
    if (vertCount == 0) {
        // invalid size
        CVLog::Warning(QString("[Height profile] Polyline '%1' is empty")
                               .arg(poly->getName()));
        return CC_FERR_NO_SAVE;
    }

    // open ASCII file for writing
    QFile file(filename);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        return CC_FERR_WRITING;
    }

    QTextStream outFile(&file);
    outFile.setRealNumberNotation(QTextStream::FixedNotation);
    outFile.setRealNumberPrecision(
            sizeof(PointCoordinateType) == 4 && !poly->isShifted() ? 8 : 12);
    outFile << "Curvilinear abscissa; Z" << QtCompat::endl;

    // curvilinear abscissa
    double s = 0;
    const CCVector3* lastP = 0;
    for (unsigned j = 0; j < vertCount; ++j) {
        const CCVector3* P = poly->getPoint(j);
        // update the curvilinear abscissa
        if (lastP) {
            s += (*P - *lastP).normd();
        }
        lastP = P;

        // convert to 'local' coordinate system
        CCVector3d Pg = poly->toGlobal3d(*P);
        outFile << s << "; " << Pg.z << QtCompat::endl;
    }

    file.close();

    return CC_FERR_NO_ERROR;
}
