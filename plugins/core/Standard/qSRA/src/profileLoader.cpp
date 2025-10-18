// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "profileLoader.h"

// ECV_DB_LIB
#include <ecvPointCloud.h>
#include <ecvPolyline.h>

// ECV_PLUGINS
#include <ecvMainAppInterface.h>

// Qt
#include <QFile>
#include <QFileInfo>
#include <QTextStream>

ccPolyline* ProfileLoader::Load(QString filename,
                                CCVector3& origin,
                                ecvMainAppInterface* app /*=0*/) {
    // load profile as a polyline
    QFile file(filename);
    assert(file.exists());
    if (!file.open(QFile::ReadOnly | QFile::Text)) {
        if (app)
            app->dispToConsole(QString("Failed to open file for reading! Check "
                                       "access rights"),
                               ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
        return 0;
    }

    QTextStream stream(&file);

    ccPolyline* polyline = 0;

    bool error = false;
    for (unsigned n = 0; n < 1; ++n)  // fake loop for easy break ;)
    {
        // read origin
        {
            QString headerLine = stream.readLine();
            if (headerLine.isEmpty() || !headerLine.startsWith("X")) {
                if (app)
                    app->dispToConsole(
                            QString("Malformed file (origin header expected on "
                                    "first line)"),
                            ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
                error = true;
                break;
            }
            QString centerLine = stream.readLine();
            {
                QStringList tokens = centerLine.split(QRegExp("\\s+"),
                                                      QString::SkipEmptyParts);
                bool validLine = false;
                if (tokens.size() == 3) {
                    bool ok[3] = {false, false, false};
                    origin.x = static_cast<PointCoordinateType>(
                            tokens[0].toDouble(ok + 0));
                    origin.y = static_cast<PointCoordinateType>(
                            tokens[1].toDouble(ok + 1));
                    origin.z = static_cast<PointCoordinateType>(
                            tokens[2].toDouble(ok + 2));
                    validLine = ok[0] && ok[1] && ok[2];
                }
                if (!validLine) {
                    if (app)
                        app->dispToConsole(
                                QString("Malformed file (origin coordinates "
                                        "expected on second line)"),
                                ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
                    error = true;
                    break;
                }
            }
        }

        // read elevations
        {
            QString headerLine = stream.readLine();
            if (headerLine.isEmpty() || !headerLine.startsWith("R")) {
                if (app)
                    app->dispToConsole(
                            QString("Malformed file (radii/heights header "
                                    "expected on third line)"),
                            ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
                error = true;
                break;
            }

            QString line = stream.readLine();
            std::vector<CCVector2d> points;
            while (!line.isEmpty()) {
                QStringList tokens =
                        line.split(QRegExp("\\s+"), QString::SkipEmptyParts);
                if (tokens.size() < 2) {
                    if (app)
                        app->dispToConsole(
                                QString("Malformed file (radius/height couple "
                                        "expected from the 4th line and "
                                        "afterwards)"),
                                ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
                    error = true;
                    break;
                }

                CCVector2d P;
                P.x = tokens[0].toDouble();  // radius
                P.y = tokens[1].toDouble();  // height

                try {
                    points.push_back(P);
                } catch (const std::bad_alloc&) {
                    // not enough memory
                    if (app)
                        app->dispToConsole(
                                QString("Not enough memory!"),
                                ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
                    error = true;
                    break;
                }

                line = stream.readLine();
            }

            // convert 2D points to polyline
            {
                unsigned count = static_cast<unsigned>(points.size());
                if (count > 1) {
                    ccPointCloud* vertices = new ccPointCloud("vertices");
                    polyline = new ccPolyline(vertices);
                    polyline->addChild(vertices);
                    if (!vertices->reserve(count) ||
                        !polyline->reserve(count)) {
                        // not enough memory
                        if (app)
                            app->dispToConsole(
                                    QString("Not enough memory!"),
                                    ecvMainAppInterface::ERR_CONSOLE_MESSAGE);
                        error = true;
                        break;
                    }

                    // add vertices
                    {
                        for (unsigned i = 0; i < count; ++i) {
                            vertices->addPoint(
                                    CCVector3(static_cast<PointCoordinateType>(
                                                      points[i].x),
                                              static_cast<PointCoordinateType>(
                                                      points[i].y),
                                              0));
                        }
                    }

                    // add segments
                    polyline->addPointIndex(0, count);
                    polyline->setClosed(false);  // just to be sure
                    polyline->set2DMode(true);

                    // add to DB
                    polyline->setName(QFileInfo(filename).baseName());
                    polyline->setColor(ecvColor::green);
                    polyline->showColors(true);
                    polyline->setEnabled(true);
                    polyline->setLocked(
                            true);  // as we have applied a purely visual
                                    // transformation, we can't let the user
                                    // rotate it!!!
                    vertices->setEnabled(false);
                } else {
                    if (app)
                        app->dispToConsole(
                                QString("Not enough points in profile?!"),
                                ecvMainAppInterface::ERR_CONSOLE_MESSAGE);

                    error = true;
                    break;
                }
            }
        }
    }

    file.close();

    if (error) {
        delete polyline;
        polyline = 0;
    }

    return polyline;
}
