// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ccCompassCommands.h"

#include "ccFitPlane.h"
#include "ccGeoObject.h"
#include "ccLineation.h"
#include "ccSNECloud.h"
#include "ccThickness.h"
#include "ccTrace.h"

#include <ecvNormalVectors.h>
#include <ecvOctree.h>
#include <ecvPlane.h>
#include <ecvPointCloud.h>
#include <ecvProgressDialog.h>
#include <ecvScalarField.h>

#include <DgmOctree.h>

#include <QFile>
#include <QFileInfo>
#include <QObject>
#include <QTextStream>
#include <QXmlStreamWriter>

static const char COMMAND_COMPASS_EXPORT[] = "COMPASS_EXPORT";
static const char COMMAND_CE_FORMAT[] = "FORMAT";
static const char COMMAND_CE_OUTPUT[] = "OUTPUT";

namespace {

int writePlanes(ccHObject* object, QTextStream* out,
                const QString& parentName = QString()) {
    QString name = parentName.isEmpty()
                           ? object->getName()
                           : QStringLiteral("%1.%2").arg(parentName,
                                                         object->getName());
    int n = 0;
    if (ccFitPlane::isFitPlane(object)) {
        *out << name << ","
             << object->getMetaData("Strike").toString() << ","
             << object->getMetaData("Dip").toString() << ","
             << object->getMetaData("DipDir").toString() << ","
             << object->getMetaData("Cx").toString() << ","
             << object->getMetaData("Cy").toString() << ","
             << object->getMetaData("Cz").toString() << ","
             << object->getMetaData("Nx").toString() << ","
             << object->getMetaData("Ny").toString() << ","
             << object->getMetaData("Nz").toString() << "\n";
        ++n;
    }
    for (unsigned i = 0; i < object->getChildrenNumber(); ++i) {
        n += writePlanes(object->getChild(i), out, name);
    }
    return n;
}

int writeLineations(ccHObject* object, QTextStream* out,
                    const QString& parentName = QString()) {
    QString name = parentName.isEmpty()
                           ? object->getName()
                           : QStringLiteral("%1.%2").arg(parentName,
                                                         object->getName());
    int n = 0;
    if (ccLineation::isLineation(object)) {
        *out << name << ","
             << object->getMetaData("Trend").toString() << ","
             << object->getMetaData("Plunge").toString() << ","
             << object->getMetaData("Cx").toString() << ","
             << object->getMetaData("Cy").toString() << ","
             << object->getMetaData("Cz").toString() << "\n";
        ++n;
    }
    for (unsigned i = 0; i < object->getChildrenNumber(); ++i) {
        n += writeLineations(object->getChild(i), out, name);
    }
    return n;
}

int writeTraces(ccHObject* object, QTextStream* out,
                const QString& parentName = QString()) {
    QString name = parentName.isEmpty()
                           ? object->getName()
                           : QStringLiteral("%1.%2").arg(parentName,
                                                         object->getName());
    int n = 0;
    if (ccTrace::isTrace(object)) {
        ccPolyline* poly = dynamic_cast<ccPolyline*>(object);
        if (poly && poly->getAssociatedCloud()) {
            auto* cloud = poly->getAssociatedCloud();
            for (unsigned p = 0; p < cloud->size(); ++p) {
                const CCVector3* pt = cloud->getPoint(p);
                *out << name << "," << pt->x << "," << pt->y << ","
                     << pt->z << "\n";
            }
            ++n;
        }
    }
    for (unsigned i = 0; i < object->getChildrenNumber(); ++i) {
        n += writeTraces(object->getChild(i), out, name);
    }
    return n;
}

}  // namespace

CommandCompassExport::CommandCompassExport()
        : ccCommandLineInterface::Command("Compass Export",
                                          COMMAND_COMPASS_EXPORT) {}

bool CommandCompassExport::process(ccCommandLineInterface& cmd) {
    cmd.print("[COMPASS_EXPORT]");

    QString format = "csv";
    QString outputFile;

    while (!cmd.arguments().empty()) {
        const QString& arg = cmd.arguments().front();
        if (ccCommandLineInterface::IsCommand(arg, COMMAND_CE_FORMAT)) {
            cmd.arguments().pop_front();
            if (cmd.arguments().empty())
                return cmd.error(QObject::tr("Missing value after \"-%1\"")
                                         .arg(COMMAND_CE_FORMAT));
            format = cmd.arguments().takeFirst().toLower();
        } else if (ccCommandLineInterface::IsCommand(arg, COMMAND_CE_OUTPUT)) {
            cmd.arguments().pop_front();
            if (cmd.arguments().empty())
                return cmd.error(QObject::tr("Missing value after \"-%1\"")
                                         .arg(COMMAND_CE_OUTPUT));
            outputFile = cmd.arguments().takeFirst();
        } else {
            break;
        }
    }

    if (outputFile.isEmpty()) {
        return cmd.error(
                QObject::tr("Missing output file (use \"-%1 <path>\")")
                        .arg(COMMAND_CE_OUTPUT));
    }

    if (cmd.clouds().empty() && cmd.meshes().empty()) {
        return cmd.error(QObject::tr(
                "No entity loaded. Load a project file containing Compass "
                "data first."));
    }

    if (format == "csv") {
        QFileInfo fi(outputFile);
        QString basePath = fi.absolutePath() + "/" + fi.completeBaseName();

        auto writeFile = [&](const QString& suffix, const QString& header,
                             auto writeFn) -> int {
            QFile file(basePath + suffix);
            if (!file.open(QIODevice::WriteOnly | QIODevice::Text))
                return 0;
            QTextStream stream(&file);
            stream << header << "\n";
            int count = 0;
            for (CLCloudDesc& desc : cmd.clouds())
                count += writeFn(desc.pc, &stream, QString());
            for (CLMeshDesc& desc : cmd.meshes())
                count += writeFn(desc.mesh, &stream, QString());
            return count;
        };

        int planes = writeFile(
                "_planes.csv",
                "Name,Strike,Dip,Dip_Dir,Cx,Cy,Cz,Nx,Ny,Nz",
                writePlanes);
        int lineations = writeFile(
                "_lineations.csv", "Name,Trend,Plunge,Cx,Cy,Cz",
                writeLineations);
        int traces = writeFile("_traces.csv", "Name,X,Y,Z", writeTraces);

        cmd.print(QObject::tr("[COMPASS_EXPORT] Exported %1 planes, %2 "
                              "lineations, %3 traces to '%4_*.csv'")
                          .arg(planes)
                          .arg(lineations)
                          .arg(traces)
                          .arg(basePath));
    } else if (format == "xml") {
        QFile file(outputFile);
        if (!file.open(QIODevice::WriteOnly)) {
            return cmd.error(
                    QObject::tr("Cannot open file '%1' for writing")
                            .arg(outputFile));
        }
        QXmlStreamWriter xml(&file);
        xml.setAutoFormatting(true);
        xml.writeStartDocument();
        xml.writeStartElement("CompassData");
        for (CLCloudDesc& desc : cmd.clouds()) {
            xml.writeStartElement("Cloud");
            xml.writeAttribute("name", desc.pc->getName());
            xml.writeEndElement();
        }
        xml.writeEndElement();
        xml.writeEndDocument();
        cmd.print(QObject::tr("[COMPASS_EXPORT] Exported XML to '%1'")
                          .arg(outputFile));
    } else {
        return cmd.error(
                QObject::tr("Unknown format '%1'. Use csv or xml.").arg(format));
    }

    return true;
}

// ═══════════════════════════════════════════════════════════════════════════
// COMPASS_IMPORT_FOL — Import foliations from scalar fields
// ═══════════════════════════════════════════════════════════════════════════

static const char COMMAND_COMPASS_IMPORT_FOL[] = "COMPASS_IMPORT_FOL";
static const char COMMAND_CIF_DIP[] = "DIP_SF";
static const char COMMAND_CIF_DIPDIR[] = "DIPDIR_SF";
static const char COMMAND_CIF_SIZE[] = "PLANE_SIZE";

CommandCompassImportFoliations::CommandCompassImportFoliations()
        : ccCommandLineInterface::Command("Compass Import Foliations",
                                          COMMAND_COMPASS_IMPORT_FOL) {}

bool CommandCompassImportFoliations::process(ccCommandLineInterface& cmd) {
    cmd.print("[COMPASS_IMPORT_FOL]");

    if (cmd.clouds().empty()) {
        return cmd.error(QObject::tr(
                "No point cloud loaded (use \"-O [filename]\" before \"-%1\")")
                                 .arg(COMMAND_COMPASS_IMPORT_FOL));
    }

    QString dipSfName = "Dip";
    QString dipDirSfName = "DipDir";
    float planeSize = 2.0f;

    while (!cmd.arguments().empty()) {
        const QString& arg = cmd.arguments().front();
        if (ccCommandLineInterface::IsCommand(arg, COMMAND_CIF_DIP)) {
            cmd.arguments().pop_front();
            if (cmd.arguments().empty())
                return cmd.error(QObject::tr("Missing value after \"-%1\"")
                                         .arg(COMMAND_CIF_DIP));
            dipSfName = cmd.arguments().takeFirst();
        } else if (ccCommandLineInterface::IsCommand(arg, COMMAND_CIF_DIPDIR)) {
            cmd.arguments().pop_front();
            if (cmd.arguments().empty())
                return cmd.error(QObject::tr("Missing value after \"-%1\"")
                                         .arg(COMMAND_CIF_DIPDIR));
            dipDirSfName = cmd.arguments().takeFirst();
        } else if (ccCommandLineInterface::IsCommand(arg, COMMAND_CIF_SIZE)) {
            cmd.arguments().pop_front();
            if (cmd.arguments().empty())
                return cmd.error(QObject::tr("Missing value after \"-%1\"")
                                         .arg(COMMAND_CIF_SIZE));
            planeSize = cmd.arguments().takeFirst().toFloat();
        } else {
            break;
        }
    }

    for (CLCloudDesc& desc : cmd.clouds()) {
        ccPointCloud* cloud = desc.pc;
        if (!cloud) continue;

        int dipIdx = cloud->getScalarFieldIndexByName(
                qPrintable(dipSfName));
        int dipDirIdx = cloud->getScalarFieldIndexByName(
                qPrintable(dipDirSfName));

        if (dipIdx < 0 || dipDirIdx < 0) {
            cmd.warning(
                    QObject::tr("[COMPASS_IMPORT_FOL] Cloud '%1': missing "
                                "scalar field '%2' or '%3', skipping")
                            .arg(cloud->getName(), dipSfName, dipDirSfName));
            continue;
        }

        int count = 0;
        for (unsigned p = 0; p < cloud->size(); ++p) {
            float dip = cloud->getScalarField(dipIdx)->at(p);
            float dipdir = cloud->getScalarField(dipDirIdx)->at(p);
            const CCVector3* Cd = cloud->getPoint(p);

            ccPlane* plane = new ccPlane(planeSize, planeSize, nullptr,
                                         QStringLiteral("%1/%2")
                                                 .arg(dip, 0, 'f', 1)
                                                 .arg(dipdir, 0, 'f', 1));

            CCVector3 N = ccNormalVectors::ConvertDipAndDipDirToNormal(
                    dip, dipdir, true);
            ccGLMatrix trans;
            trans.toIdentity();
            CCVector3 Z(0, 0, 1);
            ccGLMatrix rotation =
                    ccGLMatrix::FromToRotation(Z, N);
            trans = rotation;
            trans.setTranslation(*Cd);
            plane->applyGLTransformation_recursive(&trans);

            cloud->addChild(plane);
            ++count;
        }

        cmd.print(QObject::tr("[COMPASS_IMPORT_FOL] Created %1 planes for "
                              "cloud '%2'")
                          .arg(count)
                          .arg(cloud->getName()));

        if (cmd.autoSaveMode()) {
            desc.basename += QString("_FOL");
            QString errorStr = cmd.exportEntity(desc);
            if (!errorStr.isEmpty()) return cmd.error(errorStr);
        }
    }

    return true;
}

// ═══════════════════════════════════════════════════════════════════════════
// COMPASS_IMPORT_LIN — Import lineations from scalar fields
// ═══════════════════════════════════════════════════════════════════════════

static const char COMMAND_COMPASS_IMPORT_LIN[] = "COMPASS_IMPORT_LIN";
static const char COMMAND_CIL_TREND[] = "TREND_SF";
static const char COMMAND_CIL_PLUNGE[] = "PLUNGE_SF";
static const char COMMAND_CIL_LENGTH[] = "LENGTH";

CommandCompassImportLineations::CommandCompassImportLineations()
        : ccCommandLineInterface::Command("Compass Import Lineations",
                                          COMMAND_COMPASS_IMPORT_LIN) {}

bool CommandCompassImportLineations::process(ccCommandLineInterface& cmd) {
    cmd.print("[COMPASS_IMPORT_LIN]");

    if (cmd.clouds().empty()) {
        return cmd.error(QObject::tr(
                "No point cloud loaded (use \"-O [filename]\" before \"-%1\")")
                                 .arg(COMMAND_COMPASS_IMPORT_LIN));
    }

    QString trendSfName = "Trend";
    QString plungeSfName = "Plunge";
    float displayLength = 2.0f;

    while (!cmd.arguments().empty()) {
        const QString& arg = cmd.arguments().front();
        if (ccCommandLineInterface::IsCommand(arg, COMMAND_CIL_TREND)) {
            cmd.arguments().pop_front();
            if (cmd.arguments().empty())
                return cmd.error(QObject::tr("Missing value after \"-%1\"")
                                         .arg(COMMAND_CIL_TREND));
            trendSfName = cmd.arguments().takeFirst();
        } else if (ccCommandLineInterface::IsCommand(arg, COMMAND_CIL_PLUNGE)) {
            cmd.arguments().pop_front();
            if (cmd.arguments().empty())
                return cmd.error(QObject::tr("Missing value after \"-%1\"")
                                         .arg(COMMAND_CIL_PLUNGE));
            plungeSfName = cmd.arguments().takeFirst();
        } else if (ccCommandLineInterface::IsCommand(arg, COMMAND_CIL_LENGTH)) {
            cmd.arguments().pop_front();
            if (cmd.arguments().empty())
                return cmd.error(QObject::tr("Missing value after \"-%1\"")
                                         .arg(COMMAND_CIL_LENGTH));
            displayLength = cmd.arguments().takeFirst().toFloat();
        } else {
            break;
        }
    }

    for (CLCloudDesc& desc : cmd.clouds()) {
        ccPointCloud* cloud = desc.pc;
        if (!cloud) continue;

        int trendIdx = cloud->getScalarFieldIndexByName(
                qPrintable(trendSfName));
        int plungeIdx = cloud->getScalarFieldIndexByName(
                qPrintable(plungeSfName));

        if (trendIdx < 0 || plungeIdx < 0) {
            cmd.warning(
                    QObject::tr("[COMPASS_IMPORT_LIN] Cloud '%1': missing "
                                "scalar field '%2' or '%3', skipping")
                            .arg(cloud->getName(), trendSfName,
                                 plungeSfName));
            continue;
        }

        int count = 0;
        for (unsigned p = 0; p < cloud->size(); ++p) {
            float trend = cloud->getScalarField(trendIdx)->at(p);
            float plunge = cloud->getScalarField(plungeIdx)->at(p);
            const CCVector3* Cd = cloud->getPoint(p);

            float trendRad = static_cast<float>(
                    cloudViewer::DegreesToRadians(
                            static_cast<double>(trend)));
            float plungeRad = static_cast<float>(
                    cloudViewer::DegreesToRadians(
                            static_cast<double>(plunge)));

            CCVector3 dir(std::sin(trendRad) * std::cos(plungeRad),
                          std::cos(trendRad) * std::cos(plungeRad),
                          -std::sin(plungeRad));

            ccPointCloud* linePts = new ccPointCloud("lineation");
            linePts->reserve(2);
            linePts->addPoint(*Cd - dir * (displayLength * 0.5f));
            linePts->addPoint(*Cd + dir * (displayLength * 0.5f));

            ccPolyline* line = new ccPolyline(linePts);
            line->addPointIndex(0, 2);
            line->setClosed(false);
            line->setName(QStringLiteral("%1/%2")
                                  .arg(trend, 0, 'f', 1)
                                  .arg(plunge, 0, 'f', 1));
            line->addChild(linePts);
            cloud->addChild(line);
            ++count;
        }

        cmd.print(QObject::tr("[COMPASS_IMPORT_LIN] Created %1 lineations for "
                              "cloud '%2'")
                          .arg(count)
                          .arg(cloud->getName()));

        if (cmd.autoSaveMode()) {
            desc.basename += QString("_LIN");
            QString errorStr = cmd.exportEntity(desc);
            if (!errorStr.isEmpty()) return cmd.error(errorStr);
        }
    }

    return true;
}

// ═══════════════════════════════════════════════════════════════════════════
// COMPASS_REFIT — Recalculate fit planes for traces
// ═══════════════════════════════════════════════════════════════════════════

static const char COMMAND_COMPASS_REFIT[] = "COMPASS_REFIT";

CommandCompassRefit::CommandCompassRefit()
        : ccCommandLineInterface::Command("Compass Refit Planes",
                                          COMMAND_COMPASS_REFIT) {}

namespace {

int refitPlanesRecursive(ccHObject* object) {
    int count = 0;

    if (ccFitPlane::isFitPlane(object)) {
        ccHObject* parent = object->getParent();
        if (parent && ccTrace::isTrace(parent)) {
            ccTrace* trace = static_cast<ccTrace*>(parent);
            ccFitPlane* newPlane = trace->fitPlane();
            if (newPlane) {
                trace->addChild(newPlane);
                parent->getParent()->removeChild(object);
                ++count;
            }
        }
    }

    for (unsigned i = 0; i < object->getChildrenNumber(); ++i) {
        count += refitPlanesRecursive(object->getChild(i));
    }
    return count;
}

}  // namespace

bool CommandCompassRefit::process(ccCommandLineInterface& cmd) {
    cmd.print("[COMPASS_REFIT]");

    if (cmd.clouds().empty() && cmd.meshes().empty()) {
        return cmd.error(QObject::tr(
                "No entity loaded. Load a file with Compass data first."));
    }

    int total = 0;
    for (CLCloudDesc& desc : cmd.clouds()) {
        total += refitPlanesRecursive(desc.pc);
    }
    for (CLMeshDesc& desc : cmd.meshes()) {
        total += refitPlanesRecursive(desc.mesh);
    }

    cmd.print(QObject::tr("[COMPASS_REFIT] Recalculated %1 fit planes").arg(total));

    for (CLCloudDesc& desc : cmd.clouds()) {
        if (cmd.autoSaveMode()) {
            desc.basename += QString("_REFIT");
            QString errorStr = cmd.exportEntity(desc);
            if (!errorStr.isEmpty()) return cmd.error(errorStr);
        }
    }

    return true;
}

// ═══════════════════════════════════════════════════════════════════════════
// COMPASS_P21 — Estimate P21 fracture intensity
// ═══════════════════════════════════════════════════════════════════════════

static const char COMMAND_COMPASS_P21[] = "COMPASS_P21";
static const char COMMAND_CP21_RADIUS[] = "RADIUS";
static const char COMMAND_CP21_SUBSAMPLE[] = "SUBSAMPLE";
static const char COMMAND_CP21_OUTPUT[] = "OUTPUT";

CommandCompassP21::CommandCompassP21()
        : ccCommandLineInterface::Command("Compass Estimate P21",
                                          COMMAND_COMPASS_P21) {}

bool CommandCompassP21::process(ccCommandLineInterface& cmd) {
    cmd.print("[COMPASS_P21]");

    if (cmd.clouds().empty()) {
        return cmd.error(QObject::tr(
                "No point cloud loaded (use \"-O [filename]\" before \"-%1\")")
                                 .arg(COMMAND_COMPASS_P21));
    }

    double searchRadius = 10.0;
    unsigned subsampleRate = 25;
    QString outputFile;

    while (!cmd.arguments().empty()) {
        const QString& arg = cmd.arguments().front();
        if (ccCommandLineInterface::IsCommand(arg, COMMAND_CP21_RADIUS)) {
            cmd.arguments().pop_front();
            if (cmd.arguments().empty())
                return cmd.error(QObject::tr("Missing value after \"-%1\"")
                                         .arg(COMMAND_CP21_RADIUS));
            searchRadius = cmd.arguments().takeFirst().toDouble();
        } else if (ccCommandLineInterface::IsCommand(arg, COMMAND_CP21_SUBSAMPLE)) {
            cmd.arguments().pop_front();
            if (cmd.arguments().empty())
                return cmd.error(QObject::tr("Missing value after \"-%1\"")
                                         .arg(COMMAND_CP21_SUBSAMPLE));
            subsampleRate = cmd.arguments().takeFirst().toUInt();
            if (subsampleRate < 1) subsampleRate = 1;
        } else if (ccCommandLineInterface::IsCommand(arg, COMMAND_CP21_OUTPUT)) {
            cmd.arguments().pop_front();
            if (cmd.arguments().empty())
                return cmd.error(QObject::tr("Missing value after \"-%1\"")
                                         .arg(COMMAND_CP21_OUTPUT));
            outputFile = cmd.arguments().takeFirst();
        } else {
            break;
        }
    }

    cmd.print(QObject::tr("[COMPASS_P21] Search radius = %1, subsample = %2")
                      .arg(searchRadius)
                      .arg(subsampleRate));

    for (CLCloudDesc& desc : cmd.clouds()) {
        ccPointCloud* cloud = desc.pc;
        if (!cloud) continue;

        ccPointCloud* traceCloud = new ccPointCloud();
        ccScalarField* weight = new ccScalarField("weight");
        traceCloud->addScalarField(weight);
        traceCloud->setCurrentScalarField(0);

        std::vector<ccPolyline*> lines;
        ccHObject::Container polyObjs;
        cloud->filterChildren(polyObjs, true, CV_TYPES::POLY_LINE);
        for (ccHObject* c : polyObjs) {
            if (ccTrace::isTrace(c)) {
                lines.push_back(static_cast<ccPolyline*>(c));
            }
        }

        if (lines.empty()) {
            cmd.warning(QObject::tr("[COMPASS_P21] Cloud '%1': no traces found, "
                                    "skipping")
                                .arg(cloud->getName()));
            delete traceCloud;
            continue;
        }

        for (ccPolyline* p : lines) {
            int sID = ccGeoObject::getGeoObjectRegion(p);
            double w = 1.0;
            if (sID == ccGeoObject::UPPER_BOUNDARY ||
                sID == ccGeoObject::LOWER_BOUNDARY) {
                w = 0.5;
            }
            traceCloud->reserve(traceCloud->size() + p->size());
            weight->reserve(traceCloud->size() + p->size());
            for (unsigned i = 0; i < p->size(); ++i) {
                traceCloud->addPoint(*p->getPoint(i));
                weight->addElement(static_cast<ScalarType>(w));
            }
        }
        traceCloud->computeOctree();

        ccPointCloud* outputCloud = new ccPointCloud("P21 Intensity");
        outputCloud->reserve(cloud->size() / subsampleRate);
        for (unsigned p = 0; p < cloud->size(); p += subsampleRate) {
            outputCloud->addPoint(*cloud->getPoint(p));
        }
        outputCloud->setGlobalScale(cloud->getGlobalScale());
        outputCloud->setGlobalShift(cloud->getGlobalShift());

        ccScalarField* P21 = new ccScalarField("P21");
        outputCloud->addScalarField(P21);
        P21->reserve(outputCloud->size());

        ccOctree::Shared traceOct = traceCloud->computeOctree();
        unsigned char traceLevel =
                traceOct->findBestLevelForAGivenNeighbourhoodSizeExtraction(
                        searchRadius);

        cloudViewer::DgmOctree::NeighboursSet region;
        for (unsigned p = 0; p < outputCloud->size(); ++p) {
            region.clear();
            traceOct->getPointsInSphericalNeighbourhood(
                    *outputCloud->getPoint(p), searchRadius, region,
                    traceLevel);
            float sum = 0;
            for (size_t i = 0; i < region.size(); ++i) {
                sum += weight->getValue(region[i].pointIndex);
            }
            P21->setValue(p, sum);
        }

        ccOctree::Shared outcropOct = outputCloud->computeOctree();
        for (unsigned p = 0; p < outputCloud->size(); ++p) {
            float sum = P21->getValue(p);
            if (sum > 0) {
                region.clear();
                int nOutcrop =
                        outcropOct->getPointsInSphericalNeighbourhood(
                                *outputCloud->getPoint(p), searchRadius,
                                region, traceLevel);
                if (nOutcrop > 0) {
                    P21->setValue(p, sum / (nOutcrop * subsampleRate));
                }
            }
        }

        P21->computeMinAndMax();
        outputCloud->setCurrentDisplayedScalarField(0);
        outputCloud->showSF(true);

        cloud->addChild(outputCloud);

        cmd.print(QObject::tr("[COMPASS_P21] Cloud '%1': P21 computed (%2 output "
                              "points, %3 traces)")
                          .arg(cloud->getName())
                          .arg(outputCloud->size())
                          .arg(lines.size()));

        delete traceCloud;

        if (cmd.autoSaveMode()) {
            desc.basename += QString("_P21");
            QString errorStr = cmd.exportEntity(desc);
            if (!errorStr.isEmpty()) return cmd.error(errorStr);
        }
    }

    return true;
}
