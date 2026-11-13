// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "OpenPCDetDbExport.h"

#include <CVLog.h>
#include <ecvPointCloud.h>
#include <ecvPolyline.h>

#include <QDebug>
#include <QFileInfo>

ccHObject* BuildDetectionGroup(const OpenPCDetRunResult& result,
                               const QString& modelTag) {
    if (result.detections.isEmpty()) {
        CVLog::PrintDebug(
                QStringLiteral("[OpenPCDet] BuildDetectionGroup: empty "
                               "detections"));
        return nullptr;
    }

    const QString groupName =
            QStringLiteral("OpenPCDet detections (%1)").arg(modelTag);
    auto* group = new ccHObject(groupName);
    group->setMetaData(QStringLiteral("OpenPCDet"), true);
    group->setMetaData(QStringLiteral("OpenPCDet/Model"),
                       QFileInfo(result.modelPath).fileName());
    group->setMetaData(QStringLiteral("OpenPCDet/Count"),
                       static_cast<qlonglong>(result.detections.size()));
    group->setMetaData(QStringLiteral("Runtime (ms)"), result.runtimeMs);
    if (!result.resolvedDevice.isEmpty()) {
        group->setMetaData(QStringLiteral("Device"), result.resolvedDevice);
    }
    if (result.fromScene) {
        group->setMetaData(QStringLiteral("OpenPCDet/Scene"),
                           QFileInfo(result.sceneDir).fileName());
        group->setMetaData(QStringLiteral("OpenPCDet/Cameras"),
                           static_cast<qlonglong>(result.numCameras));
    }
    if (!result.sourceLabel.isEmpty()) {
        group->setMetaData(QStringLiteral("OpenPCDet/Source"),
                           result.sourceLabel);
    }
    if (result.inputInLiDARFrame) {
        // Traceability only: the geometry below is already back-projected
        // into the displayed cloud's local frame, so these records must not
        // be re-applied for display.
        group->setMetaData(QStringLiteral("OpenPCDet/GlobalScale"),
                           result.globalScale);
        group->setMetaData(QStringLiteral("OpenPCDet/GlobalShift"),
                           QStringLiteral("[%1,%2,%3]")
                                   .arg(result.globalShiftX, 0, 'f', 3)
                                   .arg(result.globalShiftY, 0, 'f', 3)
                                   .arg(result.globalShiftZ, 0, 'f', 3));
    }

    const std::vector<int> wire = OpenPCDetHelpers::boxWireIndices();
    for (int i = 0; i < result.detections.size(); ++i) {
        const OpenPCDetDetection& d = result.detections[i];
        float corners[24];
        OpenPCDetHelpers::boxCorners(d, corners);
        if (result.inputInLiDARFrame) {
            // Back-project the global-frame box into the displayed cloud's
            // local frame: Plocal = (Pglobal + shift) * scale (ecvShifted
            // Object convention, inverse of the worker's lift). Uniform
            // scale, so sizes scale with the positions; heading is invariant.
            for (int c = 0; c < 8; ++c) {
                corners[3 * c + 0] = static_cast<float>(
                        (corners[3 * c + 0] + result.globalShiftX) *
                        result.globalScale);
                corners[3 * c + 1] = static_cast<float>(
                        (corners[3 * c + 1] + result.globalShiftY) *
                        result.globalScale);
                corners[3 * c + 2] = static_cast<float>(
                        (corners[3 * c + 2] + result.globalShiftZ) *
                        result.globalScale);
            }
        }
        const OpenPCDetHelpers::OpenPCDetClassColor cls =
                OpenPCDetHelpers::classColor(d.label, result.numClass);
        const ecvColor::Rgb boxColor(cls.r, cls.g, cls.b);

        ccPointCloud* vertices = new ccPointCloud(QStringLiteral("vertices"));
        if (!vertices->reserve(static_cast<unsigned>(wire.size()))) {
            delete vertices;
            continue;
        }
        for (int idx : wire) {
            vertices->addPoint(CCVector3(
                    static_cast<PointCoordinateType>(corners[3 * idx + 0]),
                    static_cast<PointCoordinateType>(corners[3 * idx + 1]),
                    static_cast<PointCoordinateType>(corners[3 * idx + 2])));
        }

        auto* box = new ccPolyline(vertices);
        box->addChild(vertices);
        if (!box->reserve(static_cast<unsigned>(wire.size()))) {
            delete box;
            continue;
        }
        box->addPointIndex(0, static_cast<unsigned>(wire.size()));
        box->setClosed(false);
        // setColor alone keeps the default colorsShown()==false, which the
        // display pipeline resolves to the context fallback color; the
        // per-class color requires the paired showColors(true).
        box->setColor(boxColor);
        box->showColors(true);
        box->setWidth(2);
        box->setName(QStringLiteral("%1 %2")
                             .arg(d.className)
                             .arg(d.score, 0, 'f', 2));
        box->setMetaData(QStringLiteral("OpenPCDet/Label"),
                         static_cast<qlonglong>(d.label));
        box->setMetaData(QStringLiteral("OpenPCDet/Score"),
                         static_cast<double>(d.score));
        box->setMetaData(QStringLiteral("OpenPCDet/Size"),
                         QStringLiteral("[%1,%2,%3]")
                                 .arg(d.dx, 0, 'f', 3)
                                 .arg(d.dy, 0, 'f', 3)
                                 .arg(d.dz, 0, 'f', 3));
        box->setMetaData(QStringLiteral("OpenPCDet/Heading"),
                         static_cast<double>(d.heading));
        group->addChild(box);
    }
    CVLog::PrintDebug(QStringLiteral("[OpenPCDet] BuildDetectionGroup: %1 "
                                     "box(es) for %2")
                              .arg(result.detections.size())
                              .arg(modelTag));
    return group;
}

ccPointCloud* BuildAnnotatedCloud(ccPointCloud* source,
                                  const OpenPCDetRunResult& result,
                                  const QString& modelTag,
                                  unsigned* tintedOut,
                                  QString* error) {
    if (!source || result.detections.isEmpty()) {
        if (error) *error = QStringLiteral("empty source or detections");
        return nullptr;
    }
    ccPointCloud* annotated = source->cloneThis();
    if (!annotated) {
        if (error) *error = QStringLiteral("clone failed");
        return nullptr;
    }
    // The tint below rides the RGB table; a source with a displayed scalar
    // field (e.g. the sample cloud's intensity) would keep SF rendering
    // ahead of colors and hide the tint entirely.
    annotated->showSF(false);
    if (!annotated->hasColors() && !annotated->resizeTheRGBTable(true)) {
        if (error) *error = QStringLiteral("RGB allocation failed");
        delete annotated;
        return nullptr;
    }

    unsigned tinted = 0;
    for (const OpenPCDetDetection& d0 : result.detections) {
        OpenPCDetDetection d = d0;
        if (result.inputInLiDARFrame) {
            // The source cloud stores local coordinates; compare and tint in
            // that frame (Plocal = (Pglobal + shift) * scale).
            d.x = static_cast<float>((d.x + result.globalShiftX) *
                                     result.globalScale);
            d.y = static_cast<float>((d.y + result.globalShiftY) *
                                     result.globalScale);
            d.z = static_cast<float>((d.z + result.globalShiftZ) *
                                     result.globalScale);
        }
        const float reachX = 0.5f * (d.dx + d.dy);
        const float reachY = reachX;
        const OpenPCDetHelpers::OpenPCDetClassColor cls =
                OpenPCDetHelpers::classColor(d.label, result.numClass);
        const ecvColor::Rgb tint(cls.r, cls.g, cls.b);
        for (unsigned p = 0; p < annotated->size(); ++p) {
            const CCVector3* pt = annotated->getPoint(p);
            if (std::abs(pt->x - d.x) > reachX ||
                std::abs(pt->y - d.y) > reachY) {
                continue;
            }
            if (OpenPCDetHelpers::pointInBox(d, pt->x, pt->y, pt->z)) {
                annotated->setPointColor(p, tint);
                ++tinted;
            }
        }
    }
    if (tintedOut) *tintedOut = tinted;
    CVLog::PrintDebug(QStringLiteral("[OpenPCDet] BuildAnnotatedCloud: %1 of "
                                     "%2 point(s) tinted (%3)")
                              .arg(tinted)
                              .arg(annotated->size())
                              .arg(modelTag));

    annotated->setName(
            QStringLiteral("OpenPCDet annotated (%1)").arg(modelTag));
    annotated->setMetaData(QStringLiteral("OpenPCDet"), true);
    annotated->setMetaData(QStringLiteral("OpenPCDet/Model"),
                           QFileInfo(result.modelPath).fileName());
    annotated->setMetaData(QStringLiteral("OpenPCDet/Count"),
                           static_cast<qlonglong>(result.detections.size()));
    annotated->setMetaData(QStringLiteral("OpenPCDet/TintedPoints"),
                           static_cast<qlonglong>(tinted));
    annotated->setMetaData(QStringLiteral("Runtime (ms)"), result.runtimeMs);
    annotated->showColors(true);
    return annotated;
}
