// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// GUI-path half of the GUI/headless closure check: runs the REAL
// MapAnythingWorker (the thread the dialog starts) with Settings filled the
// way MapAnythingDialog fills them - minConf and resize mode seeded from the
// catalog entry defaults, responsive streaming = the dialog checkbox - and
// writes the merged
// cloud in the exact byte layout of the headless CLOUD_PLY. Driven by
// tests/check_gui_headless_parity.py (not a ctest: needs GGUF + images).
//
//   worker_parity_harness MODEL DEVICE RESPONSIVE(0|1) OUT_PLY IMAGE...
//
// Exit: 0 ok, 1 worker failure, 2 usage, 77 model not in the catalog.

#include <QCoreApplication>
#include <QFile>
#include <QFileInfo>
#include <QStringList>
#include <cstdio>
#include <cstdlib>

#include "MapAnythingModelCatalog.h"
#include "MapAnythingWorker.h"

int main(int argc, char** argv) {
    QCoreApplication app(argc, argv);
    if (argc < 7) {
        std::fprintf(stderr,
                     "usage: %s MODEL DEVICE RESPONSIVE OUT_PLY IMAGE "
                     "IMAGE...\n",
                     argv[0]);
        return 2;
    }
    MapAnythingWorker::Settings s;
    s.modelPath = QString::fromLocal8Bit(argv[1]);
    s.device = QString::fromLocal8Bit(argv[2]);
    s.responsiveStreaming = std::atoi(argv[3]) != 0;
    const QString outPly = QString::fromLocal8Bit(argv[4]);
    for (int i = 5; i < argc; ++i)
        s.imagePaths << QString::fromLocal8Bit(argv[i]);
    s.addPointCloud = true;
    s.addDepthRasters = false;  // display-only; no effect on the cloud
    s.addPerViewClouds = false;

    MapAnythingModelEntry entry;
    if (!MapAnythingHelpers::findModelByFilename(
                QFileInfo(s.modelPath).fileName(), &entry)) {
        std::fprintf(stderr, "model not in the catalog: %s\n", argv[1]);
        return 77;
    }
    s.minConf = entry.defaultMinConf;  // MapAnythingDialog seeds these
    if (!entry.defaultResizeMode.isEmpty()) {
        s.resizeMode = entry.defaultResizeMode;
    }

    MapAnythingResult result;
    bool ok = false;
    MapAnythingWorker worker(s);
    // Direct connections: the slots run on the worker thread while it is
    // alive; wait() below orders every write before the main-thread read.
    QObject::connect(
            &worker, &MapAnythingWorker::resultReady, &worker,
            [&](const MapAnythingResult& r) { result = r; },
            Qt::DirectConnection);
    QObject::connect(
            &worker, &MapAnythingWorker::taskFinished, &worker,
            [&](bool success) { ok = success; }, Qt::DirectConnection);
    QObject::connect(
            &worker, &MapAnythingWorker::logMessage, &worker,
            [](const QString& m) {
                std::fprintf(stderr, "%s\n", m.toLocal8Bit().constData());
            },
            Qt::DirectConnection);
    worker.start();
    worker.wait();
    worker.releaseContextOnMainThread();
    if (!ok) return 1;

    QFile f(outPly);
    if (!f.open(QIODevice::WriteOnly | QIODevice::Truncate)) return 1;
    const QByteArray header = QStringLiteral(
                                      "ply\n"
                                      "format binary_little_endian 1.0\n"
                                      "element vertex %1\n"
                                      "property float x\n"
                                      "property float y\n"
                                      "property float z\n"
                                      "property uchar red\n"
                                      "property uchar green\n"
                                      "property uchar blue\n"
                                      "end_header\n")
                                      .arg(result.pointCount)
                                      .toUtf8();
    f.write(header);
    for (int k = 0; k < result.pointCount; ++k) {
        f.write(reinterpret_cast<const char*>(result.points.constData() +
                                              3 * k),
                3 * sizeof(float));
        f.write(reinterpret_cast<const char*>(result.colors.constData() +
                                              3 * k),
                3);
    }
    f.close();
    std::printf(
            "{\"points\":%d,\"views\":%d,\"architecture\":\"%s\","
            "\"e2e_ms\":%.1f,\"result\":%s}\n",
            result.pointCount, result.viewCount,
            result.architecture.toUtf8().constData(), result.e2eMs,
            result.resultJson.constData());
    return 0;
}
