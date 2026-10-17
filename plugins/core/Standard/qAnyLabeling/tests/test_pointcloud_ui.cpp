// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <QApplication>
#include <QFile>
#include <QImage>
#include <QOpenGLWidget>
#include <QTemporaryDir>
#include <QTimer>
#include <QtEndian>
#include <cstdio>
#include <cstring>

#include "PointCloudWindow.h"

int main(int argc, char** argv) {
    QApplication app(argc, argv);
    QTemporaryDir temporary;
    if (!temporary.isValid()) return 1;
    QByteArray data;
    for (int i = 0; i < 100; ++i) {
        for (float value : {float(i % 10) - 5, float(i / 10) - 5, 0.0f, 0.5f}) {
            quint32 bits;
            std::memcpy(&bits, &value, 4);
            char raw[4];
            qToLittleEndian<quint32>(bits, reinterpret_cast<uchar*>(raw));
            data.append(raw, 4);
        }
    }
    const QString path = temporary.path() + QStringLiteral("/sample.bin");
    QFile file(path);
    if (!file.open(QIODevice::WriteOnly) || file.write(data) != data.size())
        return 1;
    file.close();
    PointCloudWindow window;
    window.show();
    window.openFile(path);
    QTimer::singleShot(250, &app, &QCoreApplication::quit);
    app.exec();
    auto* view = window.findChild<QOpenGLWidget*>();
    if (!view) return 1;
    const QImage image = view->grabFramebuffer();
    if (image.isNull()) {
        std::fprintf(stderr, "OpenGL framebuffer is empty\n");
        return 1;
    }
    int changed = 0;
    const QRgb background = image.pixel(0, 0);
    for (int y = 0; y < image.height(); ++y)
        for (int x = 0; x < image.width(); ++x)
            changed += image.pixel(x, y) != background;
    if (changed < 30) {
        std::fprintf(stderr, "Point cloud did not render: %d changed pixels\n",
                     changed);
        return 1;
    }
    return 0;
}
