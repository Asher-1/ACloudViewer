// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <QCoreApplication>
#include <array>
#include <cstdio>

#include "AnnotationSemantic.h"

int main(int argc, char** argv) {
    QCoreApplication app(argc, argv);
    constexpr int width = 8, height = 8, stride = 10;
    std::array<uint8_t, height * stride> pixels{};
    for (int y = 0; y < height; ++y) {
        pixels[y * stride + width] = 255;
        pixels[y * stride + width + 1] = 255;
    }
    for (int y = 1; y <= 4; ++y)
        for (int x = 1; x <= 4; ++x)
            if (x == 1 || x == 4 || y == 1 || y == 4)
                pixels[y * stride + x] = 1;
    pixels[6 * stride + 6] = 2;

    const QStringList names{QStringLiteral("road"), QStringLiteral("ring"),
                            QStringLiteral("dot")};
    const SemanticConversion converted = convertSemanticClassMap(
            pixels.data(), width, height, stride, names);
    if (!converted.error.isEmpty() || converted.shapes.size() != 3) {
        std::fprintf(stderr, "Semantic conversion failed\n");
        return 1;
    }
    for (int id = 0; id < 3; ++id) {
        const AnnotationShape& shape = converted.shapes[id];
        if (shape.label != names[id] ||
            shape.extra.value(QStringLiteral("class_id")).toInt(-1) != id ||
            shape.type != QStringLiteral("polygon") ||
            shape.points.size() < 3 ||
            shape.mask.size() != QSize(width, height)) {
            std::fprintf(stderr, "Semantic class metadata failed\n");
            return 1;
        }
        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x)
                if ((qAlpha(shape.mask.pixel(x, y)) > 0) !=
                    (pixels[y * stride + x] == id)) {
                    std::fprintf(stderr, "Semantic mask pixel mismatch\n");
                    return 1;
                }
    }
    if (converted.shapes[1].mask.pixel(2, 2) != 0 ||
        qAlpha(converted.shapes[0].mask.pixel(2, 2)) == 0 ||
        converted.shapes[2].points.size() != 4 ||
        converted.shapes[2].points[0] != QPointF(6, 6)) {
        std::fprintf(stderr, "Hole or one-pixel class was lost\n");
        return 1;
    }
    if (convertSemanticClassMap(pixels.data(), width, height, width - 1, names)
                .error.isEmpty() ||
        convertSemanticClassMap(pixels.data(), width, height, stride,
                                names.mid(0, 2))
                .error.isEmpty()) {
        std::fprintf(stderr, "Invalid semantic maps were accepted\n");
        return 1;
    }
    return 0;
}
