// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <array>

#include "AnnotationSemantic.h"

TEST(AnnotationSemantic, ConvertsPaddedClassMapAndPreservesHoles) {
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
    const auto converted = convertSemanticClassMap(pixels.data(), width, height,
                                                   stride, names);
    ASSERT_TRUE(converted.error.isEmpty()) << converted.error.toStdString();
    ASSERT_EQ(converted.shapes.size(), 3);
    for (int id = 0; id < 3; ++id) {
        const auto& shape = converted.shapes[id];
        EXPECT_EQ(shape.label, names[id]);
        EXPECT_EQ(shape.extra.value(QStringLiteral("class_id")).toInt(-1), id);
        EXPECT_EQ(shape.type, QStringLiteral("polygon"));
        EXPECT_GE(shape.points.size(), 3);
        EXPECT_EQ(shape.mask.size(), QSize(width, height));
        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x)
                EXPECT_EQ(qAlpha(shape.mask.pixel(x, y)) > 0,
                          pixels[y * stride + x] == id)
                        << "class=" << id << " pixel=" << x << ',' << y;
    }
    EXPECT_EQ(converted.shapes[1].mask.pixel(2, 2), 0);
    EXPECT_NE(qAlpha(converted.shapes[0].mask.pixel(2, 2)), 0);
    EXPECT_EQ(converted.shapes[2].points.size(), 4);
    EXPECT_EQ(converted.shapes[2].points[0], QPointF(6, 6));
}

TEST(AnnotationSemantic, RejectsInvalidStrideAndClassNameCount) {
    constexpr int width = 8, height = 8, stride = 10;
    std::array<uint8_t, height * stride> pixels{};
    pixels[6 * stride + 6] = 2;
    QStringList names{QStringLiteral("road"), QStringLiteral("ring"),
                      QStringLiteral("dot")};
    EXPECT_FALSE(convertSemanticClassMap(pixels.data(), width, height,
                                         width - 1, names)
                         .error.isEmpty());
    EXPECT_FALSE(convertSemanticClassMap(pixels.data(), width, height, stride,
                                         names.mid(0, 2))
                         .error.isEmpty());
}
