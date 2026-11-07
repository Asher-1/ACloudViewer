// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QPointF>
#include <QVector>
#include <cstddef>
#include <cstdint>

// Extracts the largest external boundary from a borrowed row-strided binary
// mask.
QVector<QPointF> largestMaskPolygon(const uint8_t* data,
                                    int width,
                                    int height,
                                    size_t rowStride);

QVector<QVector<QPointF>> maskPolygons(const uint8_t* data,
                                       int width,
                                       int height,
                                       size_t rowStride);
