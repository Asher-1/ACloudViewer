// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QStringList>
#include <cstddef>
#include <cstdint>

#include "AnnotationCanvas.h"

struct SemanticConversion {
    QVector<AnnotationShape> shapes;
    QString error;
};

SemanticConversion convertSemanticClassMap(const uint8_t* data,
                                           int width,
                                           int height,
                                           size_t rowStride,
                                           const QStringList& classNames);
