// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QImage>
#include <QJsonObject>

struct MaskConversion {
    QImage image;
    QString error;
};

MaskConversion convertMask(const QJsonObject& document,
                           const QJsonObject& colorMap);
