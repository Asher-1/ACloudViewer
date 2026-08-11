// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QImage>
#include <QPointF>
#include <QVector>

#include "LightGlueWorker.h"

QImage renderMatchVisualization(const QImage& image0,
                                const QImage& image1,
                                const QVector<QPointF>& keypoints0,
                                const QVector<QPointF>& keypoints1,
                                const QVector<LightGlueMatchResult>& matches,
                                int compositeWidth0,
                                int compositeHeight0,
                                int compositeWidth1,
                                int compositeHeight1);
