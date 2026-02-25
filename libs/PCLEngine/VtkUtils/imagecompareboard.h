// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QWidget>

#include "imagecompareboard.h"
#include "qPCL.h"

namespace Widgets {

class ImageCompareBoardPrivate;
class QPCL_ENGINE_LIB_API ImageCompareBoard : public QWidget {
    Q_OBJECT
public:
    explicit ImageCompareBoard(QWidget* parent = 0);
    ~ImageCompareBoard();

    void setOriginalImage(const QImage& img);
    QImage originalImage() const;

    void setComparedImage(const QImage& img);
    QImage comparedImage() const;

private:
    ImageCompareBoardPrivate* d_ptr;
    Q_DISABLE_COPY(ImageCompareBoard)
};

}  // namespace Widgets
