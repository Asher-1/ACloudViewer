// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

/// @file imagecompareboard.h
/// @brief Side-by-side image comparison widget with synchronized scrolling.

#include <QWidget>

#include "imagecompareboard.h"
#include "qVTK.h"

namespace Widgets {

class ImageCompareBoardPrivate;
/// @class ImageCompareBoard
/// @brief Widget displaying original and compared images side by side with
/// synced scroll.
class QVTK_ENGINE_LIB_API ImageCompareBoard : public QWidget {
    Q_OBJECT
public:
    explicit ImageCompareBoard(QWidget* parent = 0);
    ~ImageCompareBoard();

    /// @param img Image to display as original (left side)
    void setOriginalImage(const QImage& img);
    /// @return Stored original image
    QImage originalImage() const;

    /// @param img Image to display as compared (right side)
    void setComparedImage(const QImage& img);
    /// @return Stored compared image
    QImage comparedImage() const;

private:
    ImageCompareBoardPrivate* d_ptr;
    Q_DISABLE_COPY(ImageCompareBoard)
};

}  // namespace Widgets
