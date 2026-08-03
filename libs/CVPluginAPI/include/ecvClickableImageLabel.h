// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QLabel>

#include "CVPluginAPI.h"

/** Thumbnail label that opens a scrollable full-size preview on click. */
class CVPLUGIN_LIB_API ecvClickableImageLabel : public QLabel {
    Q_OBJECT

public:
    explicit ecvClickableImageLabel(QWidget* parent = nullptr);

    void setPreviewImage(const QImage& image, int thumbSize = 96);
    /** Scale to \p displaySize while keeping the full-resolution click target.
     */
    void setPreviewImage(const QImage& image, const QSize& displaySize);
    void setPreviewPixmap(const QPixmap& pixmap, int thumbSize = 96);
    void clearPreview();

    const QImage& fullImage() const { return m_fullImage; }

    static void showEnlargedImage(QWidget* parent,
                                  const QImage& image,
                                  const QString& title = QString());

    /** Row widget: hint text + arrow (→) + thumbnail. \p label is reparented.
     */
    static QWidget* wrapWithTapToPreviewHint(ecvClickableImageLabel* label,
                                             QWidget* parent = nullptr);

protected:
    void mousePressEvent(QMouseEvent* event) override;
    void enterEvent(QEvent* event) override;
    void leaveEvent(QEvent* event) override;

private:
    void updateInteractiveState();

    QImage m_fullImage;
    QString m_dialogTitle;
};
