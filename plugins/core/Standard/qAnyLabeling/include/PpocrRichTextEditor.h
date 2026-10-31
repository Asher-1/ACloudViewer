// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QFrame>
#include <QPointer>
#include <QTextCharFormat>

class QTextEdit;
class QToolButton;

class PpocrRichTextEditor final : public QFrame {
    Q_OBJECT
public:
    explicit PpocrRichTextEditor(const QString& content,
                                 QWidget* parent = nullptr,
                                 const QString& imagePath = {});
    QString content() const;

signals:
    void saveRequested(const QString& content);
    void cancelRequested();
    void geometryChanged();

protected:
    void resizeEvent(QResizeEvent* event) override;
    void showEvent(QShowEvent* event) override;
    bool eventFilter(QObject* watched, QEvent* event) override;

private:
    void mergeFormat(const QTextCharFormat& format);
    void applyHeading(int level);
    int selectedHeading() const;
    void updateToolbar();
    void updateHeadingStyle();
    void updateHeight();
    QTextEdit* m_text;
    QWidget* m_headingMenu = nullptr;
    QPointer<QWidget> m_scrollViewport;
    int m_imageHeight = 0;
    QToolButton *m_heading, *m_bold, *m_italic, *m_strike;
};
