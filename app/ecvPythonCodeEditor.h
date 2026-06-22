// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Code editor with line numbers, current-line highlight, and auto-indent.
// Modeled after ParaView's pqPythonLineNumberArea + pqPythonTextArea.

#pragma once

#include <QPlainTextEdit>

class QPaintEvent;
class QResizeEvent;

class ecvPythonCodeEditor : public QPlainTextEdit {
    Q_OBJECT

public:
    explicit ecvPythonCodeEditor(QWidget* parent = nullptr);

    void lineNumberAreaPaintEvent(QPaintEvent* event);
    int lineNumberAreaWidth() const;

protected:
    void resizeEvent(QResizeEvent* event) override;
    void keyPressEvent(QKeyEvent* event) override;

private slots:
    void updateLineNumberAreaWidth(int newBlockCount);
    void highlightCurrentLine();
    void updateLineNumberArea(const QRect& rect, int dy);

private:
    QWidget* m_lineNumberArea;
};

class LineNumberArea : public QWidget {
public:
    explicit LineNumberArea(ecvPythonCodeEditor* editor)
        : QWidget(editor), m_editor(editor) {}

    QSize sizeHint() const override {
        return QSize(m_editor->lineNumberAreaWidth(), 0);
    }

protected:
    void paintEvent(QPaintEvent* event) override {
        m_editor->lineNumberAreaPaintEvent(event);
    }

private:
    ecvPythonCodeEditor* m_editor;
};
