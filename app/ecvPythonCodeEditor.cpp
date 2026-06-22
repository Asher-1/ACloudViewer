// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvPythonCodeEditor.h"

#include <QKeyEvent>
#include <QPainter>
#include <QTextBlock>

ecvPythonCodeEditor::ecvPythonCodeEditor(QWidget* parent)
    : QPlainTextEdit(parent) {
    m_lineNumberArea = new LineNumberArea(this);

    connect(this, &QPlainTextEdit::blockCountChanged, this,
            &ecvPythonCodeEditor::updateLineNumberAreaWidth);
    connect(this, &QPlainTextEdit::updateRequest, this,
            &ecvPythonCodeEditor::updateLineNumberArea);
    connect(this, &QPlainTextEdit::cursorPositionChanged, this,
            &ecvPythonCodeEditor::highlightCurrentLine);

    updateLineNumberAreaWidth(0);
    highlightCurrentLine();
}

int ecvPythonCodeEditor::lineNumberAreaWidth() const {
    int digits = 1;
    int max = qMax(1, blockCount());
    while (max >= 10) {
        max /= 10;
        ++digits;
    }
    int space = 6 + fontMetrics().horizontalAdvance(QLatin1Char('9')) * digits;
    return space;
}

void ecvPythonCodeEditor::updateLineNumberAreaWidth(int /*newBlockCount*/) {
    setViewportMargins(lineNumberAreaWidth(), 0, 0, 0);
}

void ecvPythonCodeEditor::updateLineNumberArea(const QRect& rect, int dy) {
    if (dy)
        m_lineNumberArea->scroll(0, dy);
    else
        m_lineNumberArea->update(0, rect.y(), m_lineNumberArea->width(),
                                 rect.height());

    if (rect.contains(viewport()->rect())) updateLineNumberAreaWidth(0);
}

void ecvPythonCodeEditor::resizeEvent(QResizeEvent* e) {
    QPlainTextEdit::resizeEvent(e);
    QRect cr = contentsRect();
    m_lineNumberArea->setGeometry(
            QRect(cr.left(), cr.top(), lineNumberAreaWidth(), cr.height()));
}

void ecvPythonCodeEditor::highlightCurrentLine() {
    QList<QTextEdit::ExtraSelection> extraSelections;

    if (!isReadOnly()) {
        QTextEdit::ExtraSelection selection;
        QColor lineColor = QColor(40, 40, 40);
        selection.format.setBackground(lineColor);
        selection.format.setProperty(QTextFormat::FullWidthSelection, true);
        selection.cursor = textCursor();
        selection.cursor.clearSelection();
        extraSelections.append(selection);
    }

    setExtraSelections(extraSelections);
}

void ecvPythonCodeEditor::lineNumberAreaPaintEvent(QPaintEvent* event) {
    QPainter painter(m_lineNumberArea);
    painter.fillRect(event->rect(), QColor(30, 30, 30));

    QTextBlock block = firstVisibleBlock();
    int blockNumber = block.blockNumber();
    int top = qRound(
            blockBoundingGeometry(block).translated(contentOffset()).top());
    int bottom = top + qRound(blockBoundingRect(block).height());

    while (block.isValid() && top <= event->rect().bottom()) {
        if (block.isVisible() && bottom >= event->rect().top()) {
            QString number = QString::number(blockNumber + 1);

            if (blockNumber == textCursor().blockNumber()) {
                painter.setPen(QColor(200, 200, 200));
            } else {
                painter.setPen(QColor(100, 100, 100));
            }

            painter.drawText(0, top, m_lineNumberArea->width() - 3,
                             fontMetrics().height(), Qt::AlignRight, number);
        }

        block = block.next();
        top = bottom;
        bottom = top + qRound(blockBoundingRect(block).height());
        ++blockNumber;
    }
}

void ecvPythonCodeEditor::keyPressEvent(QKeyEvent* event) {
    if (event->key() == Qt::Key_Return || event->key() == Qt::Key_Enter) {
        QTextCursor cursor = textCursor();
        QString currentLine = cursor.block().text();

        QString indent;
        for (const QChar& ch : currentLine) {
            if (ch == ' ' || ch == '\t')
                indent += ch;
            else
                break;
        }

        QString trimmed = currentLine.trimmed();
        if (trimmed.endsWith(':')) {
            indent += QStringLiteral("    ");
        }

        QPlainTextEdit::keyPressEvent(event);
        insertPlainText(indent);
        return;
    }

    if (event->key() == Qt::Key_Tab) {
        QTextCursor cursor = textCursor();
        if (cursor.hasSelection()) {
            QTextCursor start = cursor;
            start.setPosition(cursor.selectionStart());
            start.movePosition(QTextCursor::StartOfBlock);

            QTextCursor end = cursor;
            end.setPosition(cursor.selectionEnd());

            cursor.beginEditBlock();
            QTextBlock block = start.block();
            while (block.isValid() && block.position() <= end.position()) {
                QTextCursor blockCursor(block);
                blockCursor.movePosition(QTextCursor::StartOfBlock);
                blockCursor.insertText(QStringLiteral("    "));
                block = block.next();
            }
            cursor.endEditBlock();
            return;
        }
        insertPlainText(QStringLiteral("    "));
        return;
    }

    if (event->key() == Qt::Key_Backtab) {
        QTextCursor cursor = textCursor();
        if (cursor.hasSelection()) {
            QTextCursor start = cursor;
            start.setPosition(cursor.selectionStart());
            start.movePosition(QTextCursor::StartOfBlock);

            QTextCursor end = cursor;
            end.setPosition(cursor.selectionEnd());

            cursor.beginEditBlock();
            QTextBlock block = start.block();
            while (block.isValid() && block.position() <= end.position()) {
                QString text = block.text();
                int spaces = 0;
                for (int i = 0; i < qMin(4, text.length()); ++i) {
                    if (text[i] == ' ')
                        ++spaces;
                    else
                        break;
                }
                if (spaces > 0) {
                    QTextCursor blockCursor(block);
                    blockCursor.movePosition(QTextCursor::StartOfBlock);
                    blockCursor.movePosition(QTextCursor::Right,
                                             QTextCursor::KeepAnchor, spaces);
                    blockCursor.removeSelectedText();
                }
                block = block.next();
            }
            cursor.endEditBlock();
        } else {
            QTextCursor tc = textCursor();
            tc.movePosition(QTextCursor::StartOfBlock);
            QString lineText = tc.block().text();
            int spaces = 0;
            for (int i = 0; i < qMin(4, lineText.length()); ++i) {
                if (lineText[i] == ' ')
                    ++spaces;
                else
                    break;
            }
            if (spaces > 0) {
                tc.movePosition(QTextCursor::Right, QTextCursor::KeepAnchor,
                                spaces);
                tc.removeSelectedText();
            }
        }
        return;
    }

    QPlainTextEdit::keyPressEvent(event);
}
