// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "EditorSettings.h"
#include <QObject>
#include <QPlainTextEdit>

class QPaintEvent;
class QResizeEvent;
class QSize;
class QWidget;

class PythonHighlighter;

class LineNumberArea;

class CodeEditor final : public QPlainTextEdit
{
    Q_OBJECT

  public:
    explicit CodeEditor(EditorSettings *settings, QWidget *parent = nullptr);

    void lineNumberAreaPaintEvent(QPaintEvent *event);

    bool eventFilter(QObject *target, QEvent *event) override;
    int lineNumberAreaWidth();
    void newFile();
    bool loadFile(const QString &fileName);
    bool save();
    bool saveAs();
    bool saveFile(const QString &fileName);
    QString userFriendlyCurrentFile();
    QString currentFile() const
    {
        return m_curFile;
    }
    void comment();
    void uncomment();
    void indentMore();
    void indentLess();

  protected:
    void resizeEvent(QResizeEvent *event) override;
    void closeEvent(QCloseEvent *event) override;

  private Q_SLOTS:
    void updateLineNumberAreaWidth(int newBlockCount);
    void highlightCurrentLine();
    void updateLineNumberArea(const QRect &, int);
    void documentWasModified();
    void startAllHighlighting();

  private:
    void updateUsingSettings();

  protected:
    void keyPressEvent(QKeyEvent *e) override;

  private:
    bool maybeSave();
    void setCurrentFile(const QString &fileName);
    QString strippedName(const QString &fullFileName);
    void createPairedCharsSelection(int pos);
    int getSelectedLineCount();

    QString m_curFile;
    bool m_isUntitled{true};
    QWidget *m_lineNumberArea{nullptr};
    EditorSettings *m_settings{nullptr};
    PythonHighlighter *m_highlighter;
};

class LineNumberArea final : public QWidget
{
  public:
    explicit LineNumberArea(CodeEditor *editor) : QWidget(editor)
    {
        m_codeEditor = editor;
    }

    QSize sizeHint() const override
    {
        return {m_codeEditor->lineNumberAreaWidth(), 0};
    }

  protected:
    void paintEvent(QPaintEvent *event) override
    {
        m_codeEditor->lineNumberAreaPaintEvent(event);
    }

  private:
    CodeEditor *m_codeEditor;
};
