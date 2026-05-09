// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Pure C++ Python syntax highlighter (no Pygments dependency).
// ParaView uses Pygments via vtkPythonInterpreter; we implement a lightweight
// QSyntaxHighlighter covering the same token categories.

#pragma once

#include <QRegularExpression>
#include <QSyntaxHighlighter>

class ecvPythonSyntaxHighlighter : public QSyntaxHighlighter {
    Q_OBJECT

public:
    explicit ecvPythonSyntaxHighlighter(QTextDocument* parent = nullptr);

protected:
    void highlightBlock(const QString& text) override;

private:
    struct Rule {
        QRegularExpression pattern;
        QTextCharFormat format;
    };
    QVector<Rule> m_rules;

    QTextCharFormat m_multiLineStringFormat;
    QRegularExpression m_triSingleStart;
    QRegularExpression m_triDoubleStart;
    QRegularExpression m_triSingleEnd;
    QRegularExpression m_triDoubleEnd;
};
