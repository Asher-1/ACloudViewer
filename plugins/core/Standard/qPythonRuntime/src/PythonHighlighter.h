// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QSyntaxHighlighter>

// Qt5/Qt6 Compatibility
#include <QtCompat.h>

class ColorScheme;

// Started from Qt Syntax Highlighter example and then ported
// https://wiki.python.org/moin/PyQt/Python%20syntax%20highlighting
class PythonHighlighter final : public QSyntaxHighlighter
{
  public:
    // For lack of a better name
    enum class CodeElement
    {
        Keyword = 0,
        Operator,
        Brace,
        Definition,
        String,
        DocString,
        Comment,
        Self,
        Numbers,
        End
    };

    static QString CodeElementName(PythonHighlighter::CodeElement e);

    void useColorScheme(const ColorScheme &colorScheme);

    explicit PythonHighlighter(QTextDocument *parent = nullptr);

  protected:
    void highlightBlock(const QString &text) override;

  private:
    struct HighlightingRule
    {
        CodeElement element = CodeElement::End;
        QtCompatRegExp pattern;
        QTextCharFormat format;
        int matchIndex = 0;

        HighlightingRule() = default;

        HighlightingRule(const CodeElement e, const QString &p, const int i)
            : element(e), pattern(qtCompatRegExp(p)), matchIndex(i)
        {
        }
    };

    void initialize();

    void highlightPythonBlock(const QString &text);

    bool matchMultiLine(const QString &text, const HighlightingRule &rule);

    QVector<HighlightingRule> m_highlightingRules;
    HighlightingRule m_triSingle;
    HighlightingRule m_triDouble;
};
