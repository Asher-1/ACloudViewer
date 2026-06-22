// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ecvPythonSyntaxHighlighter.h"

ecvPythonSyntaxHighlighter::ecvPythonSyntaxHighlighter(QTextDocument* parent)
    : QSyntaxHighlighter(parent) {
    // --- Keywords (Python 3) ---
    QTextCharFormat kwFmt;
    kwFmt.setForeground(QColor(86, 156, 214));
    kwFmt.setFontWeight(QFont::Bold);

    const QStringList keywords = {
            "False",  "None",     "True",  "and",    "as",       "assert",
            "async",  "await",    "break", "class",  "continue", "def",
            "del",    "elif",     "else",  "except", "finally",  "for",
            "from",   "global",   "if",    "import", "in",       "is",
            "lambda", "nonlocal", "not",   "or",     "pass",     "raise",
            "return", "try",      "while", "with",   "yield",
    };
    for (const auto& kw : keywords) {
        Rule rule;
        rule.pattern = QRegularExpression(QStringLiteral("\\b%1\\b").arg(kw));
        rule.format = kwFmt;
        m_rules.append(rule);
    }

    // --- Built-in functions ---
    QTextCharFormat builtinFmt;
    builtinFmt.setForeground(QColor(220, 220, 170));

    const QStringList builtins = {
            "abs",         "all",          "any",      "bin",
            "bool",        "bytes",        "callable", "chr",
            "classmethod", "compile",      "complex",  "delattr",
            "dict",        "dir",          "divmod",   "enumerate",
            "eval",        "exec",         "filter",   "float",
            "format",      "frozenset",    "getattr",  "globals",
            "hasattr",     "hash",         "help",     "hex",
            "id",          "input",        "int",      "isinstance",
            "issubclass",  "iter",         "len",      "list",
            "locals",      "map",          "max",      "memoryview",
            "min",         "next",         "object",   "oct",
            "open",        "ord",          "pow",      "print",
            "property",    "range",        "repr",     "reversed",
            "round",       "set",          "setattr",  "slice",
            "sorted",      "staticmethod", "str",      "sum",
            "super",       "tuple",        "type",     "vars",
            "zip",
    };
    for (const auto& fn : builtins) {
        Rule rule;
        rule.pattern = QRegularExpression(QStringLiteral("\\b%1\\b").arg(fn));
        rule.format = builtinFmt;
        m_rules.append(rule);
    }

    // --- self / cls ---
    QTextCharFormat selfFmt;
    selfFmt.setForeground(QColor(86, 156, 214));
    selfFmt.setFontItalic(true);
    {
        Rule rule;
        rule.pattern = QRegularExpression(QStringLiteral("\\bself\\b"));
        rule.format = selfFmt;
        m_rules.append(rule);
    }
    {
        Rule rule;
        rule.pattern = QRegularExpression(QStringLiteral("\\bcls\\b"));
        rule.format = selfFmt;
        m_rules.append(rule);
    }

    // --- Decorators ---
    QTextCharFormat decoFmt;
    decoFmt.setForeground(QColor(220, 220, 170));
    decoFmt.setFontItalic(true);
    {
        Rule rule;
        rule.pattern = QRegularExpression(QStringLiteral("@\\w+"));
        rule.format = decoFmt;
        m_rules.append(rule);
    }

    // --- Numbers (int, float, hex, oct, bin) ---
    QTextCharFormat numFmt;
    numFmt.setForeground(QColor(181, 206, 168));
    {
        Rule rule;
        rule.pattern = QRegularExpression(
                QStringLiteral("\\b0[xX][0-9a-fA-F]+\\b"
                               "|\\b0[oO][0-7]+\\b"
                               "|\\b0[bB][01]+\\b"
                               "|\\b\\d+\\.\\d*([eE][+-]?\\d+)?\\b"
                               "|\\b\\.\\d+([eE][+-]?\\d+)?\\b"
                               "|\\b\\d+([eE][+-]?\\d+)?\\b"));
        rule.format = numFmt;
        m_rules.append(rule);
    }

    // --- Operators ---
    QTextCharFormat opFmt;
    opFmt.setForeground(QColor(212, 212, 212));
    {
        Rule rule;
        rule.pattern = QRegularExpression(QStringLiteral("[+\\-*/%=<>!&|^~]+"));
        rule.format = opFmt;
        m_rules.append(rule);
    }

    // --- Single-line strings (double and single quotes) ---
    QTextCharFormat strFmt;
    strFmt.setForeground(QColor(206, 145, 120));
    {
        Rule rule;
        rule.pattern = QRegularExpression(QStringLiteral(
                "\"(?:[^\"\\\\]|\\\\.)*\"|'(?:[^'\\\\]|\\\\.)*'"));
        rule.format = strFmt;
        m_rules.append(rule);
    }

    // --- f-string prefix ---
    {
        Rule rule;
        rule.pattern =
                QRegularExpression(QStringLiteral("[fFrRbBuU]+(?=\"|\')"));
        rule.format = strFmt;
        m_rules.append(rule);
    }

    // --- Comments (must be last single-line rule) ---
    QTextCharFormat commentFmt;
    commentFmt.setForeground(QColor(106, 153, 85));
    commentFmt.setFontItalic(true);
    {
        Rule rule;
        rule.pattern = QRegularExpression(QStringLiteral("#[^\n]*"));
        rule.format = commentFmt;
        m_rules.append(rule);
    }

    // --- Multi-line strings ---
    m_multiLineStringFormat = strFmt;
    m_triSingleStart = QRegularExpression(QStringLiteral("'''"));
    m_triDoubleStart = QRegularExpression(QStringLiteral("\"\"\""));
    m_triSingleEnd = QRegularExpression(QStringLiteral("'''"));
    m_triDoubleEnd = QRegularExpression(QStringLiteral("\"\"\""));
}

void ecvPythonSyntaxHighlighter::highlightBlock(const QString& text) {
    for (const auto& rule : m_rules) {
        auto it = rule.pattern.globalMatch(text);
        while (it.hasNext()) {
            auto match = it.next();
            setFormat(match.capturedStart(), match.capturedLength(),
                      rule.format);
        }
    }

    // Multi-line string states: 0 = none, 1 = in ''', 2 = in """
    setCurrentBlockState(0);

    auto processTripleQuote = [this, &text](const QRegularExpression& startExpr,
                                            const QRegularExpression& endExpr,
                                            int state) {
        int startIdx = 0;

        if (previousBlockState() != state) {
            auto match = startExpr.match(text);
            startIdx = match.hasMatch() ? match.capturedStart() : -1;
        }

        while (startIdx >= 0) {
            auto endMatch = endExpr.match(text, startIdx + 3);
            int endIdx = endMatch.hasMatch() ? endMatch.capturedStart() : -1;
            int length;

            if (endIdx == -1) {
                setCurrentBlockState(state);
                length = text.length() - startIdx;
            } else {
                length = endIdx - startIdx + endMatch.capturedLength();
            }

            setFormat(startIdx, length, m_multiLineStringFormat);

            auto nextMatch = startExpr.match(text, startIdx + length);
            startIdx = nextMatch.hasMatch() ? nextMatch.capturedStart() : -1;
        }
    };

    processTripleQuote(m_triSingleStart, m_triSingleEnd, 1);
    processTripleQuote(m_triDoubleStart, m_triDoubleEnd, 2);
}
