// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "AnnotationYaml.h"

#include <QHash>

namespace {

struct Line {
    int indent = 0;
    QString text;    // comment-stripped, right-trimmed content
    int number = 0;  // 1-based source line for errors
};

bool isSpace(const QChar& c) { return c == QLatin1Char(' '); }

// Strip comments outside quotes. '#' only opens a comment at line start or
// after whitespace, matching source YAML plain/quoted scalar rules. Leading
// whitespace is preserved because indentation drives the block structure.
QString stripComment(const QString& text) {
    bool single = false, doubled = false;
    for (int i = 0; i < text.size(); ++i) {
        const QChar c = text.at(i);
        if (single) {
            if (c == QLatin1Char('\'')) {
                if (i + 1 < text.size() && text.at(i + 1) == QLatin1Char('\''))
                    ++i;
                else
                    single = false;
            }
        } else if (doubled) {
            if (c == QLatin1Char('\\'))
                ++i;
            else if (c == QLatin1Char('"'))
                doubled = false;
        } else if (c == QLatin1Char('\'')) {
            single = true;
        } else if (c == QLatin1Char('"')) {
            doubled = true;
        } else if (c == QLatin1Char('#') &&
                   (i == 0 || isSpace(text.at(i - 1)))) {
            return text.left(i);
        }
    }
    return text;
}

// Join flow collections that span lines so the block parser sees one node.
QVector<Line> joinFlowLines(const QStringList& raw) {
    QVector<Line> lines;
    int depth = 0;
    bool single = false, doubled = false;
    for (int i = 0; i < raw.size(); ++i) {
        const QString source = stripComment(raw[i]);
        int indent = 0;
        while (indent < source.size() && isSpace(source.at(indent))) ++indent;
        if (source.mid(indent).startsWith(QLatin1Char('\t'))) {
            lines.append({indent, QStringLiteral("\t"), i + 1});
            continue;
        }
        // Keep the content without its leading indentation: the indent is
        // stored separately and the block parser matches on the content.
        const QString content = source.mid(indent);
        if (depth > 0 && !lines.isEmpty()) {
            lines.last().text += QLatin1Char(' ') + content;
        } else {
            lines.append({indent, content, i + 1});
        }
        for (int c = 0; c < source.size(); ++c) {
            const QChar ch = source.at(c);
            if (single) {
                if (ch == QLatin1Char('\'')) {
                    if (c + 1 < source.size() &&
                        source.at(c + 1) == QLatin1Char('\''))
                        ++c;
                    else
                        single = false;
                }
            } else if (doubled) {
                if (ch == QLatin1Char('\\'))
                    ++c;
                else if (ch == QLatin1Char('"'))
                    doubled = false;
            } else if (ch == QLatin1Char('\'')) {
                single = true;
            } else if (ch == QLatin1Char('"')) {
                doubled = true;
            } else if (ch == QLatin1Char('[') || ch == QLatin1Char('{')) {
                ++depth;
            } else if (ch == QLatin1Char(']') || ch == QLatin1Char('}')) {
                --depth;
            }
        }
    }
    return lines;
}

QString parseQuoted(const QString& text, bool* ok) {
    *ok = true;
    if (text.size() >= 2 && text.startsWith(QLatin1Char('"')) &&
        text.endsWith(QLatin1Char('"'))) {
        QString out;
        for (int i = 1; i < text.size() - 1; ++i) {
            if (text.at(i) == QLatin1Char('\\') && i + 1 < text.size() - 1) {
                const QChar next = text.at(++i);
                if (next == QLatin1Char('n'))
                    out += QLatin1Char('\n');
                else if (next == QLatin1Char('t'))
                    out += QLatin1Char('\t');
                else
                    out += next;
            } else {
                out += text.at(i);
            }
        }
        return out;
    }
    if (text.size() >= 2 && text.startsWith(QLatin1Char('\'')) &&
        text.endsWith(QLatin1Char('\''))) {
        return QString(text.mid(1, text.size() - 2))
                .replace(QStringLiteral("''"), QLatin1String("'"));
    }
    *ok = false;
    return text;
}

// Parse a scalar or inline flow collection appearing after "key:" or "- ".
bool parseInline(const QString& text,
                 int line,
                 AnnotationYamlNode* node,
                 QHash<QString, AnnotationYamlNode>* anchors,
                 const QHash<QString, AnnotationYamlNode>& anchorsConst,
                 QString* error);

AnnotationYamlNode parseBlock(const QVector<Line>& lines,
                              int* pos,
                              int minIndent,
                              QHash<QString, AnnotationYamlNode>* anchors,
                              QString* error);

QString scalarOrError(const QString& text, int line, QString* error) {
    if (text.startsWith(QLatin1Char('"')) ||
        text.startsWith(QLatin1Char('\''))) {
        bool ok = false;
        const QString value = parseQuoted(text, &ok);
        if (ok) return value;
    }
    if (text.contains(QStringLiteral(": ")) ||
        text.contains(QStringLiteral(":#"))) {
        *error = QStringLiteral("Line %1: unsupported YAML syntax").arg(line);
        return {};
    }
    return text;
}

void setEntry(AnnotationYamlNode* mapping,
              const QString& key,
              const AnnotationYamlNode& value) {
    for (auto& entry : mapping->mapping) {
        if (entry.first == key) {
            entry.second = value;  // duplicate key: last value wins
            return;
        }
    }
    mapping->mapping.append({key, value});
}

// Recursive-descent parser for flow collections ([a, b] / {k: v}) with a
// cursor; handles nesting, quoted scalars and trailing content checks.
AnnotationYamlNode parseFlow(const QString& text,
                             int* cursor,
                             int line,
                             QHash<QString, AnnotationYamlNode>* anchors,
                             QString* error) {
    AnnotationYamlNode node;
    node.line = line;
    const bool mapping = text[*cursor] == QLatin1Char('{');
    node.type = mapping ? AnnotationYamlNode::Mapping
                        : AnnotationYamlNode::Sequence;
    ++*cursor;
    QString pendingKey;
    while (*cursor < text.size()) {
        while (*cursor < text.size() && isSpace(text[*cursor])) ++*cursor;
        if (*cursor >= text.size()) break;
        const QChar c = text[*cursor];
        if ((mapping && c == QLatin1Char('}')) ||
            (!mapping && c == QLatin1Char(']'))) {
            ++*cursor;
            return node;
        }
        if (c == QLatin1Char(',')) {
            ++*cursor;
            continue;
        }
        // A nested collection starts a value directly.
        if (c == QLatin1Char('[') || c == QLatin1Char('{')) {
            AnnotationYamlNode nested =
                    parseFlow(text, cursor, line, anchors, error);
            if (!error->isEmpty()) return node;
            if (mapping) {
                if (pendingKey.isNull()) {
                    *error = QStringLiteral("Line %1: flow mapping needs a key")
                                     .arg(line);
                    return node;
                }
                setEntry(&node, pendingKey, nested);
                pendingKey = QString();
            } else {
                node.sequence.append(nested);
            }
            continue;
        }
        // Read one token up to a structural delimiter (quotes respected).
        const int start = *cursor;
        bool single = false, doubled = false;
        while (*cursor < text.size()) {
            const QChar ch = text[*cursor];
            if (single) {
                if (ch == QLatin1Char('\'')) {
                    if (*cursor + 1 < text.size() &&
                        text[*cursor + 1] == QLatin1Char('\''))
                        ++*cursor;
                    else
                        single = false;
                }
            } else if (doubled) {
                if (ch == QLatin1Char('\\'))
                    ++*cursor;
                else if (ch == QLatin1Char('"'))
                    doubled = false;
            } else if (ch == QLatin1Char('\'')) {
                single = true;
            } else if (ch == QLatin1Char('"')) {
                doubled = true;
            } else if (ch == QLatin1Char(',') || ch == QLatin1Char(']') ||
                       ch == QLatin1Char('}') || ch == QLatin1Char('[') ||
                       ch == QLatin1Char('{') ||
                       (!mapping && ch == QLatin1Char(':'))) {
                break;
            } else if (mapping && ch == QLatin1Char(':') &&
                       (*cursor + 1 >= text.size() ||
                        isSpace(text[*cursor + 1]))) {
                break;
            }
            ++*cursor;
        }
        const QString token = text.mid(start, *cursor - start).trimmed();
        if (mapping && *cursor < text.size() &&
            text[*cursor] == QLatin1Char(':') && !token.isNull()) {
            ++*cursor;
            bool quoted = false;
            pendingKey = parseQuoted(token, &quoted);
            if (!quoted) pendingKey = token;
            continue;
        }
        if (mapping) {
            if (pendingKey.isNull()) {
                *error = QStringLiteral("Line %1: flow mapping needs a key")
                                 .arg(line);
                return node;
            }
            AnnotationYamlNode value;
            value.type = AnnotationYamlNode::Scalar;
            value.line = line;
            value.scalar = scalarOrError(token, line, error);
            if (!error->isEmpty()) return node;
            setEntry(&node, pendingKey, value);
            pendingKey = QString();
        } else {
            AnnotationYamlNode value;
            value.type = AnnotationYamlNode::Scalar;
            value.line = line;
            value.scalar = scalarOrError(token, line, error);
            if (!error->isEmpty()) return node;
            node.sequence.append(value);
        }
    }
    *error = QStringLiteral("Line %1: unterminated flow collection").arg(line);
    return node;
}

}  // namespace

bool AnnotationYamlNode::isBool() const {
    if (type != Scalar) return false;
    static const QStringList truthy{
            QStringLiteral("true"), QStringLiteral("True"),
            QStringLiteral("TRUE"), QStringLiteral("yes"),
            QStringLiteral("Yes"),  QStringLiteral("YES"),
            QStringLiteral("on"),   QStringLiteral("On"),
            QStringLiteral("ON")};
    static const QStringList falsy{
            QStringLiteral("false"), QStringLiteral("False"),
            QStringLiteral("FALSE"), QStringLiteral("no"),
            QStringLiteral("No"),    QStringLiteral("NO"),
            QStringLiteral("off"),   QStringLiteral("Off"),
            QStringLiteral("OFF")};
    return truthy.contains(scalar) || falsy.contains(scalar);
}

bool AnnotationYamlNode::boolValue() const {
    static const QStringList truthy{
            QStringLiteral("true"), QStringLiteral("True"),
            QStringLiteral("TRUE"), QStringLiteral("yes"),
            QStringLiteral("Yes"),  QStringLiteral("YES"),
            QStringLiteral("on"),   QStringLiteral("On"),
            QStringLiteral("ON")};
    return type == Scalar && truthy.contains(scalar);
}

namespace {

bool parseInline(const QString& text,
                 int line,
                 AnnotationYamlNode* node,
                 QHash<QString, AnnotationYamlNode>* anchors,
                 const QHash<QString, AnnotationYamlNode>& anchorsConst,
                 QString* error) {
    QString value = text.trimmed();
    // Anchors and aliases: &name attaches the parsed value, *name copies a
    // previously anchored value from the same document.
    if (value.startsWith(QLatin1Char('&'))) {
        const int split = value.indexOf(QLatin1Char(' '));
        const QString name = split < 0 ? value.mid(1) : value.mid(1, split - 1);
        if (name.isEmpty()) {
            *error = QStringLiteral("Line %1: empty anchor name").arg(line);
            return false;
        }
        AnnotationYamlNode anchored;
        if (split < 0) {
            anchored.type = AnnotationYamlNode::Null;
        } else if (!parseInline(value.mid(split + 1), line, &anchored, anchors,
                                anchorsConst, error)) {
            return false;
        }
        *node = anchored;
        anchors->insert(name, anchored);
        return true;
    }
    if (value.startsWith(QLatin1Char('*'))) {
        const QString name = value.mid(1).trimmed();
        if (!anchorsConst.contains(name)) {
            *error = QStringLiteral("Line %1: unknown alias *%2")
                             .arg(line)
                             .arg(name);
            return false;
        }
        *node = anchorsConst.value(name);
        return true;
    }
    if (value.startsWith(QLatin1Char('[')) ||
        value.startsWith(QLatin1Char('{'))) {
        int cursor = 0;
        *node = parseFlow(value, &cursor, line, anchors, error);
        if (!error->isEmpty()) return false;
        while (cursor < value.size() && isSpace(value[cursor])) ++cursor;
        if (cursor < value.size()) {
            *error = QStringLiteral("Line %1: unsupported YAML syntax")
                             .arg(line);
            return false;
        }
        return true;
    }
    node->type = AnnotationYamlNode::Scalar;
    node->scalar = scalarOrError(value, line, error);
    return error->isEmpty();
}

AnnotationYamlNode parseBlock(const QVector<Line>& lines,
                              int* pos,
                              int minIndent,
                              QHash<QString, AnnotationYamlNode>* anchors,
                              QString* error) {
    AnnotationYamlNode node;
    if (*pos >= lines.size()) return node;
    const Line first = lines[*pos];
    if (first.text == QStringLiteral("\t")) {
        *error = QStringLiteral("Line %1: tabs are not supported")
                         .arg(first.number);
        return node;
    }
    if (first.indent < minIndent) return node;
    const bool isSequence = first.text == QLatin1String("-") ||
                            first.text.startsWith(QStringLiteral("- "));
    node.line = first.number;
    if (isSequence) {
        node.type = AnnotationYamlNode::Sequence;
        const int itemIndent = first.indent;
        while (*pos < lines.size()) {
            const Line line = lines[*pos];
            if (line.text.trimmed().isEmpty()) {
                ++*pos;
                continue;
            }
            if (line.text == QLatin1String("...")) break;  // document end
            if (line.text == QStringLiteral("\t")) {
                *error = QStringLiteral("Line %1: tabs are not supported")
                                 .arg(line.number);
                return node;
            }
            if (line.indent < itemIndent) break;
            // A non-item line at the parent's indentation is the next key
            // of the enclosing mapping; break instead of erroring.
            if (line.indent == itemIndent && line.text != QLatin1String("-") &&
                !line.text.startsWith(QStringLiteral("- ")))
                break;
            if (line.indent > itemIndent || line.text == QLatin1String("-") ||
                !line.text.startsWith(QStringLiteral("- "))) {
                *error = QStringLiteral("Line %1: unsupported YAML syntax")
                                 .arg(line.number);
                return node;
            }
            const QString payload = line.text.mid(2).trimmed();
            ++*pos;
            if (payload.isEmpty()) {
                // Nested block item: "- " followed by an indented node.
                node.sequence.append(
                        parseBlock(lines, pos, itemIndent + 1, anchors, error));
                if (!error->isEmpty()) return node;
                continue;
            }
            AnnotationYamlNode item;
            if (!parseInline(payload, line.number, &item, anchors, *anchors,
                             error))
                return node;
            node.sequence.append(item);
        }
        return node;
    }
    // Block mapping.
    node.type = AnnotationYamlNode::Mapping;
    const int mapIndent = first.indent;
    while (*pos < lines.size()) {
        const Line line = lines[*pos];
        if (line.text.trimmed().isEmpty()) {
            ++*pos;
            continue;
        }
        if (line.text == QLatin1String("...")) break;  // document end
        if (line.text == QStringLiteral("\t")) {
            *error = QStringLiteral("Line %1: tabs are not supported")
                             .arg(line.number);
            return node;
        }
        if (line.indent < mapIndent) break;
        if (line.indent > mapIndent) {
            *error = QStringLiteral("Line %1: unsupported YAML syntax")
                             .arg(line.number);
            return node;
        }
        // Split "key:" outside quotes and flow brackets.
        int split = -1;
        bool single = false, doubled = false;
        int depth = 0;
        for (int c = 0; c < line.text.size(); ++c) {
            const QChar ch = line.text.at(c);
            if (single) {
                if (ch == QLatin1Char('\'')) {
                    if (c + 1 < line.text.size() &&
                        line.text.at(c + 1) == QLatin1Char('\''))
                        ++c;
                    else
                        single = false;
                }
            } else if (doubled) {
                if (ch == QLatin1Char('\\'))
                    ++c;
                else if (ch == QLatin1Char('"'))
                    doubled = false;
            } else if (ch == QLatin1Char('\'')) {
                single = true;
            } else if (ch == QLatin1Char('"')) {
                doubled = true;
            } else if (ch == QLatin1Char('[') || ch == QLatin1Char('{')) {
                ++depth;
            } else if (ch == QLatin1Char(']') || ch == QLatin1Char('}')) {
                --depth;
            } else if (depth == 0 && ch == QLatin1Char(':') &&
                       (c + 1 == line.text.size() ||
                        isSpace(line.text.at(c + 1)))) {
                split = c;
                break;
            }
        }
        if (split < 0) {
            *error = QStringLiteral("Line %1: unsupported YAML syntax")
                             .arg(line.number);
            return node;
        }
        const QString keyText = line.text.left(split).trimmed();
        bool quoted = false;
        QString key = parseQuoted(keyText, &quoted);
        if (!quoted) key = keyText;
        QString rest = line.text.mid(split + 1).trimmed();
        ++*pos;
        // "key: &anchor" applies the anchor to the value parsed next.
        QString anchorName;
        if (rest.startsWith(QLatin1Char('&'))) {
            const int space = rest.indexOf(QLatin1Char(' '));
            if (space < 0) {
                anchorName = rest.mid(1);
                rest.clear();
            } else {
                anchorName = rest.mid(1, space - 1);
                rest = rest.mid(space + 1).trimmed();
            }
            if (anchorName.isEmpty()) {
                *error = QStringLiteral("Line %1: empty anchor name")
                                 .arg(line.number);
                return node;
            }
        }
        AnnotationYamlNode value;
        if (rest.isEmpty()) {
            // Sequence items may sit at the parent key's indentation, a
            // source yaml.safe_load allowance the canonical fixtures use.
            const bool sameIndentSequence =
                    *pos < lines.size() && lines[*pos].indent >= mapIndent &&
                    (lines[*pos].text == QLatin1String("-") ||
                     lines[*pos].text.startsWith(QStringLiteral("- ")));
            value = parseBlock(lines, pos,
                               sameIndentSequence ? mapIndent : mapIndent + 1,
                               anchors, error);
            if (!error->isEmpty()) return node;
        } else if (!parseInline(rest, line.number, &value, anchors, *anchors,
                                error)) {
            return node;
        }
        if (!anchorName.isEmpty()) anchors->insert(anchorName, value);
        setEntry(&node, key, value);
    }
    return node;
}

}  // namespace

AnnotationYamlResult parseAnnotationYaml(const QString& text) {
    AnnotationYamlResult result;
    QString normalized = text;
    while (!normalized.isEmpty() && normalized.front() == QChar(0xFEFF))
        normalized.remove(0, 1);
    normalized.replace(QStringLiteral("\r\n"), QStringLiteral("\n"));
    QStringList raw = normalized.split(QLatin1Char('\n'));
    // Skip a leading document marker; reject anything that follows an end
    // marker or a second start marker (yaml.safe_load is single-document).
    while (!raw.isEmpty() && raw.first().trimmed().isEmpty()) raw.removeFirst();
    if (!raw.isEmpty() && raw.first().trimmed() == QStringLiteral("---"))
        raw.removeFirst();
    const QVector<Line> lines = joinFlowLines(raw);
    int pos = 0;
    QHash<QString, AnnotationYamlNode> anchors;
    result.root = parseBlock(lines, &pos, 0, &anchors, &result.error);
    if (!result.error.isEmpty()) return result;
    for (; pos < lines.size(); ++pos) {
        const QString trimmed = lines[pos].text.trimmed();
        if (trimmed.isEmpty() || trimmed == QStringLiteral("...")) continue;
        result.error = QStringLiteral(
                               "Line %1: multiple YAML documents are not "
                               "supported")
                               .arg(lines[pos].number);
        return result;
    }
    return result;
}
