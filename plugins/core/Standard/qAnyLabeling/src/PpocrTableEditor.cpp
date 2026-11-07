// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "PpocrTableEditor.h"

#include <QAbstractScrollArea>
#include <QAction>
#include <QApplication>
#include <QClipboard>
#include <QFrame>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QIcon>
#include <QItemSelectionModel>
#include <QKeyEvent>
#include <QLineEdit>
#include <QMap>
#include <QMenu>
#include <QMouseEvent>
#include <QPainter>
#include <QPalette>
#include <QPixmap>
#include <QPushButton>
#include <QRegularExpression>
#include <QScrollBar>
#include <QSet>
#include <QShortcut>
#include <QSignalBlocker>
#include <QStyle>
#include <QStyleOptionViewItem>
#include <QStyledItemDelegate>
#include <QTableWidget>
#include <QTimer>
#include <QToolButton>
#include <QVBoxLayout>
#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <optional>

#include "QtCompat.h"

namespace {
using Cell = QPair<int, int>;
QIcon formatIcon(int style, const QColor& color) {
    QPixmap pixmap(40, 40);
    pixmap.setDevicePixelRatio(2);
    pixmap.fill(Qt::transparent);
    QPainter painter(&pixmap);
    painter.setRenderHints(QPainter::Antialiasing | QPainter::TextAntialiasing);
    painter.setPen(color);
    QFont font("Georgia");
    font.setPixelSize(18);
    if (style == 0) font.setWeight(QFont::Bold);
    if (style == 1) font.setItalic(true);
    painter.setFont(font);
    painter.drawText(QRectF(0, 0, 20, 20), Qt::AlignCenter,
                     style == 0   ? "B"
                     : style == 1 ? "I"
                                  : "S");
    if (style == 2) {
        painter.setPen(QPen(color, 1.3));
        painter.drawLine(QPointF(3, 10), QPointF(17, 10));
    }
    painter.end();
    return QIcon(pixmap);
}
class TableDelegate final : public QStyledItemDelegate {
public:
    explicit TableDelegate(QObject* parent) : QStyledItemDelegate(parent) {}
    void paint(QPainter* painter,
               const QStyleOptionViewItem& option,
               const QModelIndex& index) const override {
        auto unfocused = option;
        unfocused.state &= ~QStyle::State_HasFocus;
        if (unfocused.state & QStyle::State_Editing) unfocused.text.clear();
        QStyledItemDelegate::paint(painter, unfocused, index);
    }
    QWidget* createEditor(QWidget* parent,
                          const QStyleOptionViewItem&,
                          const QModelIndex&) const override {
        auto* editor = new QLineEdit(parent);
        editor->setFrame(false);
        editor->setAttribute(Qt::WA_MacShowFocusRect, false);
        editor->setAlignment(Qt::AlignLeft | Qt::AlignVCenter);
        editor->setStyleSheet("QLineEdit:focus { outline: none; }");
        return editor;
    }
    void updateEditorGeometry(QWidget* editor,
                              const QStyleOptionViewItem& option,
                              const QModelIndex&) const override {
        editor->setGeometry(option.rect);
    }
};
struct State {
    QString text;
    bool bold = false, italic = false, strike = false;
};
struct Parsed {
    int rows = 0, columns = 0;
    QMap<Cell, State> states;
    QMap<Cell, Cell> spans;
};
const QRegularExpression tokenPattern(
        "<(fcel|ecel|lcel|ucel|xcel|nl)>",
        QRegularExpression::CaseInsensitiveOption);
const QRegularExpression stylePattern(
        "</?(?:b|strong|i|em|s|strike|del)>",
        QRegularExpression::CaseInsensitiveOption);
QString unescape(const QString& text) {
    // Python html.unescape: HTML5 names, optional semicolons, longest matching
    // legacy prefix, and numeric invalid-codepoint/Windows-1252 replacements.
    // The mapping is compiled in; installed plugins need no external data.
    struct Entity {
        const char* name;
        uint first, second;
    };
    static const Entity entities[] = {
#include "PpocrHtmlEntities.inc"
    };
    static const uint controls[] = {
            0x20ac, 0x81,   0x201a, 0x192,  0x201e, 0x2026, 0x2020, 0x2021,
            0x2c6,  0x2030, 0x160,  0x2039, 0x152,  0x8d,   0x17d,  0x8f,
            0x90,   0x2018, 0x2019, 0x201c, 0x201d, 0x2022, 0x2013, 0x2014,
            0x2dc,  0x2122, 0x161,  0x203a, 0x153,  0x9d,   0x17e,  0x178};
    static const QRegularExpression pattern(
            "&(#[0-9]+;?|#[xX][0-9a-fA-F]+;?|[^\\t\\n\\f <&#;]{1,32};?)");
    QString output;
    int offset = 0;
    auto matches = pattern.globalMatch(text);
    while (matches.hasNext()) {
        const auto match = matches.next();
        output += text.mid(offset, match.capturedStart() - offset);
        QString reference = match.captured(1), replacement;
        if (reference.startsWith('#')) {
            const bool hex =
                    reference.size() > 1 && reference[1].toLower() == 'x';
            if (reference.endsWith(';')) reference.chop(1);
            bool valid = false;
            auto number = reference.mid(hex ? 2 : 1)
                                  .toULongLong(&valid, hex ? 16 : 10);
            uint point = 0xfffd;
            if (valid && number <= 0x10ffff &&
                !(number >= 0xd800 && number <= 0xdfff))
                point = uint(number);
            if (point >= 0x80 && point <= 0x9f)
                point = controls[point - 0x80];
            else if (point == 0)
                point = 0xfffd;
            else if ((point >= 1 && point <= 8) || point == 11 ||
                     (point >= 14 && point <= 31) || point == 127 ||
                     (point >= 0xfdd0 && point <= 0xfdef) ||
                     (point & 0xffff) >= 0xfffe)
                point = 0;
            if (point) replacement = QString::fromUcs4(&point, 1);
        } else {
            replacement = match.captured();
            // At most 33 prefix probes. Binary search avoids building a
            // per-dialog hash and keeps the 2231-entry constant table shared.
            for (int length = reference.size(); length > 1; --length) {
                const auto name = reference.left(length).toUtf8();
                const auto found = std::lower_bound(
                        std::begin(entities), std::end(entities), name,
                        [](const Entity& item, const QByteArray& value) {
                            return QByteArray(item.name) < value;
                        });
                if (found == std::end(entities) || name != found->name)
                    continue;
                const uint points[] = {found->first, found->second};
                replacement = QString::fromUcs4(points, found->second ? 2 : 1) +
                              reference.mid(length);
                break;
            }
        }
        output += replacement;
        offset = match.capturedEnd();
    }
    output += text.mid(offset);
    return output;
}
QString escape(QString text) {
    text.replace('&', "&amp;");
    text.replace('<', "&lt;");
    text.replace('>', "&gt;");
    text.replace('"', "&quot;");
    text.replace('\'', "&#x27;");
    return text;
}
State decode(QString payload) {
    const auto lower = payload.toLower();
    State result;
    result.bold = lower.contains("<b>") || lower.contains("<strong>");
    result.italic = lower.contains("<i>") || lower.contains("<em>");
    result.strike = lower.contains("<s>") || lower.contains("<strike>") ||
                    lower.contains("<del>");
    payload.remove(stylePattern);
    result.text = unescape(payload.trimmed());
    return result;
}
std::optional<Parsed> tokenTable(const QString& content) {
    auto matches = tokenPattern.globalMatch(content);
    QVector<QRegularExpressionMatch> found;
    while (matches.hasNext()) found.append(matches.next());
    if (found.isEmpty()) return std::nullopt;
    QVector<QVector<QPair<QString, QString>>> rows;
    QVector<QPair<QString, QString>> row;
    for (int i = 0; i < found.size(); ++i) {
        const auto token = found[i].captured(1).toLower();
        const int end = i + 1 < found.size() ? found[i + 1].capturedStart()
                                             : content.size();
        const auto payload = content.mid(found[i].capturedEnd(),
                                         end - found[i].capturedEnd())
                                     .trimmed();
        if (token == "nl") {
            rows.append(row);
            row.clear();
        } else
            row.append({token, payload});
    }
    if (!row.isEmpty()) rows.append(row);
    while (!rows.isEmpty() && rows.last().isEmpty()) rows.removeLast();
    Parsed result;
    result.rows = rows.size();
    for (const auto& cells : rows)
        result.columns = std::max(result.columns, int(cells.size()));
    QMap<Cell, Cell> owner;
    for (int r = 0; r < result.rows; ++r) {
        for (int c = 0; c < result.columns; ++c) {
            const Cell cell{r, c};
            const auto entry = c < rows[r].size()
                                       ? rows[r][c]
                                       : qMakePair(QString("ecel"), QString());
            const auto token = entry.first;
            owner[cell] = cell;
            if (token == "fcel" || token == "ecel")
                result.states[cell] = decode(entry.second);
            else if (token == "lcel")
                owner[cell] = owner.value({r, c - 1}, cell);
            else if (token == "ucel")
                owner[cell] = owner.value({r - 1, c}, cell);
            else if (token == "xcel")
                owner[cell] =
                        owner.value({r - 1, c}, owner.value({r, c - 1}, cell));
            if (!result.states.contains(cell)) result.states[cell] = {};
        }
    }
    QMap<Cell, QVector<Cell>> grouped;
    for (auto it = owner.begin(); it != owner.end(); ++it)
        grouped[it.value()].append(it.key());
    for (const auto& cells : grouped) {
        if (cells.size() <= 1) continue;
        int top = result.rows, left = result.columns, bottom = -1, right = -1;
        for (const auto& cell : cells) {
            top = std::min(top, cell.first);
            left = std::min(left, cell.second);
            bottom = std::max(bottom, cell.first);
            right = std::max(right, cell.second);
        }
        if (cells.size() != (bottom - top + 1) * (right - left + 1)) continue;
        const Cell anchor{top, left};
        result.spans[anchor] = {bottom - top + 1, right - left + 1};
        // QMap traversal is the source's sorted row/column order.
        if (result.states[anchor].text.isEmpty())
            for (const auto& cell : cells)
                if (!result.states[cell].text.isEmpty()) {
                    result.states[anchor] = result.states[cell];
                    break;
                }
        for (const auto& cell : cells)
            if (cell != anchor) result.states.remove(cell);
    }
    return result;
}
bool htmlTableLike(const QString& content) {
    const auto options = QRegularExpression::CaseInsensitiveOption;
    return content.contains(QRegularExpression("<\\s*table\\b", options)) &&
           content.contains(QRegularExpression("<\\s*tr\\b", options)) &&
           content.contains(QRegularExpression("<\\s*t[dh]\\b", options));
}
int spanValue(const QString& attributes, const QString& name) {
    const QRegularExpression attribute(
            R"re(\b)re" + name +
                    R"re(\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+)))re",
            QRegularExpression::CaseInsensitiveOption);
    const auto match = attribute.match(attributes);
    if (!match.hasMatch()) return 1;
    bool ok = false;
    const QString value = !match.captured(1).isNull()   ? match.captured(1)
                          : !match.captured(2).isNull() ? match.captured(2)
                                                        : match.captured(3);
    const int size = value.trimmed().toInt(&ok);
    return ok ? std::max(1, size) : 1;
}
std::optional<Parsed> htmlTable(const QString& content) {
    if (!htmlTableLike(content)) return std::nullopt;
    struct HtmlCell {
        State state;
        int rows = 1, columns = 1;
    };
    QVector<QVector<HtmlCell>> rows;
    QVector<HtmlCell> row;
    std::optional<HtmlCell> cell;
    int depth = 0;
    // Mirror HTMLParser's explicit cell/row closure; quoted > stays in
    // attributes.
    static const QRegularExpression events(
            R"re(<!--[\s\S]*?-->|<(/?)([a-zA-Z][a-zA-Z0-9]*)(\b(?:[^>"']|"[^"]*"|'[^']*')*)>|([^<]+))re",
            QRegularExpression::CaseInsensitiveOption);
    auto matches = events.globalMatch(content);
    while (matches.hasNext()) {
        const auto match = matches.next();
        if (!match.captured(4).isNull()) {
            if (depth > 0 && cell)
                cell->state.text += unescape(match.captured(4));
            continue;
        }
        const auto tag = match.captured(2).toLower();
        const bool closing = match.captured(1) == "/";
        if (tag == "table") {
            depth = std::max(0, depth + (closing ? -1 : 1));
            continue;
        }
        if (depth <= 0) continue;
        if (tag == "tr") {
            if (closing) {
                rows.append(row);
                cell.reset();
            }
            row.clear();
            continue;
        }
        if (tag == "td" || tag == "th") {
            if (!closing) {
                cell = HtmlCell{};
                cell->rows = spanValue(match.captured(3), "rowspan");
                cell->columns = spanValue(match.captured(3), "colspan");
            } else if (cell) {
                cell->state.text = cell->state.text.trimmed();
                row.append(*cell);
                cell.reset();
            }
            continue;
        }
        if (!closing && cell) {
            if (tag == "b" || tag == "strong") cell->state.bold = true;
            if (tag == "i" || tag == "em") cell->state.italic = true;
            if (tag == "s" || tag == "strike" || tag == "del")
                cell->state.strike = true;
            if (tag == "br") cell->state.text += '\n';
        }
    }
    Parsed result;
    QSet<Cell> occupied;
    int r = 0;
    for (const auto& cells : rows) {
        if (cells.isEmpty()) continue;
        int c = 0;
        for (const auto& item : cells) {
            while (occupied.contains({r, c})) ++c;
            result.states[{r, c}] = item.state;
            if (item.rows > 1 || item.columns > 1)
                result.spans[{r, c}] = {item.rows, item.columns};
            for (int dr = 0; dr < item.rows; ++dr)
                for (int dc = 0; dc < item.columns; ++dc)
                    if (dr || dc) occupied.insert({r + dr, c + dc});
            result.rows = std::max(result.rows, r + item.rows);
            result.columns = std::max(result.columns, c + item.columns);
            c += item.columns;
        }
        ++r;
    }
    if (!result.rows || !result.columns) return std::nullopt;
    return result;
}
QTableWidgetItem* ensureItem(QTableWidget* table, int r, int c) {
    auto* item = table->item(r, c);
    if (!item) {
        item = new QTableWidgetItem;
        item->setTextAlignment(Qt::AlignLeft | Qt::AlignVCenter);
        table->setItem(r, c, item);
    }
    return item;
}
QString payload(const QTableWidgetItem* item) {
    if (!item || item->text().isEmpty()) return {};
    auto result = escape(item->text());
    const auto font = item->font();
    if (font.strikeOut()) result = "<s>" + result + "</s>";
    if (font.italic()) result = "<i>" + result + "</i>";
    if (font.bold()) result = "<b>" + result + "</b>";
    return result;
}
QMap<Cell, Cell> owners(const QTableWidget* table) {
    // O(n log n) map construction for n cells avoids scanning every preceding
    // cell for each span lookup. Extra memory is O(n), owned by this operation.
    QMap<Cell, Cell> result;
    for (int r = 0; r < table->rowCount(); ++r)
        for (int c = 0; c < table->columnCount(); ++c) {
            const Cell cell{r, c};
            if (result.contains(cell)) continue;
            result[cell] = cell;
            for (int dr = 0; dr < table->rowSpan(r, c); ++dr)
                for (int dc = 0; dc < table->columnSpan(r, c); ++dc)
                    result[{r + dr, c + dc}] = cell;
        }
    return result;
}
void syncTableGeometry(QTableWidget* grid) {
    const int columns = grid->columnCount(),
              viewport = grid->viewport()->width();
    const auto owner = owners(grid);
    const auto metrics = grid->fontMetrics();
    const int usable = std::max(columns, viewport);
    for (int c = 0; c < columns && viewport > 0; ++c) {
        int textWidth = 0;
        for (int r = 0; r < grid->rowCount(); ++r) {
            if (owner.value({r, c}) != Cell(r, c) || !grid->item(r, c))
                continue;
            textWidth = std::max(textWidth,
                                 QTCOMPAT_FONTMETRICS_WIDTH(
                                         metrics, grid->item(r, c)->text()) /
                                         std::max(1, grid->columnSpan(r, c)));
        }
        const int content = std::max(36, std::min(560, textWidth + 28));
        grid->setColumnWidth(
                c,
                std::max(usable / columns + (c < usable % columns), content));
    }
    const double height = metrics.height() * 1.45;
    const int lower = int(std::floor(height));
    // Python round uses ties to even, independently of the process FP mode.
    const int rounded = lower + (height - lower > .5 ||
                                 (height - lower == .5 && lower % 2));
    const int rowHeight = std::max(28, rounded + 12);
    grid->verticalHeader()->setDefaultSectionSize(rowHeight);
    for (int r = 0; r < grid->rowCount(); ++r) grid->setRowHeight(r, rowHeight);
    int totalWidth = 0;
    for (int c = 0; c < columns; ++c) totalWidth += grid->columnWidth(c);
    int target = std::max(1, grid->rowCount()) * rowHeight +
                 grid->frameWidth() * 2 + (grid->showGrid() ? 1 : 0);
    if (columns > 0 && viewport > 0 && totalWidth > viewport)
        target += grid->horizontalScrollBar()->sizeHint().height();
    // QScrollArea is the source container; QListWidget supplies the same
    // viewport contract in this workspace. Start above the table itself.
    for (auto* parent = grid->parentWidget(); parent;
         parent = parent->parentWidget()) {
        auto* scroll = qobject_cast<QAbstractScrollArea*>(parent);
        if (!scroll) continue;
        const int height = scroll->viewport()->height();
        const int top = std::clamp(
                grid->mapTo(scroll->viewport(), QPoint(0, 0)).y(), 0, height);
        const int available = std::max(0, height - top - 20);
        if (available > 0) target = std::min(target, available);
        break;
    }
    target = std::max(rowHeight + grid->frameWidth() * 2, target);
    grid->setMinimumHeight(target);
    grid->setMaximumHeight(target);
}
class TableGrid final : public QTableWidget {
public:
    explicit TableGrid(QWidget* parent) : QTableWidget(parent) {}
    std::function<void()> copy, paste, undo, redo, changed;
    void restoreRightClickSelection(const QModelIndex& index) {
        const auto snapshot = m_rightClickSelection;
        m_rightClickSelection.clear();
        if (!index.isValid() ||
            !snapshot.contains({index.row(), index.column()}) ||
            selectedIndexes().size() > 1)
            return;
        clearSelection();
        for (const auto& cell : snapshot)
            selectionModel()->select(model()->index(cell.first, cell.second),
                                     QItemSelectionModel::Select);
        selectionModel()->setCurrentIndex(index, QItemSelectionModel::NoUpdate);
    }

protected:
    void resizeEvent(QResizeEvent* event) override {
        QTableWidget::resizeEvent(event);
        // Coalesce scrollbar/viewport resizes; ownership cancels on teardown.
        if (m_geometryPending) return;
        m_geometryPending = true;
        QTimer::singleShot(0, this, [this]() {
            m_geometryPending = false;
            syncTableGeometry(this);
        });
    }
    void mousePressEvent(QMouseEvent* event) override {
        if (event->button() == Qt::RightButton) {
            m_rightClickSelection.clear();
            const auto index = indexAt(event->pos());
            if (index.isValid() && selectedIndexes().size() > 1 &&
                selectionModel()->isSelected(index))
                for (const auto& cell : selectedIndexes())
                    m_rightClickSelection.append({cell.row(), cell.column()});
        }
        QTableWidget::mousePressEvent(event);
    }
    void keyPressEvent(QKeyEvent* event) override {
        if (event->matches(QKeySequence::Copy))
            copy();
        else if (event->matches(QKeySequence::Paste))
            paste();
        else if (event->matches(QKeySequence::Undo))
            undo();
        else if (event->matches(QKeySequence::Redo) ||
                 (event->modifiers() == Qt::ControlModifier &&
                  event->key() == Qt::Key_Y))
            redo();
        else {
            QTableWidget::keyPressEvent(event);
            return;
        }
        event->accept();
    }

private:
    QVector<Cell> m_rightClickSelection;
    bool m_geometryPending = false;
};
}  // namespace
namespace PpocrTable {
bool isTable(const QString& label, const QString& content) {
    auto normalized = label.trimmed().toCaseFolded();
    normalized.replace(QRegularExpression("[\\s-]+"), "_");
    // Source dispatches formula labels before inspecting table-like content.
    if (normalized == "display_formula" || normalized == "formula_number" ||
        normalized == "algorithm" || normalized == "formula")
        return false;
    return normalized == "table" || content.contains(tokenPattern) ||
           htmlTableLike(content);
}
void load(QTableWidget* table, const QString& content) {
    auto parsed = tokenTable(content);
    if (!parsed) parsed = htmlTable(content);
    if (!parsed) {
        Parsed fallback;
        const auto raw = content.trimmed();
        if (!raw.isEmpty()) {
            QStringList lines;
            for (const auto& line : raw.split(QRegularExpression(
                         "\\r\\n|[\\n\\r\\x{b}\\x{c}\\x{1c}-\\x{1e}\\x{85}\\x{"
                         "2028}\\x{2029}]")))
                if (!line.trimmed().isEmpty() || line.contains('\t'))
                    lines.append(line);
            const auto grid = lines.size() > 1 || raw.contains('\t')
                                      ? lines
                                      : QStringList{raw};
            fallback.rows = grid.size();
            for (int r = 0; r < grid.size(); ++r) {
                const auto cells = grid[r].split('\t');
                fallback.columns =
                        std::max(fallback.columns, int(cells.size()));
                for (int c = 0; c < cells.size(); ++c)
                    fallback.states[{r, c}].text = cells[c].trimmed();
            }
        }
        parsed = fallback;
    }
    const QSignalBlocker blocked(table);
    table->clearSpans();
    table->clearContents();
    table->setRowCount(parsed->rows);
    table->setColumnCount(parsed->columns);
    for (int r = 0; r < parsed->rows; ++r)
        for (int c = 0; c < parsed->columns; ++c) {
            const auto state = parsed->states.value({r, c});
            auto* item = new QTableWidgetItem(state.text);
            item->setTextAlignment(Qt::AlignLeft | Qt::AlignVCenter);
            auto font = table->font();
            font.setBold(state.bold);
            font.setItalic(state.italic);
            font.setStrikeOut(state.strike);
            item->setFont(font);
            table->setItem(r, c, item);
        }
    for (auto it = parsed->spans.begin(); it != parsed->spans.end(); ++it)
        table->setSpan(it.key().first, it.key().second, it.value().first,
                       it.value().second);
    if (parsed->rows && parsed->columns) table->setCurrentCell(0, 0);
}
QString tokens(const QTableWidget* table) {
    const auto owner = owners(table);
    QString result;
    for (int r = 0; r < table->rowCount(); ++r) {
        for (int c = 0; c < table->columnCount(); ++c) {
            const auto anchor = owner.value({r, c}, {r, c});
            if (anchor == Cell(r, c)) {
                const auto text = payload(table->item(r, c));
                result += text.isEmpty() ? QString("<ecel>")
                                         : QString("<fcel>" + text);
            } else if (r > anchor.first && c > anchor.second)
                result += "<xcel>";
            else if (r > anchor.first)
                result += "<ucel>";
            else
                result += "<lcel>";
        }
        result += "<nl>";
    }
    return result;
}
}  // namespace PpocrTable

PpocrTableEditor::PpocrTableEditor(const QString& text, QWidget* parent)
    : QFrame(parent) {
    setObjectName("PPOCRTableBlockEditor");
    // Theme ownership stays with this workspace, rather than QApplication.
    if (parent) setPalette(parent->palette());
    const bool dark = palette().color(QPalette::Window).lightness() < 128;
    const QString background = dark ? "#1c1c1e" : "#ffffff",
                  textColor = dark ? "#f5f5f7" : "#1d1d1f",
                  secondary = dark ? "#aeaeb2" : "#86868b",
                  border = dark ? "#3a3a3c" : "#E5E5E5",
                  panel = dark ? "#2c2c2e" : "#f5f6f7",
                  selection = dark ? "#48484a" : "rgb(241, 245, 255)";
    auto* frame = this;
    frame->setStyleSheet(
            QString("QFrame#PPOCRTableBlockEditor { background: %1; border: "
                    "1px solid %2; border-radius: 10px; }"
                    "QFrame#PPOCRTableBlockEditorToolbar { background: %3; "
                    "border: none; border-top-left-radius: 10px; "
                    "border-top-right-radius: 10px; border-bottom: 1px solid "
                    "%2; }"
                    "QFrame#PPOCRTableBlockEditorBody { background: %1; "
                    "border: "
                    "none; border-bottom-left-radius: 10px; "
                    "border-bottom-right-radius: 10px; }")
                    .arg(background, border, panel));
    auto* layout = new QVBoxLayout(frame);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(0);
    auto* toolbarFrame = new QFrame(frame);
    toolbarFrame->setObjectName("PPOCRTableBlockEditorToolbar");
    auto* toolbar = new QHBoxLayout(toolbarFrame);
    toolbar->setContentsMargins(12, 0, 12, 0);
    toolbar->setSpacing(4);
    auto* body = new QFrame(frame);
    body->setObjectName("PPOCRTableBlockEditorBody");
    auto* bodyLayout = new QVBoxLayout(body);
    bodyLayout->setContentsMargins(16, 16, 16, 16);
    bodyLayout->setSpacing(0);
    auto* grid = new TableGrid(body);
    grid->setItemDelegate(new TableDelegate(grid));
    grid->setProperty("ppocrMultiSelect", false);
    grid->setStyleSheet(
            QString("QTableWidget { background: %1; color: %2; border: 1px "
                    "solid %3; gridline-color: %3; selection-background-color: "
                    "%4; selection-color: %2; font-size: 13px; }"
                    "QTableWidget::item { padding: 6px 10px; border: none; }"
                    "QTableWidget::item:focus { outline: none; }"
                    "QTableWidget[ppocrMultiSelect=\"true\"]::item:selected { "
                    "background: %4; }"
                    "QTableWidget[ppocrMultiSelect=\"false\"]::item:selected { "
                    "background: %4; }"
                    "QTableWidget::item:selected:active { background: %4; }"
                    "QTableWidget::item:selected:!active { background: %4; }"
                    "QTableWidget QLineEdit { background: %4; border: none; "
                    "border-radius: 0px; color: %2; padding: 0px 10px; margin: "
                    "0px; selection-background-color: rgb(70, 88, 255); "
                    "selection-color: #ffffff; }"
                    "QTableWidget QLineEdit:focus { border-radius: 0px; "
                    "border: none; }"
                    "QTableWidget QLineEdit:hover { border-radius: 0px; }")
                    .arg(background, textColor, border, selection));
    m_table = grid;
    grid->setObjectName("PpocrTableGrid");
    grid->setSelectionMode(QAbstractItemView::ExtendedSelection);
    grid->setEditTriggers(QAbstractItemView::DoubleClicked |
                          QAbstractItemView::EditKeyPressed |
                          QAbstractItemView::AnyKeyPressed);
    grid->setWordWrap(false);
    grid->setTextElideMode(Qt::ElideNone);
    grid->setCornerButtonEnabled(false);
    grid->horizontalHeader()->hide();
    grid->verticalHeader()->hide();
    grid->horizontalHeader()->setMinimumSectionSize(1);
    grid->horizontalHeader()->setSectionResizeMode(QHeaderView::Fixed);
    grid->verticalHeader()->setSectionResizeMode(QHeaderView::Fixed);
    auto tableFont = grid->font();
    tableFont.setPixelSize(13);
    grid->setFont(tableFont);
    PpocrTable::load(grid, text);
    // The source bounds history to 120 serialized table states.
    auto history = std::make_shared<QStringList>(QStringList{content()});
    auto index = std::make_shared<int>(0);
    auto record = [this, grid, history, index]() {
        const auto value = content();
        if (grid->changed) grid->changed();
        if (history->value(*index) == value) return;
        while (history->size() > *index + 1) history->removeLast();
        history->append(value);
        if (history->size() > 120) history->removeFirst();
        *index = history->size() - 1;
    };
    auto edit = [grid, record](const std::function<void()>& operation) {
        {
            const QSignalBlocker blocked(grid);
            operation();
        }
        record();
    };
    auto anchors = [grid]() {
        const auto owner = owners(grid);
        QSet<Cell> result;
        for (const auto& index : grid->selectionModel()->selectedIndexes())
            result.insert(owner.value({index.row(), index.column()},
                                      {index.row(), index.column()}));
        if (result.isEmpty() && grid->currentRow() >= 0)
            result.insert(
                    owner.value({grid->currentRow(), grid->currentColumn()},
                                {grid->currentRow(), grid->currentColumn()}));
        return result;
    };
    auto action = [this, toolbar, selection, dark, secondary](
                          const char* name, const char* label, int style,
                          const std::function<void()>& callback) {
        auto* button = new QToolButton(this);
        button->setObjectName(name);
        button->setToolTip(tr(label));
        button->setCheckable(true);
        button->setFocusPolicy(Qt::NoFocus);
        button->setCursor(Qt::PointingHandCursor);
        button->setIcon(
                formatIcon(style, QColor(dark ? secondary : "#6b6f76")));
        button->setIconSize(QSize(20, 20));
        button->setStyleSheet(
                QString("QToolButton { background: transparent; border: none; "
                        "border-radius: 8px; padding: 4px; min-width: 32px; "
                        "min-height: 32px; } QToolButton:hover { "
                        "background-color: %1; } QToolButton:checked { "
                        "background-color: %1; }")
                        .arg(selection));
        toolbar->addWidget(button);
        connect(button, &QToolButton::clicked, this, callback);
    };
    for (int style = 0; style < 3; ++style)
        action(style == 0   ? "PpocrTableBold"
               : style == 1 ? "PpocrTableItalic"
                            : "PpocrTableStrike",
               style == 0   ? "Bold"
               : style == 1 ? "Italic"
                            : "Strikethrough",
               style, [grid, edit, anchors, style]() {
                   const auto cells = anchors();
                   bool all = !cells.isEmpty();
                   for (const auto& cell : cells) {
                       const auto font =
                               ensureItem(grid, cell.first, cell.second)
                                       ->font();
                       all &= style == 0   ? font.bold()
                              : style == 1 ? font.italic()
                                           : font.strikeOut();
                   }
                   edit([&]() {
                       for (const auto& cell : cells) {
                           auto* item =
                                   ensureItem(grid, cell.first, cell.second);
                           auto font = item->font();
                           if (style == 0) font.setBold(!all);
                           if (style == 1) font.setItalic(!all);
                           if (style == 2) font.setStrikeOut(!all);
                           item->setFont(font);
                       }
                   });
               });
    toolbar->addStretch();
    grid->undo = [grid, history, index]() {
        if (*index > 0) PpocrTable::load(grid, history->at(--*index));
        if (grid->changed) grid->changed();
    };
    grid->redo = [grid, history, index]() {
        if (*index + 1 < history->size())
            PpocrTable::load(grid, history->at(++*index));
        if (grid->changed) grid->changed();
    };
    auto* cancel = new QPushButton(tr("Cancel"), toolbarFrame);
    cancel->setObjectName("PpocrTableCancel");
    cancel->setAutoDefault(false);
    cancel->setStyleSheet(
            QString("QPushButton { background: %1; color: %2; border: 1px "
                    "solid "
                    "%3; border-radius: 8px; padding: 8px 16px; font-weight: "
                    "600; } QPushButton:hover { color: rgb(70, 88, 255); "
                    "border-color: rgb(70, 88, 255); } QPushButton:disabled { "
                    "color: %4; }")
                    .arg(background, textColor, border, secondary));
    auto* save = new QPushButton(tr("Save"), toolbarFrame);
    save->setObjectName("PpocrTableSave");
    save->setAutoDefault(false);
    save->setStyleSheet(
            "QPushButton { background: rgb(70, 88, 255); color: white; "
            "border: none; border-radius: 8px; padding: 8px 16px; "
            "font-weight: 600; } QPushButton:hover { background: rgb(53, 71, "
            "235); } QPushButton:disabled { background: rgb(180, 186, 220); }");
    connect(cancel, &QPushButton::clicked, this,
            &PpocrTableEditor::cancelRequested);
    connect(save, &QPushButton::clicked, this,
            [this]() { emit saveRequested(content()); });
    toolbar->addWidget(cancel);
    toolbar->addSpacing(12);
    toolbar->addWidget(save);
    bodyLayout->addWidget(grid);
    layout->addWidget(toolbarFrame);
    layout->addWidget(body);
    connect(grid, &QTableWidget::itemChanged, this, [record]() { record(); });
    auto create = [grid, edit]() {
        if (grid->rowCount() > 0 && grid->columnCount() > 0) return;
        edit([&]() {
            grid->clearSpans();
            grid->clearContents();
            grid->setRowCount(3);
            grid->setColumnCount(3);
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c) ensureItem(grid, r, c);
            for (int c = 0; c < 3; ++c) {
                auto* item = ensureItem(grid, 0, c);
                item->setText(QString("Title %1").arg(c + 1));
                auto font = item->font();
                font.setBold(true);
                item->setFont(font);
            }
            grid->setCurrentCell(0, 0);
        });
    };
    grid->copy = [grid]() {
        const auto ranges = grid->selectedRanges();
        if (ranges.isEmpty()) return;
        const auto range = ranges.first();
        const auto owner = owners(grid);
        QStringList lines;
        for (int r = range.topRow(); r <= range.bottomRow(); ++r) {
            QStringList row;
            for (int c = range.leftColumn(); c <= range.rightColumn(); ++c)
                row.append(owner.value({r, c}, {r, c}) == Cell(r, c)
                                   ? ensureItem(grid, r, c)->text()
                                   : QString());
            lines.append(row.join('\t'));
        }
        QApplication::clipboard()->setText(lines.join('\n'));
    };
    grid->paste = [grid, edit, create]() {
        const auto text = QApplication::clipboard()->text();
        if (text.isEmpty()) return;
        // Source records default-table creation separately from the paste.
        if (!grid->rowCount() && !grid->columnCount()) create();
        if (!grid->rowCount() || !grid->columnCount()) return;
        edit([&]() {
            const auto owner = owners(grid);
            QSet<Cell> touched;
            const int startRow = std::max(0, grid->currentRow()),
                      startColumn = std::max(0, grid->currentColumn());
            auto rows = text.split(
                    QRegularExpression("\\r\\n|[\\n\\r\\x{b}\\x{c}\\x{1c}-\\x{"
                                       "1e}\\x{85}\\x{2028}\\x{2029}]"));
            if (rows.last().isEmpty()) rows.removeLast();  // Python splitlines.
            for (int r = 0; r < rows.size() && startRow + r < grid->rowCount();
                 ++r) {
                const auto cells = rows[r].split('\t');
                for (int c = 0;
                     c < cells.size() && startColumn + c < grid->columnCount();
                     ++c) {
                    const Cell cell{startRow + r, startColumn + c};
                    const auto anchor = owner.value(cell, cell);
                    if (touched.contains(anchor)) continue;
                    ensureItem(grid, anchor.first, anchor.second)
                            ->setText(cells[c]);
                    touched.insert(anchor);
                }
            }
        });
    };
    grid->setContextMenuPolicy(Qt::CustomContextMenu);
    connect(grid, &QWidget::customContextMenuRequested, this,
            [this, grid, edit, anchors, create](const QPoint& point) {
                const auto clicked = grid->indexAt(point);
                grid->restoreRightClickSelection(clicked);
                if (clicked.isValid() &&
                    grid->selectionModel()->selectedIndexes().size() <= 1)
                    grid->setCurrentCell(clicked.row(), clicked.column());
                QMenu menu(grid);
                const bool hasTable =
                        grid->rowCount() > 0 && grid->columnCount() > 0;
                auto add = [&](const char* label,
                               const std::function<void()>& call) {
                    auto* action = menu.addAction(tr(label), this, call);
                    action->setEnabled(hasTable);
                    return action;
                };
                add("Create 3x3 Table", create)->setEnabled(!hasTable);
                menu.addSeparator();
                add("Insert Row Above", [grid, edit]() {
                    if (grid->rowCount() && grid->columnCount())
                        edit([&]() {
                            grid->insertRow(std::max(0, grid->currentRow()));
                        });
                });
                add("Insert Row Below", [grid, edit]() {
                    if (grid->rowCount() && grid->columnCount())
                        edit([&]() {
                            grid->insertRow(grid->currentRow() < 0
                                                    ? grid->rowCount()
                                                    : grid->currentRow() + 1);
                        });
                });
                add("Delete Row", [grid, edit]() {
                    if (!grid->rowCount()) return;
                    edit([&]() {
                        grid->removeRow(grid->currentRow() < 0
                                                ? grid->rowCount() - 1
                                                : grid->currentRow());
                        if (!grid->rowCount()) grid->setColumnCount(0);
                    });
                });
                menu.addSeparator();
                add("Insert Column Left", [grid, edit]() {
                    if (grid->rowCount() && grid->columnCount())
                        edit([&]() {
                            grid->insertColumn(
                                    std::max(0, grid->currentColumn()));
                        });
                });
                add("Insert Column Right", [grid, edit]() {
                    if (grid->rowCount() && grid->columnCount())
                        edit([&]() {
                            grid->insertColumn(grid->currentColumn() < 0
                                                       ? grid->columnCount()
                                                       : grid->currentColumn() +
                                                                 1);
                        });
                });
                add("Delete Column", [grid, edit]() {
                    if (!grid->columnCount()) return;
                    edit([&]() {
                        grid->removeColumn(grid->currentColumn() < 0
                                                   ? grid->columnCount() - 1
                                                   : grid->currentColumn());
                        if (!grid->columnCount()) grid->setRowCount(0);
                    });
                });
                menu.addSeparator();
                int top = grid->rowCount(), left = grid->columnCount(),
                    bottom = -1, right = -1;
                QSet<Cell> cells;
                for (const auto& item :
                     grid->selectionModel()->selectedIndexes()) {
                    const Cell cell{item.row(), item.column()};
                    cells.insert(cell);
                    top = std::min(top, cell.first);
                    left = std::min(left, cell.second);
                    bottom = std::max(bottom, cell.first);
                    right = std::max(right, cell.second);
                }
                bool canMerge =
                        cells.size() > 1 &&
                        cells.size() == (bottom - top + 1) * (right - left + 1);
                const auto owner = owners(grid);
                for (const auto& cell : cells) {
                    const auto anchor = owner.value(cell, cell);
                    canMerge &=
                            anchor.first >= top && anchor.second >= left &&
                            anchor.first +
                                            grid->rowSpan(anchor.first,
                                                          anchor.second) -
                                            1 <=
                                    bottom &&
                            anchor.second +
                                            grid->columnSpan(anchor.first,
                                                             anchor.second) -
                                            1 <=
                                    right;
                }
                auto* merge = menu.addAction(tr("Merge Cells"), this, [=]() {
                    edit([&]() {
                        for (const auto& cell : cells) {
                            if (cell == Cell(top, left)) continue;
                            auto* item =
                                    ensureItem(grid, cell.first, cell.second);
                            item->setText("");
                            item->setFont(grid->font());
                        }
                        grid->setSpan(top, left, bottom - top + 1,
                                      right - left + 1);
                        grid->clearSelection();
                        grid->setCurrentCell(top, left);
                    });
                });
                merge->setEnabled(canMerge);
                Cell split{-1, -1};
                auto ordered = anchors().values();
                std::sort(ordered.begin(), ordered.end());
                for (const auto& cell : ordered)
                    if (grid->rowSpan(cell.first, cell.second) > 1 ||
                        grid->columnSpan(cell.first, cell.second) > 1) {
                        split = cell;
                        break;
                    }
                auto* unmerge = menu.addAction(tr("Split Cells"), this, [=]() {
                    edit([&]() {
                        const int h = grid->rowSpan(split.first, split.second),
                                  w = grid->columnSpan(split.first,
                                                       split.second);
                        grid->setSpan(split.first, split.second, 1, 1);
                        for (int r = split.first; r < split.first + h; ++r)
                            for (int c = split.second; c < split.second + w;
                                 ++c)
                                if (Cell(r, c) != split) {
                                    auto* item = ensureItem(grid, r, c);
                                    item->setText("");
                                    item->setFont(grid->font());
                                }
                        grid->clearSelection();
                        grid->setCurrentCell(split.first, split.second);
                    });
                });
                unmerge->setEnabled(split.first >= 0);
                menu.exec(grid->viewport()->mapToGlobal(point));
            });
    auto syncActions = [this, grid, anchors]() {
        const auto cells = anchors();
        const bool multi = grid->selectionModel()->selectedIndexes().size() > 1;
        if (grid->property("ppocrMultiSelect").toBool() != multi) {
            grid->setProperty("ppocrMultiSelect", multi);
            grid->style()->unpolish(grid);
            grid->style()->polish(grid);
            grid->viewport()->update();
        }
        const char* names[] = {"PpocrTableBold", "PpocrTableItalic",
                               "PpocrTableStrike"};
        for (int style = 0; style < 3; ++style) {
            auto* button = findChild<QToolButton*>(names[style]);
            bool checked = !cells.isEmpty();
            for (const auto& cell : cells) {
                const auto* item = grid->item(cell.first, cell.second);
                if (!item) {
                    checked = false;
                    break;
                }
                const auto font = item->font();
                checked &= style == 0   ? font.bold()
                           : style == 1 ? font.italic()
                                        : font.strikeOut();
            }
            const QSignalBlocker blocked(button);
            button->setCheckable(true);
            button->setEnabled(!cells.isEmpty());
            button->setChecked(checked);
        }
    };
    grid->changed = [this, grid, syncActions]() {
        syncTableGeometry(grid);
        syncActions();
        emit geometryChanged();
    };
    connect(grid, &QTableWidget::itemSelectionChanged, this, syncActions);
    connect(grid, &QTableWidget::currentCellChanged, this,
            [syncActions]() { syncActions(); });
    auto shortcut = [this](const QKeySequence& sequence,
                           const std::function<void()>& callback) {
        auto* binding = new QShortcut(sequence, this);
        connect(binding, &QShortcut::activated, this, callback);
    };
    const QPair<const char*, const char*> formatBindings[] = {
            {"Ctrl+B", "PpocrTableBold"},
            {"Ctrl+I", "PpocrTableItalic"},
            {"Ctrl+Shift+X", "PpocrTableStrike"}};
    for (const auto& binding : formatBindings)
        shortcut(QKeySequence(binding.first), [this, binding]() {
            findChild<QToolButton*>(binding.second)->click();
        });
    shortcut(QKeySequence("Ctrl+S"), [save]() { save->click(); });
    shortcut(QKeySequence(QKeySequence::Undo), grid->undo);
    const QKeySequence redo(QKeySequence::Redo);
    shortcut(redo, grid->redo);
    // Avoid ambiguous duplicate shortcuts on platforms where Redo is Ctrl+Y.
    if (redo != QKeySequence("Ctrl+Y"))
        shortcut(QKeySequence("Ctrl+Y"), grid->redo);
    grid->changed();
}
QString PpocrTableEditor::content() const {
    return PpocrTable::tokens(m_table);
}
void PpocrTableEditor::scheduleGeometry() {
    if (m_geometryPending) return;
    m_geometryPending = true;
    QTimer::singleShot(0, this, [this]() {
        m_geometryPending = false;
        syncTableGeometry(m_table);
        emit geometryChanged();
    });
}
void PpocrTableEditor::resizeEvent(QResizeEvent* event) {
    QFrame::resizeEvent(event);
    scheduleGeometry();
}
void PpocrTableEditor::showEvent(QShowEvent* event) {
    QFrame::showEvent(event);
    scheduleGeometry();
}
