// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <QApplication>
#include <QClipboard>
#include <QFile>
#include <QFrame>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QKeyEvent>
#include <QLayout>
#include <QMenu>
#include <QMouseEvent>
#include <QPushButton>
#include <QRandomGenerator>
#include <QScrollArea>
#include <QScrollBar>
#include <QShortcut>
#include <QTableWidget>
#include <QTimer>
#include <QToolButton>
#include <cstdio>

#include "PpocrTableEditor.h"
#include "QtCompat.h"
#define REQUIRE(condition)                                               \
    do {                                                                 \
        if (!(condition)) {                                              \
            std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); \
            return 1;                                                    \
        }                                                                \
    } while (false)
namespace {
QJsonObject snapshot(QTableWidget* table) {
    QJsonArray states;
    for (int r = 0; r < table->rowCount(); ++r)
        for (int c = 0; c < table->columnCount(); ++c) {
            auto* item = table->item(r, c);
            const auto font = item ? item->font() : table->font();
            states.append(QJsonObject{{"row", r},
                                      {"column", c},
                                      {"text", item ? item->text() : QString()},
                                      {"bold", font.bold()},
                                      {"italic", font.italic()},
                                      {"strike", font.strikeOut()},
                                      {"rows", table->rowSpan(r, c)},
                                      {"columns", table->columnSpan(r, c)}});
        }
    return {{"rows", table->rowCount()},
            {"columns", table->columnCount()},
            {"states", states},
            {"tokens", PpocrTable::tokens(table)}};
}
void key(QTableWidget* grid, int code, Qt::KeyboardModifiers modifiers) {
    QKeyEvent event(QEvent::KeyPress, code, modifiers);
    QCoreApplication::sendEvent(grid, &event);
}
bool menu(QTableWidget* grid, const QString& action) {
    bool triggered = false;
    QTimer::singleShot(0, [&]() {
        auto* popup = qobject_cast<QMenu*>(QApplication::activePopupWidget());
        if (!popup) return;
        for (auto* item : popup->actions())
            if (item->text() == action && item->isEnabled()) {
                item->trigger();
                triggered = true;
                break;
            }
        popup->close();
    });
    QMetaObject::invokeMethod(grid, "customContextMenuRequested",
                              Qt::DirectConnection,
                              Q_ARG(QPoint, QPoint(10, 10)));
    return triggered;
}
}  // namespace
int main(int argc, char** argv) {
    QApplication app(argc, argv);
    QStringList inputs{
            "",
            "plain text",
            "one\ttwo\nthree\tfour",
            "a\t\tb\n\nc",
            "<nl>",
            "<ecel><nl><nl>",
            "<fcel>A&amp;B<ecel><nl>",
            "<fcel><b><i><s>中文 &lt;x&gt; &#x27;</s></i></b><nl>",
            "<fcel>&nbsp;&copy;&#x1f600;&quot;&unknown;</fcel><nl>",
            "<fcel>&NotEqualTilde;&acE;&notit;&AMPfoo;&#128;&#0;&#xD800;&#"
            "xFFFF;<nl>",
            "<fcel>A<lcel><nl><ucel><xcel><nl>",
            "<fcel>A<lcel><fcel>C<nl><ucel><ecel><nl>",
            "<lcel><ucel><xcel><nl>",
            "<FCEL><STRONG>Bold</STRONG><NL>",
            "<table><tr><td>A &amp; "
            "B</td><td><em>C</em><br>D</td></tr></table>",
            "<TABLE><TR><TH>Heading</TH><TD><b><del>value</del></b></TD></TR></"
            "TABLE>",
            "<table><tr><td rowspan='2' "
            "colspan=2>anchor</td><td>X</td></tr><tr><td>Y</td></tr></table>",
            "<table><tr><td colspan='invalid'>A</td><td "
            "rowspan='0'>B</td></tr></table>",
            "<table><tr><td title='a>b'>x</td><td>&nbsp; y</td></tr></table>",
            "<table><tr><td><!-- ignored "
            "-->&lt;b&gt;literal&lt;/b&gt;</td></tr></table>",
            "<table><tr><td>unclosed</tr></table>",
            "<table><tr></tr><tr><td>A</td></tr></table>"};
    inputs.append(
            QString::fromUtf8("a\vb\fc\x1c"
                              "d\x1d"
                              "e\x1e"
                              "f\xc2\x85"
                              "g\xe2\x80\xa8"
                              "h\xe2\x80\xa9"
                              "i"));
    if (argc >= 3) {
        QFile fixture(QString::fromLocal8Bit(argv[2]));
        REQUIRE(fixture.open(QIODevice::ReadOnly));
        const auto values = QJsonDocument::fromJson(fixture.readAll()).array();
        REQUIRE(!values.isEmpty());
        for (const auto& value : values) inputs.append(value.toString());
    }
    QRandomGenerator random(41875);
    const QStringList payloads{"A",        "中文",     "&amp;", "<b>B</b>",
                               "<i>I</i>", "<s>S</s>", "",      "two words"};
    for (int sample = 0; sample < 100; ++sample) {
        QString text;
        const int rows = 1 + random.bounded(4), columns = 1 + random.bounded(5);
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < columns; ++c) {
                const auto p = payloads[random.bounded(payloads.size())];
                text += p.isEmpty() ? QString("<ecel>") : QString("<fcel>" + p);
            }
            text += "<nl>";
        }
        inputs.append(text);
    }
    QJsonArray cases;
    for (const auto& input : inputs) {
        QTableWidget table;
        PpocrTable::load(&table, input);
        cases.append(
                QJsonObject{{"input", input}, {"output", snapshot(&table)}});
        QTableWidget replay;
        PpocrTable::load(&replay, PpocrTable::tokens(&table));
        // Source strips literal leading NBSP on reopen, after decoding escaped
        // NBSP on the first parse. Compare both passes to source, not an
        // invented byte-idempotence requirement.
        auto value = cases.last().toObject();
        value.insert("replay", snapshot(&replay));
        cases[cases.size() - 1] = value;
    }
    if (argc >= 2) {
        QFile file(QString::fromLocal8Bit(argv[1]) +
                   "/ppocr-table-native.json");
        REQUIRE(file.open(QIODevice::WriteOnly));
        file.write(QJsonDocument(QJsonObject{{"cases", cases}}).toJson());
    }
    REQUIRE(PpocrTable::isTable(" TABLE ", "") &&
            PpocrTable::isTable("text", "<ecel><nl>") &&
            !PpocrTable::isTable("text", "ordinary"));
    for (const auto& label :
         {"display-formula", "FORMULA NUMBER", "algorithm", "formula"})
        REQUIRE(!PpocrTable::isTable(label, "<fcel>not a table<nl>"));
    PpocrTableEditor editor("<fcel>A<ecel><nl><fcel>C<ecel><nl>");
    bool saved = false;
    QObject::connect(&editor, &PpocrTableEditor::saveRequested,
                     [&saved]() { saved = true; });
    editor.show();
    QCoreApplication::processEvents();
    auto* grid = editor.findChild<QTableWidget*>("PpocrTableGrid");
    REQUIRE(grid && grid->rowCount() == 2 && grid->columnCount() == 2);
    REQUIRE(grid->font().pixelSize() == 13 && !grid->wordWrap());
    QJsonArray styleCases;
    for (bool dark : {false, true}) {
        QWidget parent;
        auto palette = parent.palette();
        palette.setColor(QPalette::Window, dark ? Qt::black : Qt::white);
        parent.setPalette(palette);
        PpocrTableEditor view("<fcel>format<ecel><nl>", &parent);
        parent.resize(900, 500);
        parent.show();
        view.resize(900, 500);
        view.show();
        QCoreApplication::processEvents();
        auto* table = view.findChild<QTableWidget*>("PpocrTableGrid");
        auto* frame = &view;
        auto* toolbar = view.findChild<QFrame*>("PPOCRTableBlockEditorToolbar");
        auto* body = view.findChild<QFrame*>("PPOCRTableBlockEditorBody");
        auto* bold = view.findChild<QToolButton*>("PpocrTableBold");
        auto* cancel = view.findChild<QPushButton*>("PpocrTableCancel");
        auto* save = view.findChild<QPushButton*>("PpocrTableSave");
        REQUIRE(bold->text().isEmpty() && !bold->icon().isNull() &&
                bold->iconSize() == QSize(20, 20) &&
                bold->focusPolicy() == Qt::NoFocus);
        REQUIRE(cancel->parentWidget() == toolbar &&
                save->parentWidget() == toolbar && cancel->x() < save->x() &&
                toolbar->y() < body->y());
        const auto margins = body->layout()->contentsMargins();
        REQUIRE(margins.left() == 16 && margins.top() == 16 &&
                margins.right() == 16 && margins.bottom() == 16);
        styleCases.append(QJsonObject{{"mode", dark ? "dark" : "light"},
                                      {"frame", frame->styleSheet()},
                                      {"table", table->styleSheet()},
                                      {"format", bold->styleSheet()},
                                      {"cancel", cancel->styleSheet()},
                                      {"save", save->styleSheet()}});
        if (argc >= 2)
            REQUIRE(view.grab().save(QString::fromLocal8Bit(argv[1]) +
                                     (dark ? "/ppocr-table-dark.png"
                                           : "/ppocr-table-light.png")));
        auto binding = [&view](const QKeySequence& sequence) -> QShortcut* {
            QShortcut* found = nullptr;
            for (auto* item : view.findChildren<QShortcut*>())
                if (item->key() == sequence) {
                    if (found) return nullptr;  // An ambiguous duplicate.
                    found = item;
                }
            return found;
        };
        for (const auto& sequence :
             {QKeySequence("Ctrl+B"), QKeySequence("Ctrl+I"),
              QKeySequence("Ctrl+Shift+X"), QKeySequence("Ctrl+S"),
              QKeySequence(QKeySequence::Undo),
              QKeySequence(QKeySequence::Redo), QKeySequence("Ctrl+Y")})
            REQUIRE(binding(sequence));
        table->setCurrentCell(0, 0);
        REQUIRE(QMetaObject::invokeMethod(binding(QKeySequence("Ctrl+B")),
                                          "activated", Qt::DirectConnection));
        REQUIRE(table->item(0, 0)->font().bold());
        REQUIRE(QMetaObject::invokeMethod(
                binding(QKeySequence(QKeySequence::Undo)), "activated",
                Qt::DirectConnection));
        REQUIRE(!table->item(0, 0)->font().bold());
        bool requested = false;
        QObject::connect(&view, &PpocrTableEditor::saveRequested,
                         [&requested]() { requested = true; });
        REQUIRE(QMetaObject::invokeMethod(binding(QKeySequence("Ctrl+S")),
                                          "activated", Qt::DirectConnection));
        REQUIRE(requested && !view.isWindow());
    }
    QJsonArray heightCases;
    for (int height : {180, 500}) {
        for (int rows : {0, 1, 40}) {
            QScrollArea host;
            host.setWidgetResizable(true);
            host.resize(380, height);
            QString input;
            for (int r = 0; r < rows; ++r)
                input += "<fcel>" + QString(120, 'W') + "<ecel><nl>";
            auto* view = new PpocrTableEditor(input);
            host.setWidget(view);
            host.show();
            for (int i = 0; i < 8; ++i) QCoreApplication::processEvents();
            auto* table = view->findChild<QTableWidget*>("PpocrTableGrid");
            int width = 0;
            for (int c = 0; c < table->columnCount(); ++c)
                width += table->columnWidth(c);
            heightCases.append(QJsonObject{
                    {"rows", rows},
                    {"font_height", table->fontMetrics().height()},
                    {"frame_width", table->frameWidth()},
                    {"show_grid", table->showGrid()},
                    {"horizontal_scrollbar",
                     table->columnCount() > 0 &&
                             width > table->viewport()->width()},
                    {"scrollbar_height",
                     table->horizontalScrollBar()->sizeHint().height()},
                    {"viewport_height", host.viewport()->height()},
                    {"table_top",
                     table->mapTo(host.viewport(), QPoint(0, 0)).y()},
                    {"minimum_height", table->minimumHeight()},
                    {"maximum_height", table->maximumHeight()}});
            REQUIRE(table->minimumHeight() == table->maximumHeight());
            if (rows == 40)
                REQUIRE(table->height() < rows * table->rowHeight(0));
        }
    }
    QJsonArray geometryCases;
    for (int width : {380, 900}) {
        for (const auto& input : QStringList{
                     "<fcel>short<ecel><nl><fcel>" + QString(120, 'W') +
                             "<ecel><nl>",
                     "<fcel>span<lcel><fcel>right<nl><ucel><xcel><ecel><nl>",
                     "<fcel>one<ecel><fcel>three<ecel><fcel>five<nl>"}) {
            PpocrTableEditor view(input);
            view.resize(width, 500);
            view.show();
            QCoreApplication::processEvents();
            QCoreApplication::processEvents();
            auto* table = view.findChild<QTableWidget*>("PpocrTableGrid");
            QJsonObject advances;
            for (int r = 0; r < table->rowCount(); ++r)
                for (int c = 0; c < table->columnCount(); ++c) {
                    const auto text = table->item(r, c)->text();
                    advances[text] = QTCOMPAT_FONTMETRICS_WIDTH(
                            table->fontMetrics(), text);
                }
            QJsonArray columns, rows;
            for (int c = 0; c < table->columnCount(); ++c)
                columns.append(table->columnWidth(c));
            for (int r = 0; r < table->rowCount(); ++r)
                rows.append(table->rowHeight(r));
            geometryCases.append(
                    QJsonObject{{"input", input},
                                {"output", snapshot(table)},
                                {"viewport", table->viewport()->width()},
                                {"font_height", table->fontMetrics().height()},
                                {"advances", advances},
                                {"widths", columns},
                                {"heights", rows}});
        }
    }
    const auto original = editor.content();
    grid->setRangeSelected(QTableWidgetSelectionRange(0, 0, 1, 1), true);
    REQUIRE(grid->property("ppocrMultiSelect").toBool());
    editor.findChild<QToolButton*>("PpocrTableBold")->click();
    REQUIRE(editor.content().contains("<b>A</b>") &&
            editor.content().contains("<b>C</b>"));
    REQUIRE(editor.findChild<QToolButton*>("PpocrTableBold")->isChecked());
    key(grid, Qt::Key_Z, Qt::ControlModifier);
    REQUIRE(editor.content() == original);
    REQUIRE(!editor.findChild<QToolButton*>("PpocrTableBold")->isChecked());
    key(grid, Qt::Key_Y, Qt::ControlModifier);
    REQUIRE(editor.content().contains("<b>A</b>"));
    grid->setRangeSelected(QTableWidgetSelectionRange(0, 0, 1, 1), true);
    const auto point = grid->visualRect(grid->model()->index(0, 0)).center();
    QMouseEvent press(QEvent::MouseButtonPress, point, Qt::RightButton,
                      Qt::RightButton, Qt::NoModifier);
    QCoreApplication::sendEvent(grid->viewport(), &press);
    // Exercise the upstream workaround when Qt collapses a multi-selection.
    grid->clearSelection();
    grid->setCurrentCell(0, 0);
    REQUIRE(menu(grid, "Merge Cells"));
    REQUIRE(grid->rowSpan(0, 0) == 2 && grid->columnSpan(0, 0) == 2 &&
            editor.content() == "<fcel><b>A</b><lcel><nl><ucel><xcel><nl>");
    QApplication::clipboard()->setText("P\tQ\nR\tS\n");
    key(grid, Qt::Key_V, Qt::ControlModifier);
    REQUIRE(editor.content() == "<fcel><b>P</b><lcel><nl><ucel><xcel><nl>");
    grid->setRangeSelected(QTableWidgetSelectionRange(0, 0, 1, 1), true);
    key(grid, Qt::Key_C, Qt::ControlModifier);
    REQUIRE(QApplication::clipboard()->text() == "P\t\n\t");
    REQUIRE(menu(grid, "Split Cells"));
    REQUIRE(grid->rowSpan(0, 0) == 1 && grid->columnSpan(0, 0) == 1 &&
            editor.content() == "<fcel><b>P</b><ecel><nl><ecel><ecel><nl>");
    REQUIRE(menu(grid, "Insert Row Below") && grid->rowCount() == 3);
    REQUIRE(menu(grid, "Insert Column Right") && grid->columnCount() == 3);
    REQUIRE(menu(grid, "Delete Row") && grid->rowCount() == 2);
    REQUIRE(menu(grid, "Delete Column") && grid->columnCount() == 2);
    key(grid, Qt::Key_Z, Qt::ControlModifier);
    REQUIRE(grid->columnCount() == 3);
    grid->setCurrentCell(0, 0);
    grid->item(0, 0)->setText("new branch");
    const auto branched = editor.content();
    key(grid, Qt::Key_Y, Qt::ControlModifier);
    REQUIRE(editor.content() == branched);
    PpocrTableEditor empty("");
    auto* emptyGrid = empty.findChild<QTableWidget*>("PpocrTableGrid");
    QApplication::clipboard()->setText("pasted");
    key(emptyGrid, Qt::Key_V, Qt::ControlModifier);
    REQUIRE(emptyGrid->rowCount() == 3 && emptyGrid->columnCount() == 3 &&
            empty.content().startsWith(
                    "<fcel><b>pasted</b><fcel><b>Title 2</b>"));
    key(emptyGrid, Qt::Key_Z, Qt::ControlModifier);
    REQUIRE(empty.content().startsWith("<fcel><b>Title 1</b>"));
    key(emptyGrid, Qt::Key_Z, Qt::ControlModifier);
    REQUIRE(emptyGrid->rowCount() == 0 && emptyGrid->columnCount() == 0);
    REQUIRE(menu(emptyGrid, "Create 3x3 Table"));
    REQUIRE(emptyGrid->rowCount() == 3 && emptyGrid->columnCount() == 3);
    key(emptyGrid, Qt::Key_Z, Qt::ControlModifier);
    REQUIRE(emptyGrid->rowCount() == 0 && emptyGrid->columnCount() == 0);
    if (argc >= 2) {
        QFile file(QString::fromLocal8Bit(argv[1]) +
                   "/ppocr-table-native.json");
        REQUIRE(file.open(QIODevice::WriteOnly));
        file.write(QJsonDocument(QJsonObject{{"cases", cases},
                                             {"geometry_cases", geometryCases},
                                             {"height_cases", heightCases},
                                             {"style_cases", styleCases},
                                             {"operations", snapshot(grid)}})
                           .toJson());
        REQUIRE(editor.grab().save(QString::fromLocal8Bit(argv[1]) +
                                   "/ppocr-table.png"));
    }
    editor.findChild<QPushButton*>("PpocrTableSave")->click();
    REQUIRE(saved);
    std::puts(
            "PASS: OCR table tokens/HTML/TSV, format/span/history, clipboard, "
            "structure");
    return 0;
}
