// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <QAction>
#include <QApplication>
#include <QButtonGroup>
#include <QCheckBox>
#include <QClipboard>
#include <QCloseEvent>
#include <QComboBox>
#include <QCoreApplication>
#include <QDateTime>
#include <QDialog>
#include <QDialogButtonBox>
#include <QDir>
#include <QDoubleSpinBox>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QFrame>
#include <QGraphicsDropShadowEffect>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QImageReader>
#include <QInputDialog>
#include <QIntValidator>
#include <QJsonDocument>
#include <QKeyEvent>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QMenu>
#include <QMessageBox>
#include <QMouseEvent>
#include <QPlainTextEdit>
#include <QPointer>
#include <QProgressDialog>
#include <QPushButton>
#include <QRadioButton>
#include <QRegularExpression>
#include <QSaveFile>
#include <QScrollArea>
#include <QScrollBar>
#include <QSet>
#include <QSettings>
#include <QShortcut>
#include <QSlider>
#include <QSpinBox>
#include <QSplitter>
#include <QTabBar>
#include <QTabWidget>
#include <QTableWidget>
#include <QTemporaryDir>
#include <QTextBrowser>
#include <QTextCharFormat>
#include <QTextDocument>
#include <QTextEdit>
#include <QTimer>
#include <QVBoxLayout>
#include <algorithm>
#include <cmath>
#include <functional>
#include <iomanip>
#include <locale>
#include <sstream>

#include "AnnotationArchive.h"
#include "AnnotationChatConfig.h"
#include "AnnotationDocument.h"
#include "AnnotationOrderedJson.h"
#include "AnnotationWindow.h"
#include "ChatPromptDialog.h"
#include "ChatServiceClient.h"
#include "QtCompat.h"

namespace {
QString chatTr(const char* text) {
    return QCoreApplication::translate("AnnotationChat", text);
}

QString chatStyle(const char* name, bool dark) {
    static const struct {
        const char* name;
        const char* light;
        const char* dark;
    } styles[] = {
#include "AnnotationChatStyles.inc"
    };
    for (const auto& style : styles)
        if (qstrcmp(style.name, name) == 0)
            return QString::fromUtf8(dark ? style.dark : style.light);
    return {};
}

QString vqaPythonString(const QString& text) {
    static constexpr struct {
        uint first, last;
    } nonPrintable[] = {
#include "AnnotationPythonUnicode.inc"
    };
    const QChar quote = text.contains('\'') && !text.contains('"') ? '"' : '\'';
    QString result(quote);
    for (int i = 0; i < text.size(); ++i) {
        uint code = text[i].unicode();
        // QString::toUcs4 replaces unpaired surrogates. Python repr preserves
        // them as escaped code points, which can occur in escaped JSON input.
        if (QChar::isHighSurrogate(code) && i + 1 < text.size() &&
            text[i + 1].isLowSurrogate()) {
            code = QChar::surrogateToUcs4(text[i], text[i + 1]);
            ++i;
        }
        if (code == uint(quote.unicode()) || code == '\\') {
            result += '\\';
            result += QChar(code);
        } else if (code == '\n')
            result += "\\n";
        else if (code == '\r')
            result += "\\r";
        else if (code == '\t')
            result += "\\t";
        else {
            // O(log r) classification with r Unicode ranges; Qt 5 and Qt 6
            // ship different Unicode versions. Match the pinned CPython 3.11
            // reference's Unicode 14 categories on every build platform.
            const auto* range = std::lower_bound(
                    std::begin(nonPrintable), std::end(nonPrintable), code,
                    [](const auto& r, uint value) { return r.last < value; });
            const bool printable =
                    range == std::end(nonPrintable) || code < range->first;
            if (printable)
                result += QString::fromUcs4(&code, 1);
            else
                result += QString(code <= 0xff     ? "\\x%1"
                                  : code <= 0xffff ? "\\u%1"
                                                   : "\\U%1")
                                  .arg(code,
                                       code <= 0xff     ? 2
                                       : code <= 0xffff ? 4
                                                        : 8,
                                       16, QChar('0'));
        }
    }
    result += quote;
    return result;
}

QString vqaPythonValue(const QByteArray& input, bool nested = false) {
    const auto raw = input.trimmed();
    if (raw.isEmpty()) return {};
    if (raw.startsWith('"')) {
        const QString text =
                QJsonDocument::fromJson(QByteArray("[") + raw + "]")
                        .array()
                        .first()
                        .toString();
        return nested ? vqaPythonString(text) : text;
    }
    if (raw.startsWith('[') || raw.startsWith('{')) {
        QStringList items;
        if (raw.startsWith('[')) {
            for (const auto& value : AnnotationOrderedJson::items(raw))
                items.append(vqaPythonValue(value, true));
        } else {
            for (const auto& member : AnnotationOrderedJson::members(raw))
                items.append(QString(vqaPythonString(member.first) + ": " +
                                     vqaPythonValue(member.second, true)));
        }
        return QString(raw.startsWith('[') ? "[%1]" : "{%1}")
                .arg(items.join(", "));
    }
    if (raw == "null") return "None";
    if (raw == "true") return "True";
    if (raw == "false") return "False";
    if (!raw.contains('.') && !raw.contains('e') && !raw.contains('E'))
        return raw == "-0" ? QString("0") : QString::fromLatin1(raw);
    // Shortest round-trip digits, then Python's fixed/scientific cutoffs.
    const double value = raw.toDouble();
    QByteArray number;
    // Use locale-independent, ties-to-even formatting. Older toolchains lack
    // floating-point std::to_chars; Qt's precision formatter rounds ties up.
    for (int precision = 0; precision < 17; ++precision) {
        std::ostringstream stream;
        stream.imbue(std::locale::classic());
        stream << std::scientific << std::setprecision(precision) << value;
        number = QByteArray::fromStdString(stream.str());
        if (number.toDouble() == value) break;
    }
    if (value == 0.0 && std::signbit(value) && !number.startsWith('-'))
        number.prepend('-');
    const bool negative = number.startsWith('-');
    if (negative) number.remove(0, 1);
    const int e = number.indexOf('e');
    const int exponent = number.mid(e + 1).toInt();
    QByteArray digits = number.left(e);
    digits.replace(".", "");
    QString result;
    if (exponent >= -4 && exponent < 16) {
        const int decimal = exponent + 1;
        if (decimal <= 0)
            result = "0." + QString(-decimal, '0') + digits;
        else if (decimal >= digits.size())
            result = digits + QString(decimal - digits.size(), '0') + ".0";
        else
            result = digits.left(decimal) + "." + digits.mid(decimal);
    } else {
        result = QString::fromLatin1(digits.left(1));
        if (digits.size() > 1)
            result += "." + QString::fromLatin1(digits.mid(1));
        result += QString("e%1%2")
                          .arg(exponent < 0 ? '-' : '+')
                          .arg(std::abs(exponent), 2, 10, QChar('0'));
    }
    return negative ? QString("-" + result) : result;
}

QString vqaConfigPath(const QString& filename) {
    // Match get_work_directory()/xanylabeling_data/vqa in the source app.
    const QString work =
            QSettings()
                    .value("qAnyLabeling/workDirectory", QDir::homePath())
                    .toString();
    return QDir(work).filePath(
            QStringLiteral("xanylabeling_data/vqa/%1").arg(filename));
}

bool writeVqaJson(const QString& path,
                  const QByteArray& bytes,
                  QString* error) {
    if (!QDir().mkpath(QFileInfo(path).absolutePath())) {
        *error = chatTr("Cannot create the configuration directory.");
        return false;
    }
    QSaveFile file(path);
    if (!file.open(QIODevice::WriteOnly) || file.write(bytes) != bytes.size() ||
        !file.commit()) {
        *error = file.errorString();
        return false;
    }
    return true;
}

bool validVqaComponents(const QJsonDocument& json) {
    const auto components = json.object().value("components");
    if (!json.isObject() || !components.isArray()) return false;
    QSet<QString> titles;
    for (const auto& value : components.toArray()) {
        const auto c = value.toObject();
        const auto title = c.value("title").toString();
        if (!value.isObject() || title.trimmed().isEmpty() ||
            titles.contains(title) ||
            !QStringList{"QLineEdit", "QRadioButton", "QComboBox", "QCheckBox"}
                     .contains(c.value("type").toString()) ||
            (!c.value("options").isUndefined() &&
             !c.value("options").isArray()))
            return false;
        for (const auto& option : c.value("options").toArray())
            if (!option.isString()) return false;
        titles.insert(title);
    }
    return true;
}

// vqa/config.py DEFAULT_TEMPLATES, pinned source 745f6c355f94.
const QVector<QPair<QString, QString>>& defaultVqaTemplates() {
    static const QVector<QPair<QString, QString>> templates{
            {"Condense text",
             "Please make this text more concise while keeping the key points. "
             "Output the condensed version only:\n@text"},
            {"Academic style",
             "Please rewrite this text in a formal academic style. Return only "
             "the reformatted text:\n@text"},
            {"Clarify meaning",
             "Please break down and explain the meaning of this text. Provide "
             "only the explanation:\n@text"},
            {"Grammar check",
             "Please review and correct any grammatical errors in this text. "
             "Return only the corrected version:\n@text"},
            {"Enhance writing",
             "Please refine and polish this text to make it more effective. "
             "Return only the enhanced version:\n@text"},
            {"Brief summary",
             "Please provide a concise summary of the key points in this text. "
             "Return only the summary:\n@text"},
            {"Chinese translation",
             "Please convert this text to Chinese language. Return only the "
             "translated version:\n@text"},
            {"Image Description", "@image Describe this image in detail."},
            {"OCR", "@image Read all texts in the image, output in lines."},
            {"Text Recognition",
             "@image Recognise all texts in the image, output in lines."},
            {"Text Spotting",
             "@image Spotting all the text in the image with line-level, and "
             "output in JSON format."}};
    return templates;
}

class VqaTemplateDialog final : public QDialog {
public:
    explicit VqaTemplateDialog(QWidget* parent) : QDialog(parent) {
        setObjectName("VQATemplateGallery");
        setWindowTitle(chatTr("Template Gallery"));
        setMinimumSize(600, 450);
        auto* layout = new QVBoxLayout(this);
        layout->setContentsMargins(24, 24, 24, 24);
        layout->setSpacing(12);
        m_table = new QTableWidget(0, 3, this);
        m_table->setObjectName("VQATemplateTable");
        m_table->setHorizontalHeaderLabels(
                {chatTr("Select"), chatTr("Template Name"), chatTr("Action")});
        m_table->setEditTriggers(QAbstractItemView::NoEditTriggers);
        m_table->setSelectionMode(QAbstractItemView::NoSelection);
        m_table->setAlternatingRowColors(true);
        m_table->setShowGrid(false);
        m_table->verticalHeader()->hide();
        m_table->horizontalHeader()->setSectionResizeMode(0,
                                                          QHeaderView::Fixed);
        m_table->horizontalHeader()->setSectionResizeMode(1,
                                                          QHeaderView::Stretch);
        m_table->horizontalHeader()->setSectionResizeMode(2,
                                                          QHeaderView::Fixed);
        m_table->setColumnWidth(0, 100);
        m_table->setColumnWidth(2, 120);
        layout->addWidget(m_table);
        m_status = new QLabel(this);
        layout->addWidget(m_status);
        auto* controls = new QHBoxLayout;
        auto* add = new QPushButton(chatTr("Add"), this);
        add->setObjectName("VQAAddTemplate");
        controls->addWidget(add);
        controls->addStretch();
        auto* buttons = new QDialogButtonBox(
                QDialogButtonBox::Ok | QDialogButtonBox::Cancel, this);
        controls->addWidget(buttons);
        layout->addLayout(controls);
        QFile file(vqaConfigPath("prompts.json"));
        if (file.exists()) {
            QJsonParseError error;
            const auto json =
                    file.open(QIODevice::ReadOnly)
                            ? QJsonDocument::fromJson(file.readAll(), &error)
                            : QJsonDocument();
            m_writable = !json.isNull() && json.isArray();
            for (const auto& value : json.array())
                m_writable &= value.isObject() &&
                              value.toObject().value("name").isString() &&
                              value.toObject().value("content").isString();
            if (m_writable)
                m_user = json.array();
            else
                m_status->setText(chatTr(
                        "Invalid template configuration; file preserved."));
        }
        add->setEnabled(m_writable);
        connect(add, &QPushButton::clicked, this, [this]() { edit(-1); });
        connect(m_table, &QTableWidget::itemDoubleClicked, this,
                [this](QTableWidgetItem* item) {
                    if (m_writable && item->column() == 1 &&
                        item->row() < m_user.size())
                        edit(item->row());
                });
        connect(buttons, &QDialogButtonBox::accepted, this, &QDialog::accept);
        connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);
        rebuild();
    }
    QString selected() const { return m_selected; }

private:
    bool save(const QJsonArray& user) {
        QString error;
        if (!writeVqaJson(vqaConfigPath("prompts.json"),
                          QJsonDocument(user).toJson(), &error)) {
            m_status->setText(error);
            return false;
        }
        m_user = user;
        return true;
    }
    void edit(int row) {
        QDialog editor(this);
        editor.setObjectName("VQATemplateEditor");
        editor.setWindowTitle(
                chatTr(row < 0 ? "Add Template" : "Edit Template"));
        editor.setFixedSize(520, 350);
        auto* form = new QFormLayout(&editor);
        QLineEdit name(&editor);
        name.setObjectName("VQATemplateName");
        QPlainTextEdit content(&editor);
        content.setObjectName("VQATemplateContent");
        if (row >= 0) {
            name.setText(m_user[row].toObject().value("name").toString());
            content.setPlainText(
                    m_user[row].toObject().value("content").toString());
        }
        form->addRow(chatTr("Template Name:"), &name);
        form->addRow(chatTr("Template Content:"), &content);
        QDialogButtonBox buttons(
                QDialogButtonBox::Ok | QDialogButtonBox::Cancel, &editor);
        form->addRow(&buttons);
        connect(&buttons, &QDialogButtonBox::accepted, &editor, [&]() {
            if (!name.text().trimmed().isEmpty() &&
                !content.toPlainText().trimmed().isEmpty())
                editor.accept();
        });
        connect(&buttons, &QDialogButtonBox::rejected, &editor,
                &QDialog::reject);
        if (editor.exec() != QDialog::Accepted) return;
        auto user = m_user;
        const QJsonObject value{{"name", name.text().trimmed()},
                                {"content", content.toPlainText().trimmed()}};
        if (row < 0)
            user.append(value);
        else
            user[row] = value;
        if (save(user)) rebuild();
    }
    void rebuild() {
        m_selected.clear();
        m_table->setRowCount(0);
        auto all = m_user;
        for (const auto& value : defaultVqaTemplates())
            all.append(QJsonObject{{"name", value.first},
                                   {"content", value.second}});
        // O(n) widgets/storage; user templates precede immutable source
        // defaults.
        m_table->setRowCount(all.size());
        for (int row = 0; row < all.size(); ++row) {
            const auto value = all[row].toObject();
            m_table->setRowHeight(row, 36);
            auto* check = new QCheckBox(m_table);
            m_table->setCellWidget(row, 0, check);
            auto* name = new QTableWidgetItem(value.value("name").toString());
            name->setToolTip(value.value("content").toString());
            m_table->setItem(row, 1, name);
            connect(check, &QCheckBox::toggled, this,
                    [this, row, name](bool on) {
                        if (on)
                            for (int i = 0; i < m_table->rowCount(); ++i)
                                if (i != row) {
                                    auto* other = static_cast<QCheckBox*>(
                                            m_table->cellWidget(i, 0));
                                    other->blockSignals(true);
                                    other->setChecked(false);
                                    other->blockSignals(false);
                                }
                        m_selected = on ? name->toolTip() : QString();
                    });
            auto* remove = new QPushButton(chatTr("Delete"), m_table);
            remove->setEnabled(m_writable && row < m_user.size());
            m_table->setCellWidget(row, 2, remove);
            connect(remove, &QPushButton::clicked, this, [this, row]() {
                if (QMessageBox::question(this, chatTr("Confirm"),
                                          chatTr("Are you sure you want to "
                                                 "delete this template?"),
                                          QMessageBox::Yes | QMessageBox::No,
                                          QMessageBox::No) != QMessageBox::Yes)
                    return;
                const auto name = m_user[row].toObject().value("name");
                QJsonArray user;
                // Source deletion removes all user entries with this name.
                for (const auto& value : m_user)
                    if (value.toObject().value("name") != name)
                        user.append(value);
                if (save(user))
                    QTimer::singleShot(0, this, [this]() { rebuild(); });
            });
        }
    }
    QTableWidget* m_table;
    QLabel* m_status;
    QJsonArray m_user;
    QString m_selected;
    bool m_writable = true;
};

struct Source {
    QString path;
    QImage image;
    QJsonObject document;
    quint64 generation = 0;
};

QString exportChatDataset(const Source& current,
                          const QString& directory,
                          QString* error) {
    if (current.path.isEmpty()) {
        *error = chatTr("No file is currently open.");
        return {};
    }
    const QDir input(QFileInfo(current.path).absolutePath());
    const QString stem = QStringLiteral("sharegpt_mllm_data_%1")
                                 .arg(QDateTime::currentDateTime().toString(
                                         "yyyyMMdd_HHmmss"));
    QTemporaryDir staging(QDir(directory).filePath(".chat-export-XXXXXX"));
    if (!staging.isValid() || !QDir(staging.path()).mkdir("images")) {
        *error = chatTr("Cannot create the export directory.");
        return {};
    }
    QJsonArray records;
    // O(sidecars + copied bytes); no decoded images or image-size buffers.
    for (const auto& name :
         input.entryList({"*.json"}, QDir::Files, QDir::Name)) {
        QFile file(input.filePath(name));
        if (!file.open(QIODevice::ReadOnly)) {
            *error = file.errorString();
            return {};
        }
        const auto document = QJsonDocument::fromJson(file.readAll());
        if (!document.isObject()) {
            *error = chatTr("Invalid annotation: %1").arg(file.fileName());
            return {};
        }
        auto object = document.object();
        if (QFileInfo(file.fileName()).completeBaseName() ==
            QFileInfo(current.path).completeBaseName())
            object = current.document;
        const auto history = object.value("chat_history");
        if (history.isUndefined() || history.toArray().isEmpty()) continue;
        const auto image = object.value("imagePath").toString();
        if (image.isEmpty()) continue;
        const QFileInfo sourceImage(input.filePath(image));
        // The portable source format uses one local image basename per record.
        if (image != QFileInfo(image).fileName() || !sourceImage.isFile() ||
            QFileInfo(sourceImage.canonicalFilePath()).absolutePath() !=
                    QFileInfo(input.absolutePath()).canonicalFilePath()) {
            *error = chatTr("Missing or unsupported dataset image: %1")
                             .arg(image);
            return {};
        }
        const QString archiveImage = QStringLiteral("images/%1").arg(image);
        const QString copied = QDir(staging.path()).filePath(archiveImage);
        if (!QFile::exists(copied) &&
            !QFile::copy(sourceImage.filePath(), copied)) {
            *error = chatTr("Cannot copy dataset image: %1").arg(image);
            return {};
        }
        QJsonArray messages, images;
        for (const auto& value : history.toArray()) {
            const auto message = value.toObject();
            if (!value.isObject() || !message.value("role").isString() ||
                !message.value("content").isString()) {
                *error = chatTr("Invalid chat history: %1").arg(name);
                return {};
            }
            messages.append(QJsonObject{{"role", message.value("role")},
                                        {"content", message.value("content")}});
            if (!message.value("image").toString().isEmpty() &&
                images.isEmpty())
                images.append(archiveImage);
        }
        records.append(QJsonObject{{"messages", messages}, {"images", images}});
    }
    if (records.isEmpty()) {
        *error = chatTr("No valid chat data found to export.");
        return {};
    }
    const QString jsonPath = QDir(staging.path()).filePath(stem + ".json");
    if (!writeVqaJson(jsonPath, QJsonDocument(records).toJson(), error))
        return {};
    const QString archive = QDir(directory).filePath(stem + ".zip");
    return writeZipDirectory(staging.path(), archive, error, false) ? archive
                                                                    : QString();
}

int importChatDataset(const QString& inputPath,
                      const QString& imageDirectory,
                      const QString& labelDirectory,
                      QString* error) {
    QFile input(inputPath);
    if (!input.open(QIODevice::ReadOnly)) {
        *error = input.errorString();
        return -1;
    }
    const auto json = QJsonDocument::fromJson(input.readAll());
    if (!json.isArray()) {
        *error = chatTr("Invalid dataset format. Expected a list of records.");
        return -1;
    }
    struct Item {
        QString source, image, label;
        QByteArray json, previous;
        bool existed = false;
    };
    QVector<Item> items;
    const QDir images(imageDirectory), labels(labelDirectory);
    const QDir imported(QFileInfo(inputPath).absolutePath());
    if (!images.exists() || !labels.exists()) {
        *error = chatTr("The image or annotation directory does not exist.");
        return -1;
    }
    // O(input JSON + sidecar bytes); image headers are read one at a time.
    // Validate the whole input before copying images or replacing sidecars.
    for (const auto& value : json.array()) {
        const auto record = value.toObject();
        if (!value.isObject() || !record.value("messages").isArray() ||
            !record.value("images").isArray()) {
            *error = chatTr("Invalid ShareGPT record.");
            return -1;
        }
        const auto messages = record.value("messages").toArray();
        const auto paths = record.value("images").toArray();
        if (messages.isEmpty() || paths.isEmpty()) continue;
        for (const auto& message : messages)
            if (!message.isObject() ||
                !message.toObject().value("role").isString() ||
                !message.toObject().value("content").isString()) {
                *error = chatTr("Invalid chat message.");
                return -1;
            }
        for (const auto& path : paths) {
            const QString relative = path.toString();
            const QString clean = QDir::cleanPath(relative);
            if (!path.isString() || relative.isEmpty() ||
                QDir::isAbsolutePath(relative) || clean == ".." ||
                clean.startsWith("../") || relative.contains('\\')) {
                *error = chatTr("Invalid dataset image path: %1").arg(relative);
                return -1;
            }
            const QString name = QFileInfo(relative).fileName();
            Item item;
            item.image = images.filePath(name);
            item.label = labels.filePath(QFileInfo(name).completeBaseName() +
                                         ".json");
            const QFileInfo local(item.image),
                    external(imported.filePath(clean));
            if (local.exists()) {
                if (!local.isFile() || local.isSymLink()) {
                    *error = chatTr("Invalid local image: %1").arg(name);
                    return -1;
                }
                item.source = item.image;
            } else if (external.isFile()) {
                const QString root = imported.canonicalPath() + '/';
                if (!external.canonicalFilePath().startsWith(root)) {
                    *error = chatTr(
                            "Dataset image is outside the import directory.");
                    return -1;
                }
                item.source = external.absoluteFilePath();
            } else
                continue;
            const QSize size = QImageReader(item.source).size();
            if (!size.isValid()) {
                *error = chatTr("Cannot read dataset image: %1").arg(name);
                return -1;
            }
            QJsonArray history;
            for (const auto& message : messages) {
                auto entry = message.toObject();
                const bool hasImage =
                        entry.value("role") == "user" &&
                        entry.value("content").toString().contains("<image>");
                history.append(QJsonObject{
                        {"role", entry.value("role")},
                        {"content", entry.value("content")},
                        {"image", hasImage ? QJsonValue(item.image)
                                           : QJsonValue(QJsonValue::Null)}});
            }
            item.json =
                    QJsonDocument(
                            QJsonObject{{"version", annotationFormatVersion()},
                                        {"flags", QJsonObject()},
                                        {"shapes", QJsonArray()},
                                        {"imagePath", name},
                                        {"imageData", QJsonValue::Null},
                                        {"imageHeight", size.height()},
                                        {"imageWidth", size.width()},
                                        {"chat_history", history},
                                        {"description", ""}})
                            .toJson();
            if (QFileInfo(item.label).isSymLink()) {
                *error = chatTr("Cannot replace a linked annotation: %1")
                                 .arg(item.label);
                return -1;
            }
            item.existed = QFileInfo::exists(item.label);
            if (item.existed) {
                QFile previous(item.label);
                if (!QFileInfo(item.label).isFile() ||
                    QFileInfo(item.label).isSymLink() ||
                    !previous.open(QIODevice::ReadOnly)) {
                    *error = chatTr("Cannot read existing annotation: %1")
                                     .arg(item.label);
                    return -1;
                }
                item.previous = previous.readAll();
                if (previous.error() != QFileDevice::NoError) {
                    *error = previous.errorString();
                    return -1;
                }
            }
            items.append(item);
            break;  // The source imports only the first available image.
        }
    }
    QStringList copied;
    int written = 0;
    for (const auto& item : items) {
        if (!QFile::exists(item.image) &&
            !QFile::copy(item.source, item.image)) {
            *error = chatTr("Cannot copy dataset image: %1").arg(item.image);
            break;
        }
        if (item.image != item.source && !copied.contains(item.image))
            copied.append(item.image);
        if (!writeVqaJson(item.label, item.json, error)) break;
        ++written;
    }
    if (written != items.size()) {
        // Restore only files written by this operation, in reverse order.
        for (int i = written - 1; i >= 0; --i) {
            const auto& item = items[i];
            QString rollbackError;
            if (item.existed) {
                if (!writeVqaJson(item.label, item.previous, &rollbackError))
                    *error += '\n' + rollbackError;
            } else if (!QFile::remove(item.label))
                *error += chatTr("\nCannot restore annotation: %1")
                                  .arg(item.label);
        }
        for (const auto& image : copied)
            if (!QFile::remove(image))
                *error +=
                        chatTr("\nCannot remove imported image: %1").arg(image);
        return -1;
    }
    return written;
}

class ChatModelRow final : public QFrame {
public:
    ChatModelRow(const QString& id,
                 const QJsonObject& metadata,
                 bool favoritesSection,
                 std::function<void()> select,
                 std::function<void(bool)> favorite,
                 QWidget* parent)
        : QFrame(parent),
          m_select(std::move(select)),
          m_favorite(metadata.value("favorite").toBool()),
          m_favoritesSection(favoritesSection) {
        setObjectName("ChatModelRow");
        setProperty("modelId", id);
        setProperty("vision", metadata.value("vision").toBool());
        setFixedHeight(32);
        auto* layout = new QHBoxLayout(this);
        layout->setContentsMargins(8, 0, 8, 0);
        layout->addWidget(new QLabel(id, this));
        layout->addStretch();
        auto icon = [](const QString& name) {
            return QIcon(":/CC/plugin/qAnyLabeling/images/" + name + ".svg");
        };
        auto* selected = new QLabel(this);
        selected->setFixedSize(16, 16);
        if (metadata.value("selected").toBool())
            selected->setPixmap(icon("check").pixmap(16, 16));
        layout->addWidget(selected);
        m_star = new QPushButton(this);
        m_star->setObjectName("ChatModelFavorite");
        m_star->setProperty("modelId", id);
        m_star->setFixedSize(16, 16);
        m_star->setIcon(icon(m_favorite ? "starred" : "star"));
        m_star->setStyleSheet("border: none; background: transparent;");
        m_star->setVisible(m_favorite && !favoritesSection);
        connect(m_star, &QPushButton::clicked, this,
                [this, favorite]() { favorite(!m_favorite); });
        layout->addWidget(m_star);
        if (metadata.value("vision").toBool()) {
            auto* vision = new QLabel(this);
            vision->setObjectName("ChatModelVision");
            vision->setPixmap(icon("vision").pixmap(16, 16));
            layout->addWidget(vision);
        }
    }

protected:
    bool event(QEvent* event) override {
        if (event->type() == QEvent::Enter) m_star->show();
        return QFrame::event(event);
    }
    void leaveEvent(QEvent* event) override {
        m_star->setVisible(m_favorite && !m_favoritesSection);
        QFrame::leaveEvent(event);
    }
    void mousePressEvent(QMouseEvent* event) override {
        if (event->button() == Qt::LeftButton) m_select();
        QFrame::mousePressEvent(event);
    }

private:
    std::function<void()> m_select;
    QPushButton* m_star;
    bool m_favorite, m_favoritesSection;
};

class ChatModelPopup final : public QWidget {
public:
    ChatModelPopup(const QString& provider,
                   const QJsonObject& models,
                   std::function<void(const QString&)> select,
                   std::function<bool(const QString&, bool)> saveFavorite,
                   QWidget* parent)
        : QWidget(parent, Qt::Popup | Qt::FramelessWindowHint),
          m_provider(provider),
          m_models(models),
          m_select(std::move(select)),
          m_saveFavorite(std::move(saveFavorite)) {
        setObjectName("ChatModelPopup");
        setAttribute(Qt::WA_DeleteOnClose);
        resize(360, 500);
        const bool dark =
                parent->palette().color(QPalette::Window).lightness() < 128;
        // Exact source theme tokens, scoped to this popup's object names.
        setStyleSheet(
                QString("QWidget#ChatModelPopup { background-color: %1; "
                        "border-radius: 8px; }"
                        "QFrame#ChatModelProviderSection, "
                        "QFrame#ChatModelFavorites {"
                        " background-color: %1; border: none; }"
                        "QFrame#ChatModelRow { background-color: transparent; "
                        "border: none; border-radius: 4px; }"
                        "QFrame#ChatModelRow:hover { background-color: %2; }"
                        "QFrame#ChatModelSeparator { color: %5; max-height: "
                        "1px; }"
                        "QLabel { color: %3; background: transparent; border: "
                        "none; font-size: 11px; }"
                        "QLineEdit#ChatModelSearch { background-color: %4; "
                        "color: %3;"
                        " border: 1px solid %5; border-radius: 8px; padding: "
                        "5px; font-size: 11px; }"
                        "QLineEdit#ChatModelSearch:focus { border: 2px solid "
                        "%6; }"
                        "QScrollArea, QWidget#ChatModelContainer { "
                        "background-color: %1; border: none; }"
                        "QScrollBar:vertical { background-color: %4; width: "
                        "10px; margin: 16px 0; }"
                        "QScrollBar::handle:vertical { background-color: %7; "
                        "min-height: 20px; border-radius: 5px; }"
                        "QScrollBar::add-line:vertical, "
                        "QScrollBar::sub-line:vertical { height: 16px; border: "
                        "none; background-color: %4; }"
                        "QScrollBar::add-line:vertical { subcontrol-position: "
                        "bottom; subcontrol-origin: margin;"
                        " image: "
                        "url(:/CC/plugin/qAnyLabeling/images/caret-down.svg); }"
                        "QScrollBar::sub-line:vertical { subcontrol-position: "
                        "top; subcontrol-origin: margin;"
                        " image: "
                        "url(:/CC/plugin/qAnyLabeling/images/caret-up.svg); }")
                        .arg(dark ? "#2c2c2e" : "#f5f5f7",
                             dark ? "#3a3a3c" : "#e5e5e5",
                             dark ? "#f5f5f7" : "#1d1d1f",
                             dark ? "#2c2c2e" : "#F9F9F9",
                             dark ? "#3a3a3c" : "#E5E5E5",
                             dark ? "#409CFF" : "#60A5FA",
                             dark ? "#48484a" : "#c1c1c1"));
        auto* layout = new QVBoxLayout(this);
        layout->setContentsMargins(12, 12, 12, 12);
        m_search = new QLineEdit(this);
        m_search->setObjectName("ChatModelSearch");
        m_search->setPlaceholderText(chatTr("Search models"));
        m_search->setFixedHeight(32);
        m_search->addAction(QIcon(":/CC/plugin/qAnyLabeling/images/search.svg"),
                            QLineEdit::LeadingPosition);
        layout->addWidget(m_search);
        auto* scroll = new QScrollArea(this);
        scroll->setWidgetResizable(true);
        scroll->setFrameShape(QFrame::NoFrame);
        auto* container = new QWidget(scroll);
        container->setObjectName("ChatModelContainer");
        m_layout = new QVBoxLayout(container);
        m_layout->setContentsMargins(0, 8, 0, 8);
        m_layout->setSpacing(8);
        scroll->setWidget(container);
        layout->addWidget(scroll);
        rebuild();
        connect(m_search, &QLineEdit::textChanged, this,
                [this]() { filter(); });
    }

    void populateEmpty(const QJsonObject& models) {
        // The source waits for a first, uncached discovery before showing
        // its menu. Keep Qt responsive and populate that empty menu on reply.
        // A menu already showing cached models retains its current snapshot.
        if (!m_models.isEmpty() || models.isEmpty()) return;
        m_models = models;
        rebuild();
    }

private:
    void rebuild() {
        while (auto* item = m_layout->takeAt(0)) {
            if (auto* widget = item->widget()) {
                widget->hide();
                widget->deleteLater();
            }
            delete item;
        }
        m_groups.clear();
        m_rows.clear();
        m_separator = nullptr;
        for (bool favorites : {true, false}) {
            auto* group = new QFrame(this);
            const QString title = favorites ? chatTr("Favorites")
                                            : m_provider.left(1).toUpper() +
                                                      m_provider.mid(1);
            group->setObjectName(favorites ? "ChatModelFavorites"
                                           : "ChatModelProviderSection");
            auto* section = new QVBoxLayout(group);
            section->setContentsMargins(8, 4, 8, 4);
            auto* header = new QHBoxLayout;
            auto* icon = new QLabel(group);
            const QString iconName = favorites ? QString("star-black.svg")
                                               : QString(m_provider + ".png");
            icon->setPixmap(QIcon(":/CC/plugin/qAnyLabeling/images/" + iconName)
                                    .pixmap(16, 16));
            icon->setFixedSize(16, 16);
            header->addWidget(icon);
            auto* label = new QLabel(title, group);
            label->setStyleSheet(
                    "font-family: sans-serif; font-weight: 700; font-size: "
                    "13px;");
            header->addWidget(label);
            header->addStretch();
            section->addLayout(header);
            auto* rows = new QVBoxLayout;
            rows->setContentsMargins(0, 0, 0, 0);
            section->addLayout(rows);
            for (auto it = m_models.begin(); it != m_models.end(); ++it) {
                const QString id = it.key();
                const auto metadata = it.value().toObject();
                if (favorites && !metadata.value("favorite").toBool()) continue;
                auto* row = new ChatModelRow(
                        id, metadata, favorites,
                        [this, id]() {
                            m_select(id);
                            close();
                        },
                        [this, id](bool favorite) {
                            if (!m_saveFavorite(id, favorite)) return;
                            auto value = m_models.value(id).toObject();
                            value.insert("favorite", favorite);
                            m_models.insert(id, value);
                            // Rebuild after the originating button event exits.
                            QTimer::singleShot(0, this,
                                               [this]() { rebuild(); });
                        },
                        group);
                rows->addWidget(row);
                m_rows.append({group, row});
            }
            if (rows->count()) {
                m_layout->addWidget(group);
                m_groups.append(group);
                if (favorites) {
                    auto* line = new QFrame(this);
                    line->setObjectName("ChatModelSeparator");
                    line->setFrameShape(QFrame::HLine);
                    line->setFrameShadow(QFrame::Plain);
                    m_layout->addWidget(line);
                    m_separator = line;
                }
            } else
                delete group;
        }
        m_empty = new QLabel(chatTr("No models found."), this);
        m_empty->setObjectName("ChatModelEmpty");
        m_empty->setAlignment(Qt::AlignCenter);
        m_empty->setStyleSheet(
                QString("color: %1; font-size: 14px; padding: 20px;")
                        .arg(palette().color(QPalette::Window).lightness() < 128
                                     ? "#aeaeb2"
                                     : "#86868b"));
        m_layout->addWidget(m_empty);
        m_layout->addStretch();
        filter();
    }
    void filter() {
        QSet<QWidget*> visible;
        for (const auto& entry : m_rows) {
            const bool match = AnnotationChatConfig::modelNameMatches(
                    m_search->text(),
                    entry.second->property("modelId").toString());
            entry.second->setVisible(match);
            if (match) visible.insert(entry.first);
        }
        for (auto* group : m_groups) group->setVisible(visible.contains(group));
        if (m_separator) m_separator->setVisible(visible.size() == 2);
        m_empty->setVisible(visible.isEmpty());
    }
    QString m_provider;
    QJsonObject m_models;
    std::function<void(const QString&)> m_select;
    std::function<bool(const QString&, bool)> m_saveFavorite;
    QLineEdit* m_search;
    QVBoxLayout* m_layout;
    QLabel* m_empty;
    QFrame* m_separator = nullptr;
    QVector<QWidget*> m_groups;
    QVector<QPair<QWidget*, QWidget*>> m_rows;
};

// Chatbot and VQA share provider selection and OpenAI-compatible transport.
// This is endpoint configuration, not a second local model catalog/cache.
class ServicePanel final : public QGroupBox {
public:
    explicit ServicePanel(QWidget* parent)
        : QGroupBox(chatTr("Model settings"), parent) {
        client->setObjectName("ChatCompletionClient");
        m_modelClient->setObjectName("ChatModelDiscoveryClient");
        auto* form = new QFormLayout(this);
        provider = new QComboBox(this);
        provider->setObjectName(QStringLiteral("ChatProvider"));
        const QStringList names{"custom", "anthropic", "deepseek",   "google",
                                "ollama", "openai",    "openrouter", "qwen"};
        const auto defaults = AnnotationChatConfig::defaultProviders();
        for (const auto& name : names)
            provider->addItem(name, defaults.value(name)
                                            .toObject()
                                            .value("api_address")
                                            .toString());
        address = new QLineEdit(this);
        address->setObjectName(QStringLiteral("ChatAddress"));
        key = new QLineEdit(this);
        key->setObjectName(QStringLiteral("ChatApiKey"));
        key->setEchoMode(QLineEdit::Password);
        model = new QComboBox(this);
        model->setObjectName(QStringLiteral("ChatModel"));
        model->setEditable(true);
        auto* refresh = new QPushButton(chatTr("Refresh models"), this);
        refresh->setObjectName(QStringLiteral("ChatRefreshModels"));
        auto* browse = new QPushButton(chatTr("Select Model"), this);
        browse->setObjectName("ChatBrowseModels");
        m_browse = browse;
        m_refresh = refresh;
        temperature = new QDoubleSpinBox(this);
        temperature->setObjectName(QStringLiteral("ChatTemperature"));
        temperature->setRange(0, 2);
        temperature->setSingleStep(0.1);
        temperature->setDecimals(1);
        tokens = new QSpinBox(this);
        tokens->setObjectName(QStringLiteral("ChatMaxTokens"));
        tokens->setRange(0, 1000000);
        tokens->setSpecialValueText(chatTr("Default"));
        system = new QLineEdit(this);
        system->setObjectName(QStringLiteral("ChatSystemPrompt"));
        status = new QLabel(this);
        status->setObjectName(QStringLiteral("ChatStatus"));
        status->setWordWrap(true);
        form->addRow(chatTr("Provider"), provider);
        form->addRow(chatTr("API address"), address);
        form->addRow(chatTr("API key"), key);
        form->addRow(chatTr("Model"), model);
        form->addRow(browse);
        form->addRow(refresh);
        form->addRow(chatTr("Temperature"), temperature);
        form->addRow(chatTr("Max length"), tokens);
        form->addRow(chatTr("System prompt"), system);
        form->addRow(status);
        QString error;
        m_configValid = AnnotationChatConfig::initialize(&error);
        if (m_configValid) {
            m_providers = AnnotationChatConfig::read("providers.json", &error);
            m_models = AnnotationChatConfig::read("models.json", &error);
            m_configValid = error.isEmpty();
        }
        if (!m_configValid) {
            m_providers = AnnotationChatConfig::defaultProviders();
            m_models = AnnotationChatConfig::defaultModels();
            status->setText(error);
            status->setProperty("configurationError", error);
        }
        for (auto it = m_providers.begin(); it != m_providers.end(); ++it)
            if (provider->findText(it.key()) < 0) provider->addItem(it.key());
        const auto parameters = m_models.value("settings").toObject();
        provider->setCurrentText(parameters.value("provider").toString());
        loadProvider();
        temperature->setValue(parameters.value("temperature").toDouble(10) /
                              10);
        tokens->setRange(0, 9999999);
        tokens->setValue(parameters.value("max_length").toInt());
        system->setText(parameters.value("system_prompt").toString());
        connect(provider, &QComboBox::currentTextChanged, this, [this]() {
            if (m_popup) m_popup->close();
            if (m_fetchingModels) {
                m_modelClient->cancel();
                m_fetchingModels = false;
            }
            saveProvider(m_loadedProvider);
            loadProvider();
            settings();
            refreshModels(false);
        });
        connect(browse, &QPushButton::clicked, this, [this, browse]() {
            if (m_popup) m_popup->close();
            settings();
            refreshModels(false);
            const QString name = provider->currentText();
            m_popup = new ChatModelPopup(
                    name,
                    m_models.value("models_data")
                            .toObject()
                            .value(name)
                            .toObject(),
                    [this](const QString& id) {
                        model->setCurrentText(id);
                        settings();
                    },
                    [this, name](const QString& id, bool favorite) {
                        if (!m_configValid) return false;
                        QString error;
                        auto document = AnnotationChatConfig::read(
                                "models.json", &error);
                        if (error.isEmpty()) {
                            auto providers =
                                    document.value("models_data").toObject();
                            auto models = providers.value(name).toObject();
                            auto value = models.value(id).toObject();
                            if (value.isEmpty()) return false;
                            value.insert("favorite", favorite);
                            models.insert(id, value);
                            providers.insert(name, models);
                            document.insert("models_data", providers);
                            if (AnnotationChatConfig::write("models.json",
                                                            document, &error)) {
                                m_models = document;
                                return true;
                            }
                        }
                        status->setText(error);
                        return false;
                    },
                    this);
            // Match ChatbotDialog.show_model_dropdown: center over the
            // button and open upward through the settings panel's available
            // height. The initial 360px width participates before resize.
            const QPoint button = browse->mapToGlobal(QPoint(0, 0));
            const int x = int(button.x() + browse->width() / 2.0 -
                              m_popup->width() / 2.0);
            const auto* panel = m_popupAnchor ? m_popupAnchor : this;
            const int available =
                    button.y() - panel->mapToGlobal(QPoint(0, 0)).y();
            m_popup->resize(panel->width(), available);
            m_popup->move(x, button.y() - m_popup->height());
            m_popup->show();
        });
        connect(refresh, &QPushButton::clicked, this,
                [this]() { refreshModels(true); });
        connect(m_modelClient, &ChatServiceClient::modelsReady, this,
                [this](const QJsonArray& models, const QString& error) {
                    m_fetchingModels = false;
                    if (!error.isEmpty()) {
                        status->setText(error);
                        return;
                    }
                    const auto selected = model->currentText();
                    if (!cacheModels(models)) return;
                    if (m_popup && m_popup->isVisible())
                        m_popup->populateEmpty(
                                m_models.value("models_data")
                                        .toObject()
                                        .value(provider->currentText())
                                        .toObject());
                    model->clear();
                    for (const auto& value : models) {
                        const auto id = value.toObject().value("id").toString();
                        if (!id.isEmpty() && model->findText(id) < 0)
                            model->addItem(id);
                    }
                    model->setCurrentText(selected);
                    status->setText(chatTr("%1 models").arg(models.size()));
                });
        for (auto* edit : {address, key, system})
            connect(edit, &QLineEdit::editingFinished, this,
                    [this]() { settings(); });
        connect(model->lineEdit(), &QLineEdit::editingFinished, this,
                [this]() { settings(); });
        connect(model, QOverload<int>::of(&QComboBox::activated), this,
                [this](int) { settings(); });
        connect(temperature, &QDoubleSpinBox::editingFinished, this,
                [this]() { settings(); });
        connect(tokens, &QSpinBox::editingFinished, this,
                [this]() { settings(); });
        QTimer::singleShot(0, this, [this]() { refreshModels(false); });
    }
    // Edits and selections persist at their commit signals. A deferred dialog
    // deletion must not overwrite newer settings from another Chat/VQA window.
    ChatServiceClient::Settings settings() {
        ChatServiceClient::Settings value;
        value.baseUrl = QUrl(address->text().trimmed());
        value.apiKey = key->text();
        value.model = model->currentText().trimmed();
        value.temperature = temperature->value();
        value.maxTokens = tokens->value();
        saveProvider(provider->currentText());
        saveSelection(value);
        return value;
    }
    QJsonArray messages(const QJsonArray& conversation) const {
        QJsonArray result;
        if (!system->text().trimmed().isEmpty())
            result.append(QJsonObject{{"role", "system"},
                                      {"content", system->text().trimmed()}});
        for (const auto& entry : conversation) result.append(entry);
        return result;
    }
    ChatServiceClient* client = new ChatServiceClient(this);
    QLabel* status;
    QComboBox *provider, *model;
    QLineEdit *address, *key, *system;
    QDoubleSpinBox* temperature;
    QSpinBox* tokens;

    // Reuse the existing controls and configuration owner in the source's
    // Chat layout. VQA/classifier retain their own compact service form.
    QWidget* createChatProviders(QWidget* parent) {
        auto* panel = new QWidget(parent);
        panel->setObjectName("ChatProvidersPanel");
        panel->setMinimumWidth(200);
        panel->setMaximumWidth(250);
        auto* layout = new QVBoxLayout(panel);
        layout->setContentsMargins(20, 24, 20, 24);
        layout->setSpacing(12);
        auto* group = new QButtonGroup(panel);
        for (int i = 0; i < provider->count(); ++i) {
            const auto name = provider->itemText(i);
            QString title = name;
            if (!title.isEmpty()) title[0] = title[0].toUpper();
            auto* button = new QPushButton(title, panel);
            button->setObjectName("ChatProviderButton_" + name);
            const auto icon = name == "qwen" ? "alibaba" : name;
            button->setIcon(QIcon(":/CC/plugin/qAnyLabeling/images/" + icon +
                                  ".png"));
            button->setIconSize(QSize(16, 16));
            button->setFixedHeight(40);
            button->setCheckable(true);
            button->setChecked(name == provider->currentText());
            group->addButton(button);
            layout->addWidget(button);
            connect(button, &QPushButton::clicked, this,
                    [this, name]() { provider->setCurrentText(name); });
            connect(provider, &QComboBox::currentTextChanged, button,
                    [button, name](const QString& selected) {
                        button->setChecked(selected == name);
                    });
        }
        layout->addStretch();
        return panel;
    }
    void useChatSettingsTabs(QWidget* previewSettingsPanel) {
        m_popupAnchor = previewSettingsPanel;
        setObjectName("ChatSettingsPanel");
        setTitle({});
        auto* form = qobject_cast<QFormLayout*>(layout());
        // QFormLayout owns generated labels, not the reusable input controls.
        while (form->count()) {
            auto* item = form->takeAt(0);
            if (auto* label = qobject_cast<QLabel*>(item->widget())) {
                if (label != status) delete label;
            }
            delete item;
        }
        delete form;
        provider->hide();
        model->hide();
        m_refresh->hide();
        temperature->hide();
        auto* outer = new QVBoxLayout(this);
        outer->setContentsMargins(0, 0, 0, 0);
        outer->setSpacing(0);
        auto* tabs = new QTabWidget(this);
        tabs->setObjectName("ChatSettingsTabs");
        tabs->setUsesScrollButtons(false);
        tabs->setDocumentMode(true);
        tabs->setElideMode(Qt::ElideNone);
        tabs->tabBar()->setExpanding(true);
        auto* backend = new QWidget(tabs);
        auto* connection = new QVBoxLayout(backend);
        connection->setContentsMargins(24, 24, 24, 24);
        connection->setSpacing(16);
        connection->addWidget(new QLabel(chatTr("API Address"), backend));
        address->setMinimumHeight(40);
        connection->addWidget(address);
        connection->addWidget(new QLabel(chatTr("API Key"), backend));
        auto* secret = new QHBoxLayout;
        key->setMinimumHeight(40);
        secret->addWidget(key, 1);
        auto* visibility = new QPushButton(backend);
        visibility->setObjectName("ChatApiKeyVisibility");
        visibility->setCheckable(true);
        visibility->setFixedSize(32, 32);
        visibility->setIcon(QIcon(":/CC/plugin/qAnyLabeling/images/eye-off.svg"));
        visibility->setToolTip(chatTr("Show API key"));
        connect(visibility, &QPushButton::toggled, key,
                [key = key, visibility](bool visible) {
                    key->setEchoMode(visible ? QLineEdit::Normal
                                             : QLineEdit::Password);
                    visibility->setIcon(QIcon(
                            QString(":/CC/plugin/qAnyLabeling/images/%1.svg")
                                    .arg(visible ? "eye" : "eye-off")));
                });
        secret->addWidget(visibility);
        connection->addLayout(secret);
        connection->addWidget(new QLabel(chatTr("Model Name"), backend));
        m_browse->setMinimumHeight(40);
        connection->addWidget(m_browse);
        // Custom providers accept a model name without a discovery endpoint.
        connection->addWidget(model);
        const auto showModel = [this](const QString& name) {
            model->setVisible(name == "custom");
            m_browse->setVisible(name != "custom");
        };
        connect(provider, &QComboBox::currentTextChanged, this, showModel);
        showModel(provider->currentText());
        const auto showSelection = [this](const QString& name) {
            m_browse->setText(name.isEmpty() ? chatTr("Select Model") : name);
        };
        connect(model, &QComboBox::currentTextChanged, this, showSelection);
        showSelection(model->currentText());
        connection->addStretch();
        auto* generation = new QWidget(tabs);
        auto* parameters = new QVBoxLayout(generation);
        parameters->setContentsMargins(24, 24, 24, 24);
        parameters->setSpacing(16);
        parameters->addWidget(
                new QLabel(chatTr("System instruction"), generation));
        parameters->addWidget(system);
        auto* header = new QHBoxLayout;
        header->addWidget(new QLabel(chatTr("Temperature"), generation));
        header->addStretch();
        auto* value = new QLabel(QString::number(temperature->value(), 'f', 1),
                                 generation);
        header->addWidget(value);
        parameters->addLayout(header);
        auto* slider = new QSlider(Qt::Horizontal, generation);
        slider->setObjectName("ChatTemperatureSlider");
        slider->setRange(0, 20);
        slider->setValue(qRound(temperature->value() * 10));
        connect(slider, &QSlider::valueChanged, this,
                [this, value](int position) {
                    temperature->setValue(position / 10.0);
                    value->setText(QString::number(position / 10.0, 'f', 1));
                    settings();
                });
        connect(temperature, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
                slider, [slider](double value) {
                    slider->setValue(qRound(value * 10));
                });
        parameters->addWidget(slider);
        auto* labels = new QHBoxLayout;
        labels->addWidget(new QLabel(chatTr("Precise"), generation));
        labels->addStretch();
        labels->addWidget(new QLabel(chatTr("Neutral"), generation));
        labels->addStretch();
        labels->addWidget(new QLabel(chatTr("Creative"), generation));
        parameters->addLayout(labels);
        parameters->addWidget(
                new QLabel(chatTr("Max output tokens"), generation));
        tokens->setFixedHeight(40);
        parameters->addWidget(tokens);
        parameters->addStretch();
        tabs->addTab(backend, chatTr("Backend"));
        tabs->addTab(generation, chatTr("Generation"));
        outer->addWidget(tabs, 1);
        outer->addWidget(status);
    }

private:
    QPushButton *m_browse = nullptr, *m_refresh = nullptr;
    QWidget* m_popupAnchor = nullptr;
    void refreshModels(bool force) {
        const auto name = provider->currentText();
        if ((!force && (name == "custom" || !m_configValid)) ||
            m_modelClient->busy())
            return;
        QString error;
        if (m_configValid) {
            const auto document =
                    AnnotationChatConfig::read("models.json", &error);
            if (!error.isEmpty()) {
                status->setText(error);
                return;
            }
            m_models = document;
        }
        // Shared by Chat and VQA, as in the source provider module. It owns
        // call counters, never a second list of models or downloaded assets.
        static AnnotationChatConfig::ModelRefreshTracker tracker;
        if (!force &&
            !tracker.shouldFetch(
                    name,
                    m_models.value("models_data").toObject().contains(name),
                    QDateTime::currentMSecsSinceEpoch() / 1000.0))
            return;
        auto options = settings();
        options.timeoutMs = 5000;  // Source model-discovery request timeout.
        status->setText(chatTr("Loading models…"));
        m_fetchingModels = true;
        m_modelClient->fetchModels(options);
    }
    void saveProvider(const QString& name) {
        if (name.isEmpty() || !m_configValid) return;
        QString error;
        auto providers = AnnotationChatConfig::read("providers.json", &error);
        if (!error.isEmpty()) {
            status->setText(error);
            return;
        }
        auto entry = providers.value(name).toObject();
        if (entry.isEmpty())
            entry = AnnotationChatConfig::defaultProviders()
                            .value(name)
                            .toObject();
        entry.insert("api_address", address->text());
        entry.insert("api_key", key->text());
        if (name == "custom")
            entry.insert("model_name", model->currentText().trimmed());
        providers.insert(name, entry);
        if (AnnotationChatConfig::write("providers.json", providers, &error))
            m_providers = providers;
        else
            status->setText(error);
    }
    void saveSelection(const ChatServiceClient::Settings& value) {
        if (!m_configValid) return;
        QString error;
        auto document = AnnotationChatConfig::read("models.json", &error);
        if (!error.isEmpty()) {
            status->setText(error);
            return;
        }
        const auto selectedProvider = provider->currentText();
        auto providers = document.value("models_data").toObject();
        for (auto p = providers.begin(); p != providers.end(); ++p) {
            auto models = p.value().toObject();
            for (auto m = models.begin(); m != models.end(); ++m) {
                auto entry = m.value().toObject();
                entry.insert("selected", p.key() == selectedProvider &&
                                                 m.key() == value.model);
                m.value() = entry;
            }
            p.value() = models;
        }
        auto parameters = document.value("settings").toObject();
        parameters.insert("provider", selectedProvider);
        parameters.insert("model_id", value.model.isEmpty()
                                              ? QJsonValue(QJsonValue::Null)
                                              : QJsonValue(value.model));
        parameters.insert("temperature", qRound(value.temperature * 10));
        parameters.insert("max_length", value.maxTokens > 0
                                                ? QJsonValue(value.maxTokens)
                                                : QJsonValue(QJsonValue::Null));
        parameters.insert("system_prompt",
                          system->text().isEmpty()
                                  ? QJsonValue(QJsonValue::Null)
                                  : QJsonValue(system->text()));
        document.insert("settings", parameters);
        document.insert("models_data", providers);
        if (AnnotationChatConfig::write("models.json", document, &error))
            m_models = document;
        else
            status->setText(error);
    }
    bool cacheModels(const QJsonArray& response) {
        if (!m_configValid)
            return true;  // Network use remains possible without destroying
                          // malformed files.
        QString error;
        auto document = AnnotationChatConfig::read("models.json", &error);
        if (!error.isEmpty()) {
            status->setText(error);
            return false;
        }
        auto providers = document.value("models_data").toObject();
        const auto previous =
                providers.value(provider->currentText()).toObject();
        QJsonObject models;
        auto visionModels = document.value("supported_vision_models").toArray();
        for (const auto& value : AnnotationChatConfig::defaultModels()
                                         .value("supported_vision_models")
                                         .toArray())
            if (!visionModels.contains(value)) visionModels.append(value);
        document.insert("supported_vision_models", visionModels);
        for (const auto& value : response) {
            const auto metadata = value.toObject();
            const auto id = metadata.value("id").toString();
            if (id.isEmpty()) continue;
            auto entry = previous.value(id).toObject();
            if (!entry.contains("selected")) entry.insert("selected", false);
            if (!entry.contains("favorite")) entry.insert("favorite", false);
            entry.insert("vision", AnnotationChatConfig::supportsVision(
                                           id, metadata, visionModels));
            models.insert(id, entry);
        }
        providers.insert(provider->currentText(), models);
        document.insert("models_data", providers);
        if (!AnnotationChatConfig::write("models.json", document, &error)) {
            status->setText(error);
            return false;
        }
        m_models = document;
        return true;
    }
    void loadProvider() {
        m_loadedProvider = provider->currentText();
        const auto defaults = AnnotationChatConfig::defaultProviders()
                                      .value(m_loadedProvider)
                                      .toObject();
        const auto entry =
                m_providers.value(m_loadedProvider).toObject(defaults);
        address->setText(entry.value("api_address").toString());
        key->setText(entry.value("api_key").toString());
        model->clear();
        const auto models = m_models.value("models_data")
                                    .toObject()
                                    .value(m_loadedProvider)
                                    .toObject();
        QString selected;
        for (auto it = models.begin(); it != models.end(); ++it) {
            model->addItem(it.key());
            if (it.value().toObject().value("selected").toBool())
                selected = it.key();
        }
        const auto parameters = m_models.value("settings").toObject();
        if (parameters.value("provider").toString() == m_loadedProvider)
            selected = parameters.value("model_id").toString();
        if (m_loadedProvider == "custom")
            selected = entry.value("model_name").toString(selected);
        model->setCurrentText(selected);
    }
    QString m_loadedProvider;
    QPointer<ChatModelPopup> m_popup;
    QJsonObject m_providers, m_models;
    // Discovery must not cancel or occupy the active completion request.
    ChatServiceClient* m_modelClient = new ChatServiceClient(this);
    bool m_configValid = false, m_fetchingModels = false;
};

// Message presentation adapted from upstream chatbot/chat.py; the dialog's
// history remains the only conversation source of truth.
class ChatMessage final : public QWidget {
public:
    using Action =
            std::function<void(ChatMessage*, const QString&, const QString&)>;
    ChatMessage(const QString& role, Action action, QWidget* parent)
        : QWidget(parent), m_role(role), m_action(std::move(action)) {
        setObjectName("ChatMessage");
        setProperty("role", role);
        auto* row = new QHBoxLayout(this);
        row->setContentsMargins(0, 0, 0, 0);
        row->setSpacing(0);
        if (role == "user") row->addStretch(30);
        auto* bubble = new QWidget(this);
        m_bubble = bubble;
        bubble->setObjectName("messageBubble");
        row->addWidget(bubble, role == "user" ? 70 : 100);
        auto* layout = new QVBoxLayout(bubble);
        layout->setContentsMargins(10, 8, 10, 8);
        layout->setSpacing(4);
        if (role != "user") {
            auto* header = new QHBoxLayout;
            m_roleFrame = new QFrame(bubble);
            m_roleFrame->setObjectName("roleLabelContainer");
            auto* iconLayout = new QHBoxLayout(m_roleFrame);
            iconLayout->setContentsMargins(2, 2, 2, 2);
            iconLayout->setSpacing(0);
            m_roleIcon = new QLabel(m_roleFrame);
            m_roleIcon->setObjectName("ChatMessageRoleIcon");
            m_roleIcon->setFixedSize(16, 16);
            iconLayout->addWidget(m_roleIcon);
            header->addWidget(m_roleFrame);
            header->addStretch();
            layout->addLayout(header);
        }
        m_content = new QLabel(bubble);
        m_content->setObjectName("ChatMessageContent");
        m_content->setWordWrap(true);
        m_content->setMinimumWidth(200);
        m_content->setTextInteractionFlags(Qt::TextBrowserInteraction);
        m_content->setOpenExternalLinks(role != "user");
        m_content->setMinimumHeight(10);
        m_content->setAlignment(Qt::AlignLeft | Qt::AlignTop);
        m_content->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
        layout->addWidget(m_content);
        auto* controls = new QHBoxLayout;
        controls->setContentsMargins(0, 2, 0, 2);
        controls->setSpacing(12);
        controls->addStretch();
        layout->addLayout(controls);
        auto button = [&](const char* name, const char* icon,
                          const char* title) {
            auto* b = new QPushButton(bubble);
            b->setObjectName(QString("ChatMessage%1").arg(name));
            b->setToolTip(chatTr(title));
            b->setIcon(QIcon(QString(":/CC/plugin/qAnyLabeling/images/%1.svg")
                                     .arg(icon)));
            b->setIconSize(QSize(16, 16));
            b->setFixedSize(20, 20);
            controls->addWidget(b);
            m_buttons.append(b);
            return b;
        };
        auto* copy = button("Copy", "copy", "Copy");
        m_copy = copy;
        auto* edit = button("Edit", "edit", "Edit");
        QPushButton* regenerate = nullptr;
        if (role != "user")
            regenerate = button("Regenerate", "refresh", "Regenerate");
        auto* remove = button("Delete", "trash", "Delete");
        m_editor = new QPlainTextEdit(bubble);
        m_editor->setObjectName("ChatMessageEditor");
        m_editor->setMinimumHeight(80);
        m_editor->setMaximumHeight(200);
        layout->addWidget(m_editor);
        m_editControls = new QWidget(bubble);
        auto* editLayout = new QHBoxLayout(m_editControls);
        editLayout->setContentsMargins(0, 8, 0, 0);
        auto* resend = new QPushButton(chatTr("Resend"), m_editControls);
        m_resend = resend;
        resend->setObjectName("ChatMessageResend");
        editLayout->addWidget(resend);
        resend->setVisible(role == "user");
        editLayout->addStretch();
        auto* cancel = new QPushButton(chatTr("Cancel"), m_editControls);
        m_cancel = cancel;
        cancel->setObjectName("ChatMessageCancel");
        auto* save = new QPushButton(chatTr("Save"), m_editControls);
        m_save = save;
        save->setObjectName("ChatMessageSave");
        editLayout->addWidget(cancel);
        editLayout->addWidget(save);
        layout->addWidget(m_editControls);
        m_editor->hide();
        m_editControls->hide();
        m_copyReset = new QTimer(this);
        m_copyReset->setObjectName("ChatMessageCopyReset");
        m_copyReset->setSingleShot(true);
        m_copyReset->setInterval(1000);
        connect(m_copyReset, &QTimer::timeout, this, [this]() {
            m_copy->setIcon(QIcon(":/CC/plugin/qAnyLabeling/images/copy.svg"));
            setProperty("copyFeedback", false);
        });
        connect(copy, &QPushButton::clicked, this, [this]() {
            copyContent(true);
        });
        connect(edit, &QPushButton::clicked, this, [this]() {
            m_editing = true;
            m_editor->setPlainText(m_text);
            m_editor->show();
            m_editControls->show();
            m_content->hide();
            updateButtons();
            m_editor->setFocus();
        });
        connect(cancel, &QPushButton::clicked, this, [this]() { endEdit(); });
        connect(save, &QPushButton::clicked, this, [this]() {
            const auto text = m_editor->toPlainText().trimmed();
            endEdit();
            if (!text.isEmpty()) m_action(this, "save", text);
        });
        connect(resend, &QPushButton::clicked, this, [this]() {
            const auto text = m_editor->toPlainText().trimmed();
            endEdit();
            if (!text.isEmpty() && text != m_text)
                m_action(this, "resend", text);
        });
        if (regenerate)
            connect(regenerate, &QPushButton::clicked, this,
                    [this]() { m_action(this, "regenerate", {}); });
        connect(remove, &QPushButton::clicked, this, [this]() {
            QPointer<ChatMessage> guard(this);
            const auto content = m_text;
            if (QMessageBox::warning(
                        this, chatTr("Delete"),
                        chatTr("Are you sure to delete this message forever?"),
                        QMessageBox::Yes | QMessageBox::No,
                        QMessageBox::No) == QMessageBox::Yes &&
                guard && content == m_text)
                m_action(this, "delete", {});
        });
        setContextMenuPolicy(Qt::CustomContextMenu);
        connect(this, &QWidget::customContextMenuRequested, this,
                [this, copy, edit, regenerate, remove](const QPoint& position) {
                    auto* menu = new QMenu(this);
                    QPointer<QMenu> guard(menu);
                    menu->setObjectName("ChatMessageMenu");
                    menu->setStyleSheet(chatStyle("menu", m_dark));
                    auto action = [menu](const char* title, const char* name,
                                           QPushButton* button) {
                        auto* item = menu->addAction(chatTr(title));
                        item->setObjectName(name);
                        item->setEnabled(button->isEnabled());
                        connect(item, &QAction::triggered, button,
                                &QPushButton::click);
                    };
                    auto* copyAction = menu->addAction(chatTr("Copy message"));
                    copyAction->setObjectName("ChatMessageMenuCopy");
                    copyAction->setEnabled(copy->isEnabled());
                    connect(copyAction, &QAction::triggered, this,
                            [this]() {
                                if (m_copy->isEnabled()) copyContent(false);
                            });
                    menu->addSeparator();
                    action("Edit message", "ChatMessageMenuEdit", edit);
                    if (regenerate)
                        action("Regenerate response", "ChatMessageMenuRegenerate",
                               regenerate);
                    menu->addSeparator();
                    action("Delete message", "ChatMessageMenuDelete", remove);
                    // A nested menu can outlive a source change. QObject
                    // disconnects its actions when their message is destroyed.
                    menu->exec(mapToGlobal(position));
                    if (guard) guard->deleteLater();
                });
        applyStyles(palette().color(QPalette::Window).lightness() < 128);
        updateButtons();
    }
    void setProvider(const QString& provider) {
        if (!m_roleIcon || m_provider == provider) return;
        m_provider = provider;
        QPixmap image(QString(":/CC/plugin/qAnyLabeling/images/%1.png")
                              .arg(provider));
        if (image.isNull())
            image.load(":/CC/plugin/qAnyLabeling/images/brain.png");
        m_roleIcon->setPixmap(image.scaled(16, 16, Qt::KeepAspectRatio,
                                          Qt::SmoothTransformation));
        m_roleIcon->setProperty("provider", provider);
    }
    void applyStyles(bool dark) {
        m_dark = dark;
        m_bubble->setStyleSheet(chatStyle(m_role == "user"
                                                 ? "message_bubble_user"
                                                 : "message_bubble_assistant",
                                         dark));
        if (m_roleFrame) {
            m_roleFrame->setStyleSheet(chatStyle("message_role_background", dark));
            m_roleIcon->setStyleSheet(chatStyle("message_role", dark));
        }
        m_content->setStyleSheet(chatStyle("message_content", dark));
        for (auto* button : m_buttons)
            button->setStyleSheet(chatStyle("message_button", dark));
        // The native plain-text editor preserves the source's input contract;
        // adapt only the source selector to its Qt widget type.
        QString editor = chatStyle("message_editor", dark);
        editor.replace("QTextEdit", "QPlainTextEdit");
        m_editor->setStyleSheet(editor);
        m_editControls->setStyleSheet(chatStyle("message_edit_controls", dark));
        m_resend->setStyleSheet(chatStyle("message_resend", dark));
        m_cancel->setStyleSheet(chatStyle("message_cancel", dark));
        m_save->setStyleSheet(chatStyle("message_save", dark));
    }
    void setText(const QString& text) {
        if (m_text == text) return;
        if (m_editing) endEdit();
        m_text = text;
        setProperty("content", text);
        if (m_role == "user") {
            m_content->setTextFormat(Qt::PlainText);
            m_content->setText(text);
        } else {
            QTextDocument document;
            qtCompatSetMarkdown(document, text);
            m_content->setTextFormat(Qt::RichText);
            m_content->setText(document.toHtml());
        }
    }
    void setActionsEnabled(bool enabled) {
        for (auto* button : m_buttons) button->setEnabled(enabled);
        m_editControls->setEnabled(enabled);
    }

private:
    void copyContent(bool feedback) {
        QApplication::clipboard()->setText(m_text);
        if (feedback) {
            m_copy->setIcon(QIcon(":/CC/plugin/qAnyLabeling/images/check.svg"));
            setProperty("copyFeedback", true);
            m_copyReset->start();
        }
    }
    bool event(QEvent* event) override {
        if (event->type() == QEvent::Enter) {
            m_hovered = true;
            updateButtons();
        }
        return QWidget::event(event);
    }
    void leaveEvent(QEvent* event) override {
        m_hovered = false;
        updateButtons();
        QWidget::leaveEvent(event);
    }
    void endEdit() {
        m_editing = false;
        m_editor->hide();
        m_editControls->hide();
        m_content->show();
        updateButtons();
    }
    void updateButtons() {
        for (auto* button : m_buttons)
            button->setVisible(m_hovered && !m_editing);
    }
    QString m_role, m_text;
    QString m_provider;
    Action m_action;
    QWidget* m_bubble;
    QFrame* m_roleFrame = nullptr;
    QLabel* m_roleIcon = nullptr;
    QLabel* m_content;
    QPlainTextEdit* m_editor;
    QWidget* m_editControls;
    QVector<QPushButton*> m_buttons;
    QPushButton *m_copy, *m_resend, *m_cancel, *m_save;
    QTimer* m_copyReset;
    bool m_dark = false;
    bool m_hovered = false, m_editing = false;
};

class ChatbotDialog final : public QDialog {
public:
    using Write = std::function<bool(const Source&, const QJsonArray&)>;
    using BatchWrite = std::function<bool(
            const QString&, quint64, const QJsonArray&, QString*)>;
    ChatbotDialog(QWidget* parent,
                  std::function<Source()> source,
                  Write write,
                  std::function<QStringList()> files,
                  std::function<void(int)> navigate,
                  std::function<void(bool)> openMedia,
                  BatchWrite batchWrite,
                  std::function<int(const QString&, QString*)> importDataset)
        : QDialog(parent),
          m_source(std::move(source)),
          m_write(std::move(write)),
          m_files(std::move(files)),
          m_batchWrite(std::move(batchWrite)),
          m_importDataset(std::move(importDataset)) {
        setObjectName(QStringLiteral("AnnotationChatbot"));
        setPalette(parent->palette());
        setStyleSheet(parent->styleSheet());
        setWindowTitle(chatTr("Chatbot"));
        setAttribute(Qt::WA_DeleteOnClose);
        setWindowFlags(Qt::Window | Qt::WindowMinMaxButtonsHint |
                       Qt::WindowCloseButtonHint);
        resize(1200, 700);
        auto* root = new QHBoxLayout(this);
        root->setContentsMargins(0, 0, 0, 0);
        root->setSpacing(0);
        auto* splitter = new QSplitter(Qt::Horizontal, this);
        splitter->setObjectName("ChatMainSplitter");
        splitter->setHandleWidth(1);
        root->addWidget(splitter);
        auto* right = new QWidget(splitter);
        right->setObjectName("ChatPreviewSettingsPanel");
        right->setFixedWidth(360);
        auto* rightLayout = new QVBoxLayout(right);
        rightLayout->setContentsMargins(0, 0, 0, 0);
        rightLayout->setSpacing(0);
        auto* rightSplitter = new QSplitter(Qt::Vertical, right);
        rightSplitter->setObjectName("ChatRightSplitter");
        rightSplitter->setHandleWidth(1);
        rightLayout->addWidget(rightSplitter);
        auto* preview = new QWidget(rightSplitter);
        auto* previewLayout = new QVBoxLayout(preview);
        previewLayout->setContentsMargins(24, 24, 24, 16);
        m_preview = new QLabel(preview);
        m_preview->setObjectName("ChatImagePreview");
        m_preview->setMinimumHeight(200);
        m_preview->setAlignment(Qt::AlignCenter);
        previewLayout->addWidget(m_preview, 1);
        auto* navigation = new QHBoxLayout;
        previewLayout->addLayout(navigation);
        const auto mediaButton = [preview](const QString& name,
                                           const QString& icon,
                                           const QString& tooltip) {
            auto* button = new QPushButton(preview);
            button->setObjectName(name);
            button->setFixedSize(32, 32);
            button->setIcon(QIcon(":/CC/plugin/qAnyLabeling/images/" + icon +
                                  ".svg"));
            button->setToolTip(tooltip);
            return button;
        };
        auto* previous = mediaButton("ChatPreviousImage", "arrow-left",
                                     chatTr("Previous Image"));
        auto* next = mediaButton("ChatNextImage", "arrow-right",
                                 chatTr("Next Image"));
        auto* image = mediaButton("ChatOpenImage", "image",
                                  chatTr("Open Image File"));
        auto* folder = mediaButton("ChatOpenFolder", "folder-chatbot",
                                   chatTr("Open Image Folder"));
        navigation->addWidget(previous);
        navigation->addStretch();
        navigation->addWidget(image);
        navigation->addWidget(folder);
        navigation->addStretch();
        navigation->addWidget(next);
        connect(previous, &QPushButton::clicked, this,
                [navigate]() { navigate(-1); });
        connect(next, &QPushButton::clicked, this,
                [navigate]() { navigate(1); });
        connect(image, &QPushButton::clicked, this,
                [openMedia]() { openMedia(false); });
        connect(folder, &QPushButton::clicked, this,
                [openMedia]() { openMedia(true); });
        auto* settings = new ServicePanel(rightSplitter);
        m_service = settings;
        settings->useChatSettingsTabs(right);
        rightSplitter->setSizes({300, 400});
        auto* providers = settings->createChatProviders(splitter);
        auto* middle = new QWidget(splitter);
        middle->setObjectName("ChatConversationPanel");
        auto* layout = new QVBoxLayout(middle);
        layout->setContentsMargins(0, 0, 0, 0);
        layout->setSpacing(0);
        splitter->insertWidget(0, providers);
        splitter->insertWidget(1, middle);
        splitter->setSizes({200, 700, 300});
        splitter->setStretchFactor(0, 0);
        splitter->setStretchFactor(1, 1);
        splitter->setStretchFactor(2, 0);
        m_historyView = new QScrollArea(middle);
        m_historyView->setObjectName(QStringLiteral("ChatHistory"));
        m_historyView->setWidgetResizable(true);
        m_historyView->setFrameShape(QFrame::NoFrame);
        m_historyView->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
        auto* messages = new QWidget(m_historyView);
        messages->setObjectName("ChatMessagesContainer");
        m_messages = new QVBoxLayout(messages);
        m_messages->setContentsMargins(24, 12, 24, 12);
        m_messages->setSpacing(16);
        m_messages->addStretch();
        m_historyView->setWidget(messages);
        auto* chatContainer = new QWidget(middle);
        chatContainer->setObjectName("ChatConversationContainer");
        auto* chatLayout = new QVBoxLayout(chatContainer);
        chatLayout->setContentsMargins(0, 20, 0, 20);
        chatLayout->setSpacing(16);
        chatLayout->addWidget(m_historyView);
        layout->addWidget(chatContainer, 88);
        auto* inputContainer = new QWidget(middle);
        inputContainer->setObjectName("ChatInputContainer");
        auto* inputLayout = new QVBoxLayout(inputContainer);
        inputLayout->setContentsMargins(24, 12, 24, 12);
        inputLayout->setSpacing(0);
        auto* inputFrame = new QFrame(inputContainer);
        inputFrame->setObjectName("inputFrame");
        auto* inputFrameLayout = new QVBoxLayout(inputFrame);
        inputFrameLayout->setContentsMargins(12, 8, 12, 8);
        inputFrameLayout->setSpacing(0);
        inputLayout->addWidget(inputFrame);
        layout->addWidget(inputContainer, 12);
        m_input = new QTextEdit(inputFrame);
        m_input->setObjectName(QStringLiteral("ChatInput"));
        m_input->setPlaceholderText(
                chatTr("Type a message and press Enter to send."));
        m_input->setAcceptRichText(false);
        m_input->setFrameShape(QFrame::NoFrame);
        m_input->setFrameShadow(QFrame::Plain);
        m_input->document()->setDocumentMargin(0);
        m_input->setLineWrapMode(QTextEdit::WidgetWidth);
        m_input->installEventFilter(this);
        inputFrameLayout->addWidget(m_input, 1);
        auto* controls = new QHBoxLayout;
        inputFrameLayout->addLayout(controls);
        controls->setContentsMargins(0, 4, 0, 0);
        controls->setSpacing(8);
        auto* importButton = new QPushButton(chatTr("Import"), this);
        auto* exportButton = new QPushButton(chatTr("Export"), this);
        auto* exportDataset = new QPushButton(chatTr("Export Dataset"), this);
        exportDataset->setObjectName("ChatExportDataset");
        auto* importDatasetButton =
                new QPushButton(chatTr("Import Dataset"), this);
        importDatasetButton->setObjectName("ChatImportDataset");
        m_autoRun = new QPushButton(chatTr("AutoRun"), this);
        m_autoRun->setObjectName("ChatAutoRun");
        auto* clear = new QPushButton(chatTr("Clear"), this);
        clear->setObjectName("ChatClear");
        m_send = new QPushButton(chatTr("Send"), this);
        m_send->setObjectName(QStringLiteral("ChatSend"));
        for (auto* button : {clear, m_send}) {
            button->setToolTip(button->text());
            button->setText({});
            button->setFixedSize(16, 16);
            button->setIconSize(QSize(16, 16));
        }
        clear->setIcon(QIcon(":/CC/plugin/qAnyLabeling/images/eraser.svg"));
        clear->installEventFilter(this);
        connect(m_input, &QTextEdit::textChanged, this, [this]() {
            resizeInput();
            updateSendState();
            const auto original = m_input->textCursor();
            auto cursor = original;
            const QString text = m_input->toPlainText();
            const bool blocked = m_input->blockSignals(true);
            cursor.select(QTextCursor::Document);
            cursor.setCharFormat(QTextCharFormat());
            // The source highlights the first @image token while retaining
            // plain text as the request and sidecar representation.
            const int start = text.indexOf("@image");
            if (start >= 0) {
                QTextCharFormat highlight;
                highlight.setBackground(QColor("#E3F2FD"));
                highlight.setForeground(QColor("#1976D2"));
                cursor.setPosition(start);
                cursor.setPosition(start + 6, QTextCursor::KeepAnchor);
                cursor.setCharFormat(highlight);
            }
            m_input->setTextCursor(original);
            m_input->blockSignals(blocked);
        });
        // Dataset operations stay with image/settings controls, leaving the
        // expanding conversation column available for messages and input.
        auto* datasets = new QHBoxLayout;
        previewLayout->addLayout(datasets);
        const QVector<QPair<QPushButton*, QString>> datasetButtons{
                {importButton, "open"}, {exportButton, "export"},
                {importDatasetButton, "import-export"},
                {exportDataset, "download"}, {m_autoRun, "run"}};
        for (const auto& item : datasetButtons) {
            auto* button = item.first;
            button->setToolTip(button->text());
            button->setText({});
            button->setFixedSize(32, 32);
            button->setIcon(QIcon(":/CC/plugin/qAnyLabeling/images/" +
                                  item.second + ".svg"));
            datasets->addWidget(button);
        }
        controls->addWidget(clear);
        controls->addStretch();
        controls->addWidget(m_send);
        m_progress = new QProgressDialog(chatTr("Inferencing..."),
                                         chatTr("Cancel"), 0, 0, this);
        m_progress->setObjectName("ChatBatchProgress");
        m_progress->setWindowModality(Qt::WindowModal);
        m_progress->setMinimumDuration(0);
        m_progress->hide();
        connect(m_progress, &QProgressDialog::canceled, this,
                [this]() { cancelBatch(); });
        connect(m_autoRun, &QPushButton::clicked, this,
                [this]() { configureBatch(); });
        connect(importDatasetButton, &QPushButton::clicked, this, [this]() {
            if (m_batchActive || m_service->client->busy()) return;
            const auto file = QFileDialog::getOpenFileName(
                    this, chatTr("Select Dataset File"),
                    QFileInfo(m_source().path).absolutePath(), "JSON (*.json)");
            if (file.isEmpty()) return;
            QString error;
            const int count = m_importDataset(file, &error);
            m_service->status->setText(
                    count < 0 ? error
                    : count   ? chatTr("Successfully imported %1 items.")
                                      .arg(count)
                            : chatTr("No valid items were found to import."));
        });
        connect(exportDataset, &QPushButton::clicked, this, [this]() {
            if (m_batchActive || m_service->client->busy()) return;
            const auto source = m_source();
            const auto directory = QFileDialog::getExistingDirectory(
                    this, chatTr("Select Export Directory"),
                    QFileInfo(source.path).absolutePath());
            if (directory.isEmpty()) return;
            QString error;
            const auto archive = exportChatDataset(source, directory, &error);
            m_service->status->setText(
                    archive.isEmpty()
                            ? error
                            : chatTr("Dataset exported successfully to:\n%1")
                                      .arg(archive));
        });
        connect(m_send, &QPushButton::clicked, this, [this]() {
            if (m_batchActive) return;
            if (m_service->client->busy()) {
                stopGeneration();
                return;
            }
            QString text = m_input->toPlainText().trimmed();
            if (text.isEmpty()) return;
            const auto source = m_source();
            const bool image = text.contains("@image") || text.contains("<image>");
            if (image && source.path.isEmpty()) {
                m_service->status->setText(
                        chatTr("No image available for @image reference."));
                return;
            }
            text.replace("@image", "<image>");
            if (image && !text.contains("<image>")) text.prepend("<image> ");
            // Sidecars use the upstream role/content/image schema. Base64
            // belongs only in the service request, never in chat_history.
            const QJsonObject entry{
                    {"role", "user"},
                    {"content", text},
                    {"image", image ? QJsonValue(source.path)
                                    : QJsonValue(QJsonValue::Null)}};
            auto conversation = m_history;
            conversation.append(entry);
            generate(conversation, source);
        });
        connect(settings->client, &ChatServiceClient::textReceived, this,
                [this](const QString& delta) {
                    m_partial += delta;
                    render();
                });
        connect(settings->client, &ChatServiceClient::finished, this,
                [this](const QString& text, const QString& error) {
                    if (error.isEmpty()) {
                        m_history.append(
                                QJsonObject{{"role", "assistant"},
                                            {"content", text},
                                            {"image", QJsonValue::Null}});
                        persist();
                    }
                    m_partial.clear();
                    updateSendState();
                    m_service->status->setText(error);
                    render();
                });
        connect(clear, &QPushButton::clicked, this, [this]() {
            QMessageBox confirm(this);
            confirm.setText(chatTr(
                    "Are you sure you want to clear the entire conversation?"));
            confirm.setStandardButtons(QMessageBox::Yes | QMessageBox::No);
            confirm.setDefaultButton(QMessageBox::No);
            confirm.setIcon(QMessageBox::Warning);
            if (confirm.exec() != QMessageBox::Yes) return;
            cancelBatch();
            m_service->client->cancel();
            m_history = {};
            m_partial.clear();
            persist();
            updateSendState();
            render();
        });
        connect(exportButton, &QPushButton::clicked, this, [this]() {
            const auto path = QFileDialog::getSaveFileName(
                    this, chatTr("Export chat"), "chat.json", "JSON (*.json)");
            if (path.isEmpty()) return;
            QSaveFile file(path);
            if (!file.open(QIODevice::WriteOnly) ||
                file.write(QJsonDocument(m_history).toJson()) < 0 ||
                !file.commit())
                m_service->status->setText(chatTr("Unable to export chat."));
        });
        connect(importButton, &QPushButton::clicked, this, [this]() {
            if (m_batchActive) return;
            const auto path = QFileDialog::getOpenFileName(
                    this, chatTr("Import chat"), {}, "JSON (*.json)");
            if (path.isEmpty()) return;
            QFile file(path);
            if (!file.open(QIODevice::ReadOnly)) return;
            const auto json = QJsonDocument::fromJson(file.readAll());
            if (!json.isArray()) {
                m_service->status->setText(chatTr("Invalid chat JSON."));
                return;
            }
            for (const auto& value : json.array()) {
                const auto message = value.toObject();
                if (!message.value("role").isString() ||
                    !message.value("content").isString() ||
                    (!message.value("image").isUndefined() &&
                     !message.value("image").isNull() &&
                     !message.value("image").isString())) {
                    m_service->status->setText(chatTr("Invalid chat message."));
                    return;
                }
            }
            m_service->client->cancel();
            m_partial.clear();
            m_history = json.array();
            persist();
            updateSendState();
            render();
        });
        for (auto* edit : {m_service->address, m_service->key,
                            m_service->model->lineEdit()})
            edit->installEventFilter(this);
        if (auto* selector = findChild<QPushButton*>("ChatBrowseModels"))
            selector->installEventFilter(this);
        m_ready = true;
        applyChatStyles();
        resizeInput();
        sourceChanged();
    }
    void sourceChanged() {
        cancelBatch();
        m_service->client->cancel();
        m_partial.clear();
        updateSendState();
        m_service->status->setText(
                m_service->status->property("configurationError").toString());
        m_current = m_source();
        updatePreview();
        for (auto* widget : m_messageViews) {
            m_messages->removeWidget(widget);
            widget->hide();
            widget->deleteLater();
        }
        m_messageViews.clear();
        m_history = {};
        for (const auto& value :
             m_current.document.value("chat_history").toArray()) {
            auto message = value.toObject();
            const auto role = message.value("role").toString();
            if (message.value("role").isString() &&
                message.value("content").isString()) {
                // The source rebinds image tokens to the active image when
                // loading a sidecar, so moving a dataset preserves chat.
                const auto content = message.value("content").toString();
                if (role == "user" &&
                    (content.contains("@image") || content.contains("<image>")))
                    message.insert("image", m_current.path);
                m_history.append(message);
            }
        }
        render();
    }

private:
    bool eventFilter(QObject* object, QEvent* event) override {
        if (event->type() == QEvent::KeyPress) {
            const auto* key = static_cast<QKeyEvent*>(event);
            if (key->key() == Qt::Key_Return || key->key() == Qt::Key_Enter) {
                if (object == m_input) {
                    if (key->modifiers() & Qt::ShiftModifier) return false;
                    if (!m_service->client->busy()) m_send->click();
                    return true;
                }
                return true;
            }
        }
        return QDialog::eventFilter(object, event);
    }
    void resizeEvent(QResizeEvent* event) override {
        QDialog::resizeEvent(event);
        updatePreview();
        if (m_ready) resizeInput();
    }
    void changeEvent(QEvent* event) override {
        QDialog::changeEvent(event);
        if (m_ready && !m_styling && !m_stylePending &&
            (event->type() == QEvent::PaletteChange ||
             event->type() == QEvent::StyleChange)) {
            // The workspace may still be propagating its global stylesheet.
            // Apply the source's per-control overrides after that completes.
            m_stylePending = true;
            QTimer::singleShot(0, this, [this]() {
                m_stylePending = false;
                applyChatStyles();
            });
        }
    }
    void applyChatStyles() {
        // Palette/style events can arrive while the dialog is still being
        // constructed. Only polish the owned controls after they all exist.
        if (!m_ready || m_styling) return;
        m_styling = true;
        // This dialog's QSS also changes its palette. The workspace owns the
        // selected theme; reading our styled palette can lock it in dark mode.
        const bool dark = parentWidget()->palette().color(QPalette::Window)
                                  .lightness() < 128;
        const QString dialog =
                chatStyle("dialog", dark) + chatStyle("menu", dark);
        if (styleSheet() != dialog) setStyleSheet(dialog);
        const auto apply = [this, dark](const char* name, const char* style) {
            if (auto* widget = findChild<QWidget*>(QString::fromLatin1(name)))
                widget->setStyleSheet(chatStyle(style, dark));
        };
        apply("ChatMainSplitter", "main_splitter");
        apply("ChatProvidersPanel", "left_widget");
        apply("ChatConversationPanel", "middle_widget");
        apply("ChatHistory", "chat_scroll_area");
        apply("ChatConversationContainer", "chat_container");
        apply("ChatMessagesContainer", "chat_container");
        apply("ChatInputContainer", "input_container");
        apply("inputFrame", "input_frame");
        apply("ChatInput", "message_input");
        apply("ChatPreviewSettingsPanel", "right_widget");
        apply("ChatRightSplitter", "right_splitter");
        apply("ChatImagePreview", "image_preview");
        apply("ChatBrowseModels", "model_button");
        apply("ChatAddress", "settings_edit");
        apply("ChatApiKey", "settings_edit");
        apply("ChatSystemPrompt", "settings_edit");
        apply("ChatTemperatureSlider", "slider");
        if (auto* panel = findChild<QGroupBox*>("ChatSettingsPanel"))
            panel->setStyleSheet("QGroupBox { border: none; margin: 0; "
                                "padding: 0; }");
        if (auto* tabs = findChild<QTabWidget*>("ChatSettingsTabs")) {
            tabs->setStyleSheet(chatStyle("tab_widget", dark));
            tabs->tabBar()->setStyleSheet(chatStyle("settings_tabs", dark));
        }
        for (auto* button : findChildren<QPushButton*>())
            button->setStyleSheet(chatStyle(
                    button->objectName().startsWith("ChatProviderButton_")
                            ? "provider_button"
                            : "button",
                    dark));
        apply("ChatBrowseModels", "model_button");
        apply("ChatSend", "send_button");
        apply("ChatClear", "send_button");
        for (auto* message : m_messageViews)
            if (message) message->applyStyles(dark);
        m_styling = false;
    }
    void resizeInput() {
        const auto margins = m_input->contentsMargins();
        const int height = int(m_input->document()->size().height()) +
                           margins.top() + margins.bottom() + 4;
        m_input->setFixedHeight(std::max(20, std::min(300, height)));
        m_input->setVerticalScrollBarPolicy(
                height > 300 ? Qt::ScrollBarAlwaysOn : Qt::ScrollBarAlwaysOff);
        if (height > 300) m_input->ensureCursorVisible();
    }
    void updateSendState() {
        const bool streaming = m_service->client->busy();
        m_input->setEnabled(!m_batchActive && !streaming);
        m_send->setEnabled(!m_batchActive &&
                          (streaming || !m_input->toPlainText().trimmed().isEmpty()));
        m_send->setIcon(QIcon(streaming
                                    ? ":/CC/plugin/qAnyLabeling/images/stop.svg"
                                    : ":/CC/plugin/qAnyLabeling/images/send.svg"));
        m_send->setToolTip(chatTr(streaming ? "Stop" : "Send"));
    }
    void stopGeneration() {
        // cancel() suppresses the old reply's callbacks; retain a received
        // partial answer using the same sidecar representation as completion.
        m_service->client->cancel();
        if (!m_partial.isEmpty())
            m_history.append(QJsonObject{{"role", "assistant"},
                                         {"content", m_partial},
                                         {"image", QJsonValue::Null}});
        persist();
        m_partial.clear();
        updateSendState();
        render();
        m_service->status->setText(chatTr("Cancelled"));
    }
    void updatePreview() {
        if (!m_preview) return;
        m_preview->setPixmap(QPixmap::fromImage(m_current.image)
                                     .scaled(m_preview->size(),
                                             Qt::KeepAspectRatio,
                                             Qt::SmoothTransformation));
    }
    bool generate(const QJsonArray& conversation, const Source& source) {
        const auto config = m_service->settings();
        if (config.model.isEmpty()) {
            m_service->status->setText(chatTr("Choose a model."));
            return false;
        }
        QString error;
        QJsonArray wire;
        for (const auto& value : conversation) {
            const auto message = value.toObject();
            if (message.value("role") == "user") {
                wire.append(ChatServiceClient::userMessage(
                        message.value("content").toString(),
                        message.value("image").toString(), &error));
                if (!error.isEmpty()) {
                    m_service->status->setText(error);
                    return false;
                }
            } else
                wire.append(QJsonObject{{"role", message.value("role")},
                                        {"content", message.value("content")}});
        }
        m_current = source;
        m_history = conversation;
        persist();
        m_input->clear();
        m_partial.clear();
        m_service->status->setText(chatTr("Generating…"));
        auto request = config;
        request.timeoutMs = 300000;
        m_service->client->complete(request, m_service->messages(wire));
        updateSendState();
        render();
        return true;
    }
    void messageAction(ChatMessage* widget,
                       const QString& action,
                       const QString& text) {
        if (m_batchActive || m_service->client->busy()) return;
        const int index = m_messageViews.indexOf(widget);
        if (index < 0 || index >= m_history.size()) return;
        if (action == "delete") {
            m_history.removeAt(index);
        } else if (action == "save") {
            auto entry = m_history[index].toObject();
            entry.insert("content", text);
            m_history[index] = entry;
        } else if (action == "resend" || action == "regenerate") {
            QJsonArray conversation;
            for (int i = 0; i < index; ++i) conversation.append(m_history[i]);
            if (action == "resend") {
                const auto source = m_source();
                QString content = text;
                const bool image = content.contains("@image") ||
                                   content.contains("<image>");
                content.replace("@image", "<image>");
                conversation.append(QJsonObject{
                        {"role", "user"},
                        {"content", content},
                        {"image", image ? QJsonValue(source.path)
                                        : QJsonValue(QJsonValue::Null)}});
            }
            if (!conversation.isEmpty()) generate(conversation, m_source());
            return;
        }
        persist();
        render();
    }
    void reject() override {
        cancelBatch();
        m_service->client->cancel();
        QDialog::reject();
    }
    void closeEvent(QCloseEvent* event) override {
        cancelBatch();
        m_service->client->cancel();
        QDialog::closeEvent(event);
    }
    void cancelBatch() {
        if (!m_batchActive) return;
        m_batchActive = false;
        for (auto* client : m_batchClients) {
            client->disconnect(this);
            client->cancel();
            client->deleteLater();
        }
        m_batchClients.clear();
        m_batchFiles.clear();
        m_progress->hide();
        updateSendState();
        m_autoRun->setEnabled(true);
        m_service->status->setText(chatTr("Cancelled"));
        render();
    }
    void configureBatch() {
        if (m_batchActive || m_service->client->busy()) return;
        const auto files = m_files();
        if (files.isEmpty()) return;
        QDialog dialog(this);
        dialog.setObjectName("ChatBatchOptions");
        dialog.setWindowTitle(chatTr("Batch Processing"));
        auto* form = new QFormLayout(&dialog);
        QPlainTextEdit prompt(&dialog);
        prompt.setObjectName("ChatBatchPrompt");
        prompt.setPlainText(m_input->toPlainText());
        QSpinBox concurrency(&dialog);
        concurrency.setObjectName("ChatBatchConcurrency");
        const int cores = std::max(1, QThread::idealThreadCount());
        concurrency.setRange(1, std::max(1, int(cores * .95)));
        concurrency.setValue(std::max(1, int(cores * .8)));
        form->addRow(chatTr("Prompt:"), &prompt);
        form->addRow(chatTr("Concurrency:"), &concurrency);
        QDialogButtonBox buttons(
                QDialogButtonBox::Ok | QDialogButtonBox::Cancel, &dialog);
        form->addRow(&buttons);
        connect(&buttons, &QDialogButtonBox::accepted, &dialog, [&]() {
            if (!prompt.toPlainText().trimmed().isEmpty()) dialog.accept();
        });
        connect(&buttons, &QDialogButtonBox::rejected, &dialog,
                &QDialog::reject);
        if (dialog.exec() != QDialog::Accepted) return;
        m_batchSettings = m_service->settings();
        if (m_batchSettings.model.isEmpty()) {
            m_service->status->setText(chatTr("Choose a model."));
            return;
        }
        m_batchSettings.timeoutMs = 300000;
        m_batchSystem = m_service->system->text().trimmed();
        m_batchPrompt = prompt.toPlainText().trimmed();
        m_batchPrompt.replace(QRegularExpression("@image\\s*(?=\\S|$)"),
                              "<image>");
        m_batchFiles = files;
        m_batchGeneration = m_source().generation;
        m_batchNext = m_batchDone = m_batchFailures = 0;
        m_batchLimit = concurrency.value();
        m_batchActive = true;
        render();
        updateSendState();
        m_autoRun->setEnabled(false);
        m_progress->setRange(0, files.size());
        m_progress->setValue(0);
        m_progress->show();
        pumpBatch();
    }
    void pumpBatch() {
        if (!m_batchActive) return;
        // O(n) submissions; at most limit encoded images are held by requests.
        while (m_batchNext < m_batchFiles.size() &&
               m_batchClients.size() < m_batchLimit) {
            const QString path = m_batchFiles[m_batchNext++];
            QString error;
            const auto user = ChatServiceClient::userMessage(
                    m_batchPrompt,
                    m_batchPrompt.contains("<image>") ? path : QString(),
                    &error);
            if (!error.isEmpty()) {
                ++m_batchDone;
                ++m_batchFailures;
                m_service->status->setText(error);
                continue;
            }
            QJsonArray messages;
            if (!m_batchSystem.isEmpty())
                messages.append(QJsonObject{{"role", "system"},
                                            {"content", m_batchSystem}});
            messages.append(user);
            auto* client = new ChatServiceClient(this);
            m_batchClients.append(client);
            connect(client, &ChatServiceClient::finished, this,
                    [this, client, path](const QString& answer,
                                         const QString& failure) {
                        m_batchClients.removeOne(client);
                        client->deleteLater();
                        if (!m_batchActive) return;
                        QString error = failure;
                        const QJsonArray history{
                                QJsonObject{{"role", "user"},
                                            {"content", m_batchPrompt},
                                            {"image", path}},
                                QJsonObject{{"role", "assistant"},
                                            {"content", answer},
                                            {"image", QJsonValue::Null}}};
                        if (error.isEmpty() && !answer.isEmpty() &&
                            !m_batchWrite(path, m_batchGeneration, history,
                                          &error)) {
                            if (error.isEmpty())
                                error = chatTr("Unable to save chat result.");
                        }
                        if (!error.isEmpty() || answer.isEmpty())
                            ++m_batchFailures;
                        ++m_batchDone;
                        if (!error.isEmpty()) m_service->status->setText(error);
                        // Modal progress updates process queued events,
                        // including the pump that may publish the final batch
                        // summary.
                        m_progress->setValue(m_batchDone);
                        QTimer::singleShot(0, this, [this]() { pumpBatch(); });
                    });
            client->complete(m_batchSettings, messages, false);
        }
        if (m_batchDone == m_batchFiles.size() && m_batchClients.isEmpty()) {
            m_batchActive = false;
            m_progress->hide();
            m_autoRun->setEnabled(true);
            sourceChanged();
            m_service->status->setText(
                    chatTr("Processed %1 images successfully.\n%2 images "
                           "failed.")
                            .arg(m_batchDone - m_batchFailures)
                            .arg(m_batchFailures));
        }
    }
    void persist() {
        if (!m_current.path.isEmpty()) m_write(m_current, m_history);
    }
    void render() {
        // O(n) history reconciliation; streaming updates only reformat the
        // changing final message rather than rebuilding every document.
        const int count = m_history.size() + (!m_partial.isEmpty() ? 1 : 0);
        while (m_messageViews.size() > count) {
            auto* widget = m_messageViews.takeLast();
            m_messages->removeWidget(widget);
            widget->hide();
            widget->deleteLater();
        }
        for (int i = 0; i < count; ++i) {
            const bool partial = i == m_history.size();
            const auto message = partial ? QJsonObject{{"role", "assistant"},
                                                       {"content", m_partial}}
                                         : m_history[i].toObject();
            const auto role = message.value("role").toString();
            if (i < m_messageViews.size() &&
                m_messageViews[i]->property("role").toString() != role) {
                auto* old = m_messageViews[i];
                m_messages->removeWidget(old);
                old->hide();
                old->deleteLater();
                m_messageViews[i] = nullptr;
            }
            if (i == m_messageViews.size()) m_messageViews.append(nullptr);
            if (!m_messageViews[i]) {
                auto* widget = new ChatMessage(
                        role,
                        [this](ChatMessage* widget, const QString& action,
                               const QString& text) {
                            messageAction(widget, action, text);
                        },
                        m_historyView->widget());
                m_messageViews[i] = widget;
                m_messages->insertWidget(i, widget);
            }
            QString text;
            if (message.value("content").isString())
                text = message.value("content").toString();
            else
                for (const auto& part : message.value("content").toArray())
                    if (part.toObject().value("type") == "text")
                        text += part.toObject().value("text").toString();
            m_messageViews[i]->setText(text);
            m_messageViews[i]->setProvider(m_service->provider->currentText());
            m_messageViews[i]->setActionsEnabled(!partial && !m_batchActive &&
                                                 !m_service->client->busy());
        }
        auto* bar = m_historyView->verticalScrollBar();
        QTimer::singleShot(0, bar, [bar]() { bar->setValue(bar->maximum()); });
    }
    std::function<Source()> m_source;
    Write m_write;
    std::function<QStringList()> m_files;
    BatchWrite m_batchWrite;
    std::function<int(const QString&, QString*)> m_importDataset;
    QVector<ChatServiceClient*> m_batchClients;
    QStringList m_batchFiles;
    QString m_batchPrompt, m_batchSystem;
    ChatServiceClient::Settings m_batchSettings;
    QProgressDialog* m_progress;
    QPushButton* m_autoRun;
    quint64 m_batchGeneration = 0;
    int m_batchNext = 0, m_batchDone = 0, m_batchFailures = 0, m_batchLimit = 1;
    bool m_batchActive = false;
    Source m_current;
    ServicePanel* m_service;
    QLabel* m_preview = nullptr;
    bool m_ready = false;
    bool m_styling = false;
    bool m_stylePending = false;
    QScrollArea* m_historyView;
    QVBoxLayout* m_messages;
    QVector<ChatMessage*> m_messageViews;
    QTextEdit* m_input;
    QPushButton* m_send;
    QJsonArray m_history;
    QString m_partial;
};

// Pure source QSS, embedded with Chat styles at build time.
QString vqaStyle(const char* name, const QWidget* owner) {
    const auto* themeOwner = owner->parentWidget() ? owner->parentWidget() : owner;
    return chatStyle(name, themeOwner->palette().color(QPalette::Window)
                                  .lightness() < 128);
}
QString vqaButtonStyle(const QWidget* owner, bool primary) {
    return vqaStyle(primary ? "vqa_dialog_button_primary"
                            : "vqa_dialog_button_secondary", owner);
}

class VqaLoadingDialog final : public QDialog {
public:
    explicit VqaLoadingDialog(QWidget* parent) : QDialog(parent) {
        setObjectName("VQALoadingDialog");
        setWindowTitle(chatTr("AI Processing"));
        setFixedSize(420, 180);
        setWindowFlags(Qt::Dialog | Qt::FramelessWindowHint |
                       Qt::WindowStaysOnTopHint);
        setModal(true);
        auto* layout = new QVBoxLayout(this);
        layout->setContentsMargins(24, 24, 24, 24);
        layout->setSpacing(16);
        auto* content = new QVBoxLayout;
        content->setSpacing(12);
        auto* title = new QLabel(chatTr("AI Processing"), this);
        title->setAlignment(Qt::AlignCenter);
        title->setStyleSheet(vqaStyle("vqa_title_label", this));
        content->addWidget(title);
        auto* message =
                new QLabel(chatTr("Generating content, please wait."), this);
        message->setAlignment(Qt::AlignCenter);
        message->setStyleSheet(vqaStyle("vqa_message_label", this));
        content->addWidget(message);
        layout->addLayout(content);
        layout->addStretch();
        auto* buttons = new QHBoxLayout;
        buttons->setContentsMargins(0, 0, 0, 0);
        buttons->addStretch();
        auto* cancel = new QPushButton(chatTr("Cancel"), this);
        cancel->setObjectName("VQALoadingCancel");
        cancel->setCursor(Qt::PointingHandCursor);
        cancel->setStyleSheet(vqaButtonStyle(this, false));
        buttons->addWidget(cancel);
        layout->addLayout(buttons);
        connect(cancel, &QPushButton::clicked, this, &QDialog::reject);
        auto* shadow = new QGraphicsDropShadowEffect(this);
        shadow->setBlurRadius(15);
        shadow->setColor(QColor(0, 0, 0, 25));
        shadow->setOffset(0, 4);
        setGraphicsEffect(shadow);
        auto* timer = new QTimer(this);
        connect(timer, &QTimer::timeout, this, [message, dot = 1]() mutable {
            message->setText(chatTr("Generating content, please wait") +
                             QString(dot, '.') + QString(3 - dot, ' '));
            dot = dot % 3 + 1;
        });
        connect(this, &QDialog::finished, timer, &QTimer::stop);
        timer->start(300);
        move(parent->mapToGlobal(parent->rect().center()) - rect().center());
    }
};

class VqaAiPromptDialog final : public QDialog {
public:
    VqaAiPromptDialog(QWidget* parent, const QStringList& textFields)
        : QDialog(parent) {
        setObjectName("VQAAiPromptDialog");
        setWindowTitle(chatTr("AI Assistance"));
        setMinimumWidth(600);
        auto* layout = new QVBoxLayout(this);
        layout->setContentsMargins(24, 24, 24, 24);
        layout->setSpacing(12);
        m_prompt = new QTextEdit(this);
        m_prompt->setObjectName("VQAPrompt");
        m_prompt->setAcceptRichText(true);
        m_prompt->setStyleSheet(vqaStyle("vqa_prompt_input", this));
        m_prompt->setPlaceholderText(
                chatTr("Examples:\n"
                       "   1. @image Describe this image\n"
                       "   2. Translate to English, return translated text "
                       "only: @text\n"
                       "   3. @widget.title - Reference widget values\n"
                       "   4. @label.shapes - Reference annotation data"));
        m_prompt->setMinimumHeight(120);
        m_prompt->setMaximumHeight(160);
        m_prompt->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
        layout->addWidget(m_prompt);
        QStringList tags{"@image",
                         "@text",
                         "@label.shapes",
                         "@label.imagePath",
                         "@label.imageHeight",
                         "@label.imageWidth",
                         "@label.flags"};
        for (const auto& title : textFields)
            tags.append(QString("@widget." + title));
        connect(m_prompt, &QTextEdit::textChanged, this, [this, tags]() {
            const auto original = m_prompt->textCursor();
            auto cursor = original;
            const QString text = m_prompt->toPlainText();
            m_prompt->blockSignals(true);
            cursor.select(QTextCursor::Document);
            cursor.setCharFormat(QTextCharFormat());
            QTextCharFormat highlight;
            highlight.setBackground(QColor("#E3F2FD"));
            highlight.setForeground(QColor("#1976D2"));
            for (const auto& tag : tags) {
                int start = 0;
                while ((start = text.indexOf(tag, start)) >= 0) {
                    cursor.setPosition(start);
                    cursor.setPosition(start + tag.size(),
                                       QTextCursor::KeepAnchor);
                    cursor.setCharFormat(highlight);
                    start += tag.size();
                }
            }
            m_prompt->setTextCursor(original);
            m_prompt->blockSignals(false);
        });
        auto* buttons = new QHBoxLayout;
        buttons->setContentsMargins(0, 8, 0, 0);
        buttons->setSpacing(12);
        layout->addLayout(buttons);
        auto* templates = new QPushButton(chatTr("Templates"), this);
        templates->setObjectName("VQATemplates");
        buttons->addWidget(templates);
        buttons->addStretch();
        auto* cancel = new QPushButton(chatTr("Cancel"), this);
        cancel->setObjectName("VQAPromptCancel");
        auto* generate = new QPushButton(chatTr("Generate"), this);
        generate->setObjectName("VQAPromptGenerate");
        buttons->addWidget(cancel);
        buttons->addWidget(generate);
        for (auto* button : {templates, cancel, generate}) {
            button->setCursor(Qt::PointingHandCursor);
            button->setStyleSheet(vqaButtonStyle(this, button == generate));
        }
        connect(cancel, &QPushButton::clicked, this, &QDialog::reject);
        connect(generate, &QPushButton::clicked, this, &QDialog::accept);
        connect(templates, &QPushButton::clicked, this, [this]() {
            VqaTemplateDialog dialog(this);
            if (dialog.exec() == QDialog::Accepted &&
                !dialog.selected().isEmpty())
                m_prompt->setPlainText(dialog.selected());
        });
        adjustSize();
        move(parent->mapToGlobal(parent->rect().center()) - rect().center());
    }
    QString prompt() const { return m_prompt->toPlainText().trimmed(); }

private:
    QTextEdit* m_prompt;
};

class VqaDialog final : public QDialog {
public:
    using Write = std::function<bool(const Source&, const QJsonObject&)>;
    using Read = std::function<AnnotationDocument(const QString&)>;
    using Transform = std::function<QJsonObject(QJsonObject)>;
    using Migrate = std::function<bool(const Transform&, QString*)>;
    VqaDialog(QWidget* parent,
              std::function<Source()> source,
              Write write,
              std::function<void(int)> navigate,
              std::function<QStringList()> files,
              Read read,
              std::function<void()> openFolder,
              std::function<QString(const QString&)> labelPath,
              Migrate migrate)
        : QDialog(parent),
          m_source(std::move(source)),
          m_write(std::move(write)),
          m_files(std::move(files)),
          m_read(std::move(read)),
          m_labelPath(std::move(labelPath)),
          m_migrate(std::move(migrate)),
          m_navigate(std::move(navigate)) {
        setObjectName(QStringLiteral("AnnotationVQA"));
        setPalette(parent->palette());
        setStyleSheet(parent->styleSheet());
        setWindowTitle(chatTr("VQA"));
        setAttribute(Qt::WA_DeleteOnClose);
        setWindowFlags(Qt::Window | Qt::WindowMinMaxButtonsHint |
                       Qt::WindowCloseButtonHint);
        resize(1200, 700);
        setMinimumSize(600, 600);
        auto* root = new QHBoxLayout(this);
        root->setContentsMargins(10, 10, 10, 10);
        root->setSpacing(20);
        auto* splitter = new QSplitter(Qt::Horizontal, this);
        m_splitter = splitter;
        splitter->setObjectName("VQAMainSplitter");
        splitter->setHandleWidth(3);
        root->addWidget(splitter);
        auto* left = new QWidget(splitter);
        left->setObjectName("VQAImagePanel");
        auto* leftLayout = new QVBoxLayout(left);
        leftLayout->setContentsMargins(0, 0, 10, 0);
        leftLayout->setSpacing(10);
        auto* folder = new QPushButton(chatTr("Open Folder"), left);
        folder->setObjectName("VQAOpenFolder");
        leftLayout->addWidget(folder);
        connect(folder, &QPushButton::clicked, this,
                [openFolder]() { openFolder(); });
        m_filename = new QLabel(left);
        m_filename->setObjectName("VQAFilename");
        m_filename->setAlignment(Qt::AlignCenter);
        m_filename->setFixedHeight(32);
        auto* header = new QHBoxLayout;
        header->setContentsMargins(0, 0, 0, 0);
        header->setSpacing(4);
        header->addWidget(m_filename, 1);
        const auto refresh = [this]() {
            if (m_current.path.isEmpty()) {
                m_service->status->setText(chatTr("No images loaded in main window!"));
                return;
            }
            m_navigate(0);
        };
        auto* refreshButton = new QPushButton(left);
        refreshButton->setObjectName("VQARefresh");
        refreshButton->setIcon(QIcon(":/CC/plugin/qAnyLabeling/images/refresh.svg"));
        refreshButton->setIconSize(QSize(16, 16));
        refreshButton->setFixedSize(28, 28);
        refreshButton->setToolTip(chatTr("Refresh Data"));
        connect(refreshButton, &QPushButton::clicked, this, refresh);
        header->addWidget(refreshButton);
        auto* toggle = new QPushButton(left);
        toggle->setObjectName("VQAToggleSidebar");
        toggle->setIcon(QIcon(":/CC/plugin/qAnyLabeling/images/sidebar.svg"));
        toggle->setIconSize(QSize(16, 16));
        toggle->setFixedSize(28, 28);
        toggle->setToolTip(chatTr("Toggle Sidebar"));
        header->addWidget(toggle);
        leftLayout->addLayout(header);
        auto* imageContainer = new QWidget(left);
        imageContainer->setObjectName("VQAImageContainer");
        auto* imageLayout = new QVBoxLayout(imageContainer);
        imageLayout->setContentsMargins(8, 8, 8, 8);
        m_preview = new QLabel(imageContainer);
        m_preview->setObjectName("VQAImagePreview");
        m_preview->setAlignment(Qt::AlignCenter);
        m_preview->setMinimumSize(200, 200);
        imageLayout->addWidget(m_preview, 1);
        leftLayout->addWidget(imageContainer, 1);
        auto* navigation = new QHBoxLayout;
        navigation->setContentsMargins(10, 0, 10, 0);
        navigation->setSpacing(8);
        auto* previous = new QPushButton(chatTr("Previous"), left);
        auto* next = new QPushButton(chatTr("Next"), left);
        m_previous = previous;
        m_next = next;
        previous->setObjectName("VQAPrevious");
        next->setObjectName("VQANext");
        for (auto* button : {previous, next}) {
            button->setText({});
            button->setFixedSize(32, 32);
            button->setIconSize(QSize(32, 32));
        }
        previous->setIcon(QIcon(":/CC/plugin/qAnyLabeling/images/arrow-left.svg"));
        previous->setToolTip(chatTr("Previous"));
        next->setIcon(QIcon(":/CC/plugin/qAnyLabeling/images/arrow-right.svg"));
        next->setToolTip(chatTr("Next"));
        navigation->addWidget(previous);
        navigation->addStretch();
        auto* pageContainer = new QWidget(this);
        pageContainer->setFixedWidth(82);
        auto* pageLayout = new QHBoxLayout(pageContainer);
        pageLayout->setContentsMargins(0, 0, 0, 0);
        pageLayout->setSpacing(2);
        m_page = new QLineEdit(pageContainer);
        m_page->setObjectName("VQAPageInput");
        m_page->setFixedSize(68, 32);
        m_page->setAlignment(Qt::AlignCenter);
        m_page->installEventFilter(this);
        m_pageValidator = new QIntValidator(1, 1, m_page);
        m_page->setValidator(m_pageValidator);
        pageLayout->addWidget(m_page);
        navigation->addWidget(pageContainer);
        navigation->addStretch();
        navigation->addWidget(next);
        connect(previous, &QPushButton::clicked, this,
                [this]() { m_navigate(-1); });
        connect(next, &QPushButton::clicked, this,
                [this]() { m_navigate(1); });
        connect(m_page, &QLineEdit::textChanged, this, [this](const QString& text) {
            const int maximum =
                    std::max(1, static_cast<int>(m_files().size()));
            if (text.toInt() > maximum) m_page->setText(QString::number(maximum));
        });
        connect(m_page, &QLineEdit::editingFinished, this, [this]() {
            if (m_page->text().isEmpty()) updateNavigation();
            else if (m_page->text().toInt() < 1) m_page->setText("1");
        });
        connect(new QShortcut(QKeySequence(Qt::Key_A), this),
                &QShortcut::activated, previous, &QPushButton::click);
        connect(new QShortcut(QKeySequence(Qt::Key_D), this),
                &QShortcut::activated, next, &QPushButton::click);
        auto* right = new QWidget(splitter);
        right->setObjectName("VQAComponentsPanel");
        auto* layout = new QVBoxLayout(right);
        layout->setContentsMargins(0, 0, 0, 0);
        layout->setSpacing(10);
        m_service = new ServicePanel(right);
        layout->addWidget(m_service);
        auto* controls = new QHBoxLayout;
        layout->addLayout(controls);
        m_refreshRight = new QPushButton(right);
        m_refreshRight->setObjectName("VQARefreshRight");
        m_refreshRight->setIcon(refreshButton->icon());
        m_refreshRight->setIconSize(refreshButton->iconSize());
        m_refreshRight->setFixedSize(refreshButton->size());
        m_refreshRight->setToolTip(refreshButton->toolTip());
        m_refreshRight->hide();
        connect(m_refreshRight, &QPushButton::clicked, this, refresh);
        controls->addWidget(m_refreshRight);
        m_toggleRight = new QPushButton(right);
        m_toggleRight->setObjectName("VQAToggleSidebarRight");
        m_toggleRight->setIcon(toggle->icon());
        m_toggleRight->setIconSize(QSize(16, 16));
        m_toggleRight->setFixedSize(28, 28);
        m_toggleRight->setToolTip(chatTr("Toggle Sidebar"));
        m_toggleRight->hide();
        controls->addWidget(m_toggleRight);
        const auto toggleSidebar = [this]() {
            const bool expand = m_splitter->sizes().first() == 0;
            const int width = m_splitter->width();
            m_splitter->setSizes(expand ? QList<int>{width / 2, width - width / 2}
                                        : QList<int>{0, width});
            m_refreshRight->setVisible(!expand);
            m_toggleRight->setVisible(!expand);
            updatePreview();
        };
        connect(toggle, &QPushButton::clicked, this, toggleSidebar);
        connect(m_toggleRight, &QPushButton::clicked, this, toggleSidebar);
        auto* add = new QPushButton(chatTr("Add Component"), right);
        add->setObjectName("VQAAddComponent");
        controls->addWidget(add);
        auto* exportButton = new QPushButton(chatTr("Export"), right);
        exportButton->setObjectName("VQAExport");
        controls->addWidget(exportButton);
        auto* clear = new QPushButton(chatTr("Clear All"), right);
        clear->setObjectName("VQAClearCurrent");
        controls->addWidget(clear);
        connect(clear, &QPushButton::clicked, this, [this]() {
            const auto source = m_source();
            if (QMessageBox::question(
                        this, chatTr("Confirm Clear"),
                        chatTr("Are you sure you want to clear all current annotations?"),
                        QMessageBox::Yes | QMessageBox::No,
                        QMessageBox::No) != QMessageBox::Yes)
                return;
            const auto current = m_source();
            if (source.path != current.path || source.generation != current.generation)
                return;
            cancelRequest();
            for (auto& component : m_components)
                setValue(component, component.type == "QCheckBox"
                                            ? QJsonValue(QJsonArray())
                                    : component.type == "QLineEdit"
                                            ? QJsonValue(QString())
                                            : QJsonValue(QJsonValue::Null));
            this->write();
        });
        auto* config = new QHBoxLayout;
        layout->addLayout(config);
        auto* importConfig =
                new QPushButton(chatTr("Import Components"), right);
        importConfig->setObjectName("VQAImportComponents");
        auto* exportConfig =
                new QPushButton(chatTr("Export Components"), right);
        exportConfig->setObjectName("VQAExportComponents");
        config->addWidget(importConfig);
        config->addWidget(exportConfig);
        connect(importConfig, &QPushButton::clicked, this, [this]() {
            const auto path = QFileDialog::getOpenFileName(
                    this, chatTr("Import Components"), {}, "JSON (*.json)");
            if (path.isEmpty()) return;
            QFile file(path);
            if (!file.open(QIODevice::ReadOnly)) {
                m_service->status->setText(file.errorString());
                return;
            }
            QJsonParseError error;
            const auto json = QJsonDocument::fromJson(file.readAll(), &error);
            if (!validVqaComponents(json)) {
                m_service->status->setText(
                        chatTr("Invalid component configuration."));
                return;
            }
            // Import changes the schema only. Existing annotation keys survive.
            m_config = json.object().value("components").toArray();
            m_configWritable =
                    true;  // Explicit import replaces invalid config.
            saveConfig();
            rebuild();
            loadValues();
        });
        connect(exportConfig, &QPushButton::clicked, this, [this]() {
            const auto path = QFileDialog::getSaveFileName(
                    this, chatTr("Export Components"), "components.json",
                    "JSON (*.json)");
            if (path.isEmpty()) return;
            QSaveFile file(path);
            const auto bytes =
                    QJsonDocument(QJsonObject{{"components", m_config}})
                            .toJson();
            if (!file.open(QIODevice::WriteOnly) ||
                file.write(bytes) != bytes.size() || !file.commit())
                m_service->status->setText(file.errorString());
        });
        auto* scroll = new QScrollArea(right);
        scroll->setWidgetResizable(true);
        auto* components = new QWidget(scroll);
        m_layout = new QVBoxLayout(components);
        m_layout->setAlignment(Qt::AlignTop);
        scroll->setWidget(components);
        layout->addWidget(scroll, 1);
        layout->addLayout(navigation);
        auto* stop = new QPushButton(chatTr("Stop"), right);
        stop->setObjectName("VQAStop");
        layout->addWidget(stop);
        splitter->setSizes({600, 600});
        QSettings saved;
        m_config = QJsonDocument::fromJson(
                           saved.value("qAnyLabeling/vqa/components")
                                   .toByteArray())
                           .object()
                           .value("components")
                           .toArray();
        QFile componentFile(vqaConfigPath("components.json"));
        if (componentFile.exists()) {
            const auto json =
                    componentFile.open(QIODevice::ReadOnly)
                            ? QJsonDocument::fromJson(componentFile.readAll())
                            : QJsonDocument();
            m_configWritable = validVqaComponents(json);
            if (m_configWritable) {
                m_config = json.object().value("components").toArray();
                saved.setValue("qAnyLabeling/vqa/components",
                               json.toJson(QJsonDocument::Compact));
            } else
                m_service->status->setText(chatTr(
                        "Invalid component configuration; file preserved. "
                        "Import a valid configuration to replace it."));
        } else {
            saveConfig();
        }
        add->setEnabled(m_configWritable);
        m_addComponent = add;
        rebuild();
        connect(add, &QPushButton::clicked, this,
                [this]() { editComponent(-1); });
        connect(stop, &QPushButton::clicked, this, [this]() {
            cancelRequest();
            m_service->status->setText(chatTr("Cancelled"));
        });
        connect(exportButton, &QPushButton::clicked, this,
                [this]() { exportLabels(); });
        connect(m_service->client, &ChatServiceClient::finished, this,
                [this](const QString& answer, const QString& error) {
                    finishLoading();
                    const auto current = m_source();
                    if (current.path != m_request.path ||
                        current.generation != m_request.generation)
                        return;
                    if (!error.isEmpty()) {
                        m_service->status->setText(error);
                        return;
                    }
                    const QString title = m_pendingTitle;
                    m_pendingTitle.clear();
                    // The source previews generated text and applies it only
                    // after explicit acceptance. A model answer is not an
                    // annotation yet.
                    auto* preview = new QDialog(this);
                    preview->setObjectName("VQAAnswerPreview");
                    preview->setWindowTitle(chatTr("AI Generated Result"));
                    preview->setAttribute(Qt::WA_DeleteOnClose);
                    preview->setWindowModality(Qt::WindowModal);
                    preview->setWindowFlag(Qt::WindowStaysOnTopHint);
                    preview->resize(500, 400);
                    auto* layout = new QVBoxLayout(preview);
                    layout->setSpacing(10);
                    auto* text = new QTextEdit(preview);
                    text->setPlainText(answer);
                    text->setReadOnly(true);
                    layout->addWidget(text);
                    auto* buttons = new QDialogButtonBox(
                            QDialogButtonBox::Apply | QDialogButtonBox::Cancel,
                            preview);
                    layout->addWidget(buttons);
                    buttons->button(QDialogButtonBox::Apply)->setDefault(true);
                    buttons->button(QDialogButtonBox::Apply)
                            ->setStyleSheet(vqaButtonStyle(preview, true));
                    buttons->button(QDialogButtonBox::Cancel)
                            ->setStyleSheet(vqaButtonStyle(preview, false));
                    const Source request = m_request;
                    connect(buttons->button(QDialogButtonBox::Apply),
                            &QPushButton::clicked, preview,
                            [this, preview, request, title, answer]() {
                                const auto source = m_source();
                                if (source.path == request.path &&
                                    source.generation == request.generation)
                                    for (auto& component : m_components)
                                        if (component.title == title) {
                                            setValue(component, answer);
                                            this->write();
                                        }
                                preview->accept();
                            });
                    connect(buttons, &QDialogButtonBox::rejected, preview,
                            &QDialog::reject);
                    m_service->status->clear();
                    preview->open();
                });
        m_ready = true;
        applyStyles();
        refreshSource();
    }
    void sourceChanged() { refreshSource(); }
    void reject() override {
        cancelRequest();
        QDialog::reject();
    }

protected:
    void changeEvent(QEvent* event) override {
        QDialog::changeEvent(event);
        if (m_ready && !m_styling && !m_stylePending &&
            (event->type() == QEvent::PaletteChange ||
             event->type() == QEvent::StyleChange)) {
            m_stylePending = true;
            QTimer::singleShot(0, this, [this]() {
                m_stylePending = false;
                applyStyles();
            });
        }
    }
    bool eventFilter(QObject* object, QEvent* event) override {
        if (object == m_page && event->type() == QEvent::KeyPress) {
            const auto* key = static_cast<QKeyEvent*>(event);
            if (key->key() == Qt::Key_Return || key->key() == Qt::Key_Enter) {
                if (m_page->text().trimmed().isEmpty()) {
                    updateNavigation();
                } else {
                    const auto files = m_files();
                    const int index = files.indexOf(m_current.path);
                    const int page = m_page->text().toInt();
                    if (index >= 0 && page > 0 && page <= files.size())
                        m_navigate(page - 1 - index);
                }
                return true;
            }
        }
        return QDialog::eventFilter(object, event);
    }
    void resizeEvent(QResizeEvent* event) override {
        QDialog::resizeEvent(event);
        if (m_preview) updatePreview();
    }
    void closeEvent(QCloseEvent* event) override {
        cancelRequest();
        QDialog::closeEvent(event);
    }

private:
    void applyStyles() {
        if (!m_ready || m_styling) return;
        m_styling = true;
        const auto apply = [this](const char* name, const char* style) {
            if (auto* widget = findChild<QWidget*>(QString::fromLatin1(name))) {
                const auto qss = vqaStyle(style, this);
                if (widget->styleSheet() != qss) widget->setStyleSheet(qss);
            }
        };
        apply("VQAMainSplitter", "vqa_main_splitter");
        apply("VQAFilename", "vqa_filename_label");
        apply("VQAImageContainer", "vqa_image_container");
        apply("VQAImagePreview", "vqa_image_label");
        apply("VQAPageInput", "vqa_page_input");
        for (const char* name : {"VQARefresh", "VQARefreshRight",
                                 "VQAToggleSidebar", "VQAToggleSidebarRight"})
            apply(name, "vqa_panel_button");
        for (const char* name : {"VQAPrevious", "VQANext"})
            apply(name, "vqa_button");
        apply("VQAAddComponent", "vqa_dialog_button_primary");
        apply("VQAExport", "vqa_dialog_button_success");
        apply("VQAClearCurrent", "vqa_dialog_button_secondary");
        m_styling = false;
    }
    void finishLoading() {
        if (!m_loading) return;
        auto* dialog = m_loading.data();
        m_loading.clear();
        dialog->accept();
    }
    void cancelRequest() {
        m_service->client->cancel();
        m_pendingTitle.clear();
        finishLoading();
    }
    void exportLabels() {
        auto files = m_files();
        if (files.isEmpty() && !m_current.path.isEmpty())
            files.append(m_current.path);
        if (files.isEmpty()) {
            m_service->status->setText(chatTr("No images loaded!"));
            return;
        }
        // Source: vqa/dialogs.py ExportLabelsDialog and
        // vqa_dialog.export_labels, X-AnyLabeling-GGML 745f6c355f94
        // (GPL-3.0-only; UPSTREAM-LICENSE).
        QDialog selection(this);
        selection.setObjectName("VQAExportFields");
        selection.setWindowTitle(chatTr("Export Labels"));
        selection.resize(520, 500);
        auto* layout = new QVBoxLayout(&selection);
        auto* all = new QCheckBox(chatTr("Select All:"), &selection);
        all->setObjectName("VQAExportAll");
        layout->addWidget(all);
        QTableWidget fields(4 + m_components.size(), 4, &selection);
        fields.setObjectName("VQAExportTable");
        fields.setHorizontalHeaderLabels(
                {chatTr("Type"), chatTr("Original Key"), chatTr("Export Key"),
                 chatTr("Select")});
        fields.verticalHeader()->hide();
        fields.setAlternatingRowColors(true);
        fields.setSelectionMode(QAbstractItemView::NoSelection);
        fields.horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
        layout->addWidget(&fields);
        QStringList keys{"image", "width", "height", "shapes"};
        for (const auto& c : m_components) keys.append(c.title);
        auto* buttons = new QDialogButtonBox(
                QDialogButtonBox::Save | QDialogButtonBox::Cancel, &selection);
        buttons->button(QDialogButtonBox::Save)->setText(chatTr("Export"));
        layout->addWidget(buttons);
        auto update = [&]() {
            bool selected = false;
            for (int row = 0; row < fields.rowCount(); ++row)
                selected |= static_cast<QCheckBox*>(fields.cellWidget(row, 3))
                                    ->isChecked();
            buttons->button(QDialogButtonBox::Save)->setEnabled(selected);
        };
        for (int row = 0; row < fields.rowCount(); ++row) {
            for (int column = 0; column < 2; ++column) {
                auto* item = new QTableWidgetItem(
                        column    ? keys[row]
                        : row < 4 ? chatTr("Basic")
                                  : m_components[row - 4].type);
                item->setFlags(item->flags() & ~Qt::ItemIsEditable);
                fields.setItem(row, column, item);
            }
            fields.setCellWidget(row, 2, new QLineEdit(keys[row], &fields));
            auto* check = new QCheckBox(&fields);
            check->setChecked(keys[row] != "shapes");
            fields.setCellWidget(row, 3, check);
            connect(check, &QCheckBox::toggled, &selection, update);
        }
        connect(all, &QCheckBox::toggled, &selection, [&](bool checked) {
            for (int row = 0; row < fields.rowCount(); ++row)
                static_cast<QCheckBox*>(fields.cellWidget(row, 3))
                        ->setChecked(checked);
        });
        connect(buttons, &QDialogButtonBox::accepted, &selection,
                &QDialog::accept);
        connect(buttons, &QDialogButtonBox::rejected, &selection,
                &QDialog::reject);
        if (selection.exec() != QDialog::Accepted) return;
        QVector<QPair<QString, QString>> mapping;
        for (int row = 0; row < fields.rowCount(); ++row) {
            const auto key = static_cast<QLineEdit*>(fields.cellWidget(row, 2))
                                     ->text()
                                     .trimmed();
            if (static_cast<QCheckBox*>(fields.cellWidget(row, 3))
                        ->isChecked() &&
                !key.isEmpty())
                mapping.append({keys[row], key});
        }
        if (mapping.isEmpty()) {
            m_service->status->setText(
                    chatTr("No fields selected for export!"));
            return;
        }
        const auto path = QFileDialog::getSaveFileName(
                this, chatTr("Export Labels"), "vqa_labels.jsonl",
                "JSONL (*.jsonl)");
        if (path.isEmpty()) return;
        QSaveFile output(path);
        if (!output.open(QIODevice::WriteOnly)) {
            m_service->status->setText(output.errorString());
            return;
        }
        // O(n) streaming: hold one decoded image/sidecar, never the full
        // folder.
        for (const auto& file : files) {
            auto document = m_read(file);
            if (!document.error.isEmpty()) {
                m_service->status->setText(file + ": " + document.error);
                return;
            }
            QJsonObject row{{"image", QFileInfo(file).fileName()},
                            {"width", document.image.width()},
                            {"height", document.image.height()}};
            const auto values = document.json.value("vqaData").toObject();
            for (auto it = values.begin(); it != values.end(); ++it)
                row.insert(it.key(), it.value());
            row.insert("shapes", document.json.value("shapes").toArray());
            for (const auto& c : m_components)
                if (!row.contains(c.title))
                    row.insert(c.title, c.type == "QLineEdit"
                                                ? QJsonValue(QString())
                                        : c.type == "QCheckBox"
                                                ? QJsonValue(QJsonArray())
                                                : QJsonValue(QJsonValue::Null));
            QJsonObject selected;
            for (const auto& field : mapping)
                if (row.contains(field.first))
                    selected.insert(field.second, row.value(field.first));
            // QT_USE_QSTRINGBUILDER makes auto retain references to the
            // temporary JSON byte array. Materialize before appending newline.
            QByteArray line =
                    QJsonDocument(selected).toJson(QJsonDocument::Compact);
            line.append('\n');
            if (output.write(line) != line.size()) {
                m_service->status->setText(output.errorString());
                return;
            }
        }
        m_service->status->setText(
                output.commit() ? chatTr("Labels exported to %1.").arg(path)
                                : output.errorString());
    }
    struct Component {
        QString title, type;
        QStringList options;
        QWidget* widget;
        QVector<QAbstractButton*> buttons;
    };
    QJsonValue value(const Component& c) const {
        if (c.type == "QLineEdit")
            return static_cast<QPlainTextEdit*>(c.widget)
                    ->toPlainText()
                    .trimmed();
        if (c.type == "QComboBox") {
            const auto* combo = static_cast<QComboBox*>(c.widget);
            return combo->currentIndex() <= 0
                           ? QJsonValue(QJsonValue::Null)
                           : QJsonValue(combo->currentText());
        }
        QJsonArray selected;
        for (auto* button : c.buttons)
            if (button->isChecked()) selected.append(button->text());
        return c.type == "QCheckBox" ? QJsonValue(selected)
               : selected.isEmpty()  ? QJsonValue(QJsonValue::Null)
                                     : selected.first();
    }
    void setValue(Component& c, const QJsonValue& v) {
        m_updating = true;
        if (c.type == "QLineEdit")
            static_cast<QPlainTextEdit*>(c.widget)->setPlainText(v.toString());
        else if (c.type == "QComboBox") {
            auto* combo = static_cast<QComboBox*>(c.widget);
            combo->setCurrentIndex(std::max(0, combo->findText(v.toString())));
        } else {
            // Temporarily lift radio exclusivity so null can clear every
            // option.
            for (auto* button : c.buttons) button->setAutoExclusive(false);
            for (auto* button : c.buttons)
                button->setChecked(
                        c.type == "QCheckBox"
                                ? v.toArray().contains(button->text())
                                : v.toString() == button->text());
            if (c.type == "QRadioButton")
                for (auto* button : c.buttons) button->setAutoExclusive(true);
        }
        m_updating = false;
    }
    QJsonObject data() const {
        auto result = m_current.document.value("vqaData").toObject();
        for (const auto& c : m_components) result.insert(c.title, value(c));
        return result;
    }
    void write() {
        if (m_updating || m_current.path.isEmpty()) return;
        const auto values = data();
        if (m_write(m_current, values))
            m_current.document.insert("vqaData", values);
    }
    void refreshSource() {
        const auto source = m_source();
        if (source.path == m_current.path &&
            source.generation == m_current.generation)
            return;
        cancelRequest();
        m_current = source;
        m_filename->setText(source.path.isEmpty()
                                    ? chatTr("No image loaded")
                                    : QFileInfo(source.path).fileName());
        updatePreview();
        updateNavigation();
        loadValues();
    }
    void updatePreview() {
        m_preview->setPixmap(QPixmap::fromImage(m_current.image)
                                     .scaled(m_preview->size(), Qt::KeepAspectRatio,
                                             Qt::SmoothTransformation));
    }
    void updateNavigation() {
        const auto files = m_files();
        const int index = files.indexOf(m_current.path);
        m_previous->setEnabled(index > 0);
        m_next->setEnabled(index >= 0 && index + 1 < files.size());
        m_pageValidator->setRange(1, std::max(1, static_cast<int>(files.size())));
        m_page->setText(QString::number(index >= 0 ? index + 1 : 1));
    }
    void loadValues() {
        const auto values = m_current.document.value("vqaData").toObject();
        for (auto& c : m_components) {
            QJsonValue fallback =
                    c.type == "QCheckBox" ? QJsonValue(QJsonArray())
                    : c.type == "QComboBox"
                            ? QJsonValue(QJsonValue::Null)
                            : QJsonValue(c.type == "QRadioButton" &&
                                                         !c.options.isEmpty()
                                                 ? c.options.first()
                                                 : QString());
            setValue(c, values.contains(c.title) ? values.value(c.title)
                                                 : fallback);
        }
    }
    void rebuild() {
        cancelRequest();
        m_addComponent->setEnabled(m_configWritable);
        m_components.clear();
        while (auto* item = m_layout->takeAt(0)) {
            delete item->widget();
            delete item;
        }
        for (int i = 0; i < m_config.size(); ++i) {
            const auto config = m_config[i].toObject();
            const auto type = config.value("type").toString();
            if (!QStringList{"QLineEdit", "QRadioButton", "QComboBox",
                             "QCheckBox"}
                         .contains(type))
                continue;
            Component c;
            c.title = config.value("title").toString();
            c.type = type;
            if (c.title.isEmpty()) continue;
            for (const auto& option : config.value("options").toArray())
                c.options.append(option.toString());
            auto* group = new QGroupBox(c.title, this);
            auto* layout = new QVBoxLayout(group);
            if (type == "QLineEdit") {
                auto* edit = new QPlainTextEdit(group);
                edit->setMaximumHeight(110);
                c.widget = edit;
                connect(edit, &QPlainTextEdit::textChanged, this,
                        [this]() { write(); });
            } else if (type == "QComboBox") {
                auto* combo = new QComboBox(group);
                combo->addItem("-- Select --");
                combo->addItems(c.options);
                c.widget = combo;
                connect(combo, &QComboBox::currentTextChanged, this,
                        [this]() { write(); });
            } else {
                c.widget = new QWidget(group);
                auto* buttons = new QVBoxLayout(c.widget);
                for (const auto& option : c.options) {
                    QAbstractButton* button =
                            type == "QCheckBox"
                                    ? static_cast<QAbstractButton*>(
                                              new QCheckBox(option, c.widget))
                                    : new QRadioButton(option, c.widget);
                    buttons->addWidget(button);
                    c.buttons.append(button);
                    connect(button, &QAbstractButton::toggled, this,
                            [this]() { write(); });
                }
            }
            c.widget->setObjectName("VQAComponent");
            c.widget->setProperty("title", c.title);
            layout->addWidget(c.widget);
            auto* actions = new QHBoxLayout;
            layout->addLayout(actions);
            auto* edit = new QPushButton(chatTr("Edit"), group);
            edit->setObjectName("VQAEditComponent");
            edit->setProperty("title", c.title);
            auto* remove = new QPushButton(chatTr("Delete"), group);
            remove->setObjectName("VQADeleteComponent");
            remove->setProperty("title", c.title);
            edit->setEnabled(m_configWritable);
            remove->setEnabled(m_configWritable);
            auto* up = new QPushButton(chatTr("Up"), group);
            up->setObjectName("VQAComponentUp");
            up->setProperty("title", c.title);
            up->setEnabled(m_configWritable && i > 0);
            auto* down = new QPushButton(chatTr("Down"), group);
            down->setObjectName("VQAComponentDown");
            down->setProperty("title", c.title);
            down->setEnabled(m_configWritable && i + 1 < m_config.size());
            auto* ai = new QPushButton(chatTr("AI Assistant"), group);
            ai->setObjectName("VQAAssistant");
            ai->setProperty("title", c.title);
            actions->addWidget(edit);
            actions->addWidget(remove);
            actions->addWidget(up);
            actions->addWidget(down);
            actions->addWidget(ai);
            ai->setVisible(type == "QLineEdit");
            connect(edit, &QPushButton::clicked, this,
                    [this, i]() { editComponent(i); });
            connect(remove, &QPushButton::clicked, this,
                    [this, i, title = c.title]() {
                        if (QMessageBox::question(
                                    this, chatTr("Confirm Delete"),
                                    chatTr("Deleting component '%1' will "
                                           "remove all related annotation data "
                                           "from the current task.\nDo you "
                                           "want to continue?")
                                            .arg(title),
                                    QMessageBox::Yes | QMessageBox::No,
                                    QMessageBox::No) != QMessageBox::Yes)
                            return;
                        QString error;
                        if (!m_migrate(
                                    [title](QJsonObject values) {
                                        values.remove(title);
                                        return values;
                                    },
                                    &error)) {
                            m_service->status->setText(error);
                            return;
                        }
                        m_current = m_source();
                        m_config.removeAt(i);
                        saveConfig();
                        QTimer::singleShot(0, this, [this]() {
                            rebuild();
                            loadValues();
                        });
                    });
            const auto move = [this, i](int step) {
                const auto component = m_config.takeAt(i);
                m_config.insert(i + step, component);
                saveConfig();
                QTimer::singleShot(0, this, [this]() {
                    rebuild();
                    loadValues();
                });
            };
            connect(up, &QPushButton::clicked, this, [move]() { move(-1); });
            connect(down, &QPushButton::clicked, this, [move]() { move(1); });
            connect(ai, &QPushButton::clicked, this,
                    [this, title = c.title]() { ask(title); });
            m_layout->addWidget(group);
            m_components.append(c);
        }
    }
    void saveConfig() {
        if (!m_configWritable) return;
        const QJsonDocument json(QJsonObject{{"components", m_config}});
        QString error;
        if (!writeVqaJson(vqaConfigPath("components.json"), json.toJson(),
                          &error)) {
            m_service->status->setText(error);
            return;
        }
        QSettings().setValue("qAnyLabeling/vqa/components",
                             json.toJson(QJsonDocument::Compact));
    }
    void editComponent(int index) {
        QDialog dialog(this);
        dialog.setObjectName("VQAComponentEditor");
        dialog.setWindowTitle(chatTr("Component"));
        auto* form = new QFormLayout(&dialog);
        QLineEdit title(&dialog);
        title.setObjectName("VQAComponentTitle");
        QComboBox type(&dialog);
        type.setObjectName("VQAComponentType");
        type.addItems({"QLineEdit", "QRadioButton", "QComboBox", "QCheckBox"});
        QPlainTextEdit options(&dialog);
        options.setObjectName("VQAComponentOptions");
        if (index >= 0) {
            const auto config = m_config[index].toObject();
            title.setText(config.value("title").toString());
            type.setCurrentText(config.value("type").toString());
            type.setEnabled(false);  // Source edits keep the component type.
            QStringList names;
            for (const auto& value : config.value("options").toArray())
                names.append(value.toString());
            options.setPlainText(names.join('\n'));
        }
        options.setEnabled(type.currentText() != "QLineEdit");
        connect(&type, &QComboBox::currentTextChanged, &dialog,
                [&](const QString& value) {
                    options.setEnabled(value != "QLineEdit");
                });
        form->addRow(chatTr("Title"), &title);
        form->addRow(chatTr("Type"), &type);
        form->addRow(chatTr("Options (one per line)"), &options);
        QDialogButtonBox buttons(
                QDialogButtonBox::Ok | QDialogButtonBox::Cancel, &dialog);
        form->addRow(&buttons);
        connect(&buttons, &QDialogButtonBox::accepted, &dialog,
                &QDialog::accept);
        connect(&buttons, &QDialogButtonBox::rejected, &dialog,
                &QDialog::reject);
        if (dialog.exec() != QDialog::Accepted) return;
        const auto name = title.text().trimmed();
        if (name.isEmpty()) return;
        for (int i = 0; i < m_config.size(); ++i)
            if (i != index &&
                m_config[i].toObject().value("title").toString() == name) {
                m_service->status->setText(
                        chatTr("Component titles must be unique."));
                return;
            }
        QStringList names;
        for (const auto& line : options.toPlainText().split('\n'))
            if (!line.trimmed().isEmpty() && !names.contains(line.trimmed()))
                names.append(line.trimmed());
        if (type.currentText() != "QLineEdit" && names.isEmpty()) return;
        const QJsonObject config{
                {"title", name},
                {"type", type.currentText()},
                {"options", QJsonArray::fromStringList(names)}};
        if (index >= 0) {
            const auto old = m_config[index].toObject();
            const auto oldName = old.value("title").toString();
            const auto oldType = old.value("type").toString();
            const auto oldOptions = old.value("options").toArray();
            QMap<QString, QString> mapping;
            QSet<QString> deleted;
            for (int i = 0; i < oldOptions.size(); ++i) {
                const auto option = oldOptions[i].toString();
                if (!names.contains(option)) {
                    deleted.insert(option);
                    if (oldOptions.size() == names.size())
                        mapping.insert(option, names[i]);
                }
            }
            if (oldType != "QLineEdit" && !deleted.isEmpty() &&
                QMessageBox::question(
                        this, chatTr("Confirm Option Delete"),
                        chatTr("Removing options %1 will reset related "
                               "annotation data to default values.\nDo you "
                               "want to continue?")
                                .arg(QStringList(deleted.values()).join(", ")),
                        QMessageBox::Yes | QMessageBox::No,
                        QMessageBox::No) != QMessageBox::Yes)
                return;
            const auto newType = type.currentText();
            QString error;
            if (!m_migrate(
                        [=](QJsonObject values) {
                            if (!values.contains(oldName)) return values;
                            auto value = values.take(oldName);
                            const auto mapped = [&](const QJsonValue& option)
                                    -> QJsonValue {
                                return mapping.contains(option.toString())
                                               ? QJsonValue(mapping.value(
                                                         option.toString()))
                                               : option;
                            };
                            if (oldType != "QLineEdit") {
                                if (value.isArray()) {
                                    QJsonArray options;
                                    for (const auto& option : value.toArray())
                                        options.append(mapped(option));
                                    value = options;
                                } else
                                    value = mapped(value);
                                bool reset = value.isString() &&
                                             deleted.contains(value.toString());
                                for (const auto& option : value.toArray())
                                    reset |=
                                            deleted.contains(option.toString());
                                if (reset || oldType != newType)
                                    value = newType == "QRadioButton" &&
                                                            !names.isEmpty()
                                                    ? QJsonValue(names.first())
                                            : newType == "QCheckBox"
                                                    ? QJsonValue(QJsonArray())
                                            : newType == "QComboBox"
                                                    ? QJsonValue(
                                                              QJsonValue::Null)
                                                    : QJsonValue(QString());
                            }
                            values.insert(name, value);
                            return values;
                        },
                        &error)) {
                m_service->status->setText(error);
                return;
            }
            m_current = m_source();
        }
        if (index < 0)
            m_config.append(config);
        else
            m_config[index] = config;
        saveConfig();
        QTimer::singleShot(0, this, [this]() {
            rebuild();
            loadValues();
        });
    }
    void ask(const QString& title) {
        if (m_service->client->busy() || m_current.path.isEmpty()) return;
        QStringList fields;
        for (const auto& c : m_components)
            if (c.type == "QLineEdit") fields.append(c.title);
        const auto source = m_source();
        VqaAiPromptDialog dialog(this, fields);
        if (dialog.exec() != QDialog::Accepted) return;
        QString text = dialog.prompt();
        if (text.trimmed().isEmpty()) {
            m_service->status->setText(chatTr("Enter an AI prompt."));
            return;
        }
        m_request = m_source();
        if (m_request.path != source.path ||
            m_request.generation != source.generation)
            return;
        for (const auto& c : m_components) {
            const QString v = value(c).toString().trimmed();
            if (c.title == title && !v.isEmpty()) text.replace("@text", v);
            if (c.type == "QLineEdit") text.replace("@widget." + c.title, v);
        }
        QFile file(m_labelPath(m_request.path));
        if (file.exists()) {
            QJsonParseError parse;
            const auto bytes = file.open(QIODevice::ReadOnly) ? file.readAll()
                                                              : QByteArray();
            const auto labels = QJsonDocument::fromJson(bytes, &parse);
            if (parse.error != QJsonParseError::NoError || !labels.isObject()) {
                m_service->status->setText(chatTr("Invalid image annotation"));
                return;
            }
            const auto members =
                    AnnotationOrderedJson::members(bytes.trimmed());
            QVector<QPair<QString, QString>> refs;
            for (const QString& key : {"shapes", "imagePath", "imageHeight",
                                       "imageWidth", "flags"}) {
                QByteArray raw = key == "shapes"  ? "[]"
                                 : key == "flags" ? "{}"
                                                  : "\"\"";
                for (const auto& m : members)
                    if (m.first == key) raw = m.second;
                refs.append({"@label." + key, vqaPythonValue(raw)});
            }
            for (const auto& m : members)
                if (m.first == "vqaData" && m.second.startsWith('{'))
                    for (const auto& v :
                         AnnotationOrderedJson::members(m.second)) {
                        const QString ref = "@label." + v.first;
                        auto found = std::find_if(
                                refs.begin(), refs.end(),
                                [&](const auto& r) { return r.first == ref; });
                        if (found == refs.end())
                            refs.append({ref, vqaPythonValue(v.second)});
                        else
                            found->second = vqaPythonValue(v.second);
                    }
            for (const auto& ref : refs) text.replace(ref.first, ref.second);
        }
        const bool image = text.contains("@image");
        if (image) text.replace(QRegularExpression("@image\\s*"), "<image> ");
        QString error;
        const auto message = ChatServiceClient::userMessage(
                image ? text.trimmed() : text,
                image ? m_request.path : QString(), &error);
        if (!error.isEmpty()) {
            m_service->status->setText(error);
            return;
        }
        m_pendingTitle = title;
        m_service->status->setText(chatTr("Generating…"));
        m_loading = new VqaLoadingDialog(this);
        m_loading->setAttribute(Qt::WA_DeleteOnClose);
        connect(m_loading, &QDialog::rejected, this, [this]() {
            m_loading.clear();
            cancelRequest();
            m_service->status->setText(chatTr("Cancelled"));
        });
        m_loading->open();
        auto settings = m_service->settings();
        settings.timeoutMs = 120000;  // vqa/config.py REQUEST_TIMEOUT
        m_service->client->complete(
                settings, m_service->messages(QJsonArray{message}), false);
    }
    std::function<Source()> m_source;
    Write m_write;
    Source m_current, m_request;
    std::function<QStringList()> m_files;
    Read m_read;
    std::function<QString(const QString&)> m_labelPath;
    Migrate m_migrate;
    std::function<void(int)> m_navigate;
    ServicePanel* m_service;
    QLabel *m_filename, *m_preview = nullptr;
    QSplitter* m_splitter;
    QPushButton *m_previous, *m_next, *m_toggleRight, *m_refreshRight;
    QLineEdit* m_page;
    QIntValidator* m_pageValidator;
    QPushButton* m_addComponent = nullptr;
    QPointer<VqaLoadingDialog> m_loading;
    QVBoxLayout* m_layout;
    QJsonArray m_config;
    QVector<Component> m_components;
    QString m_pendingTitle;
    bool m_updating = false;
    bool m_configWritable = true;
    bool m_ready = false;
    bool m_styling = false;
    bool m_stylePending = false;
};
}  // namespace

QJsonArray AnnotationChatRequest::messages(const QString& imagePath,
                                           QString* error) const {
    QString processed = prompt;
    const bool hasImage = processed.contains(QStringLiteral("@image"));
    if (hasImage)
        processed.replace(QRegularExpression(QStringLiteral("@image\\s*")),
                          QStringLiteral("<image> "));
    const auto message = ChatServiceClient::userMessage(
            processed.trimmed(), hasImage ? imagePath : QString(), error);
    if (message.isEmpty()) return {};
    QJsonArray result;
    if (!systemPrompt.trimmed().isEmpty())
        result.append(QJsonObject{{"role", "system"},
                                  {"content", systemPrompt.trimmed()}});
    result.append(message);
    return result;
}

bool configureAnnotationChatRequest(QWidget* parent,
                                    const QString& title,
                                    const QString& prompt,
                                    AnnotationChatRequest* request) {
    QDialog dialog(parent);
    dialog.setObjectName(QStringLiteral("AnnotationClassifierPrompt"));
    dialog.setWindowTitle(title);
    dialog.resize(800, 650);
    auto* layout = new QHBoxLayout(&dialog);
    auto* service = new ServicePanel(&dialog);
    layout->addWidget(service);
    auto* content = new QVBoxLayout;
    layout->addLayout(content, 1);
    auto* text = new QPlainTextEdit(prompt, &dialog);
    text->setObjectName(QStringLiteral("ClassifierPrompt"));
    content->addWidget(text, 1);
    auto* buttons = new QDialogButtonBox(
            QDialogButtonBox::Ok | QDialogButtonBox::Cancel, &dialog);
    content->addWidget(buttons);
    QObject::connect(buttons, &QDialogButtonBox::rejected, &dialog,
                     &QDialog::reject);
    QObject::connect(buttons, &QDialogButtonBox::accepted, &dialog, [&]() {
        const auto config = service->settings();
        if (config.model.isEmpty()) {
            service->status->setText(chatTr("Choose a model."));
            return;
        }
        if (text->toPlainText().trimmed().isEmpty()) {
            service->status->setText(chatTr("Enter an AI prompt."));
            return;
        }
        request->settings = config;
        request->prompt = text->toPlainText();
        request->systemPrompt = service->system->text();
        dialog.accept();
    });
    return dialog.exec() == QDialog::Accepted;
}

void AnnotationWindow::openChatbot() {
    if (auto* existing =
                findChild<QDialog*>(QStringLiteral("AnnotationChatbot"))) {
        existing->show();
        existing->raise();
        existing->activateWindow();
        return;
    }
    auto* dialog = new ChatbotDialog(
            this,
            [this]() {
                return Source{m_imagePath, m_image, m_document,
                              m_sourceGeneration};
            },
            [this](const Source& source, const QJsonArray& history) {
                if (source.path != m_imagePath ||
                    source.generation != m_sourceGeneration ||
                    m_sidecarBlocked || m_imagePath.isEmpty())
                    return false;
                if (m_document.value("chat_history").toArray() != history) {
                    m_document.insert("chat_history", history);
                    setDirty(true);
                }
                return true;
            },
            [this]() {
                QStringList files;
                for (int i = 0; i < m_files->count(); ++i)
                    files.append(
                            m_files->item(i)->data(Qt::UserRole).toString());
                return files;
            },
            [this](int direction) {
                if (m_inferenceRunning || m_batchRunning) return;
                const int row = m_files->currentRow() + direction;
                if (row >= 0 && row < m_files->count())
                    openFile(m_files->item(row)->data(Qt::UserRole).toString());
            },
            [this](bool folder) {
                if (m_inferenceRunning || m_batchRunning) return;
                if (folder) {
                    openDirectory();
                } else {
                    const auto file = QFileDialog::getOpenFileName(
                            this, chatTr("Open Image File"),
                            QFileInfo(m_imagePath).absolutePath(),
                            chatTr("Images (*.png *.jpg *.jpeg *.bmp *.webp "
                                   "*.tif *.tiff)"));
                    if (!file.isEmpty()) openFile(file);
                }
            },
            [this](const QString& path, quint64 generation,
                   const QJsonArray& history, QString* error) {
                if (generation != m_sourceGeneration || m_sidecarBlocked ||
                    m_inferenceRunning || m_batchRunning) {
                    *error = chatTr("Annotation data changed or is busy.");
                    return false;
                }
                if (path == m_imagePath) {
                    m_document.insert("chat_history", history);
                    setDirty(true);
                    return true;
                }
                auto document = loadAnnotationDocument(path, m_outputDir,
                                                       m_exifScanEnabled);
                if (!document.error.isEmpty()) {
                    *error = document.error;
                    return false;
                }
                auto object = document.json;
                if (object.isEmpty()) {
                    object = QJsonObject{
                            {"version", annotationFormatVersion()},
                            {"flags", QJsonObject()},
                            {"shapes", QJsonArray()},
                            {"imagePath", QFileInfo(path).fileName()},
                            {"imageData", QJsonValue::Null},
                            {"imageHeight", document.image.height()},
                            {"imageWidth", document.image.width()},
                            {"description", ""}};
                }
                object.insert("chat_history", history);
                const auto bytes = QJsonDocument(object).toJson();
                QSaveFile file(sidecarPathFor(path));
                if (!file.open(QIODevice::WriteOnly) ||
                    file.write(bytes) != bytes.size() || !file.commit()) {
                    *error = file.errorString();
                    return false;
                }
                return true;
            },
            [this](const QString& file, QString* error) {
                if (m_imagePath.isEmpty() || m_sidecarBlocked ||
                    m_inferenceRunning || m_batchRunning || !confirmSave()) {
                    *error = chatTr("Annotation data changed or is busy.");
                    return -1;
                }
                const QString path = m_imagePath;
                const QString directory = QFileInfo(path).absolutePath();
                const int count = importChatDataset(
                        file, directory,
                        m_outputDir.isEmpty() ? directory : m_outputDir, error);
                if (count > 0) {
                    refreshFileList();
                    openFile(path);
                }
                return count;
            });
    connect(this, &AnnotationWindow::annotationSourceChanged, dialog,
            [dialog]() { dialog->sourceChanged(); });
    dialog->show();
}

void AnnotationWindow::openVqa() {
    if (auto* existing = findChild<QDialog*>(QStringLiteral("AnnotationVQA"))) {
        existing->show();
        existing->raise();
        existing->activateWindow();
        return;
    }
    auto source = [this]() {
        auto document = m_document;
        QJsonArray shapes;
        for (const auto& shape : m_canvas->shapes())
            shapes.append(shape.toJson());
        document.insert("shapes", shapes);
        document.insert("imagePath", QFileInfo(m_imagePath).fileName());
        document.insert("imageWidth", m_image.width());
        document.insert("imageHeight", m_image.height());
        return Source{m_imagePath, m_image, document, m_sourceGeneration};
    };
    auto* dialog = new VqaDialog(
            this, source,
            [this](const Source& source, const QJsonObject& data) {
                if (source.path != m_imagePath ||
                    source.generation != m_sourceGeneration ||
                    m_sidecarBlocked || m_imagePath.isEmpty())
                    return false;
                if (m_document.value("vqaData").toObject() != data) {
                    m_document.insert("vqaData", data);
                    setDirty(true);
                }
                return true;
            },
            [this](int direction) {
                if (m_inferenceRunning || m_batchRunning) return;
                const int row = m_files->currentRow() + direction;
                if (row >= 0 && row < m_files->count())
                    openFile(m_files->item(row)->data(Qt::UserRole).toString());
            },
            [this]() {
                QStringList files;
                for (int i = 0; i < m_files->count(); ++i)
                    files.append(
                            m_files->item(i)->data(Qt::UserRole).toString());
                return files;
            },
            [this, source](const QString& file) {
                if (file == m_imagePath) {
                    AnnotationDocument document;
                    if (m_sidecarBlocked) {
                        document.error = chatTr("Invalid image annotation");
                        return document;
                    }
                    const auto current = source();
                    document.image = current.image;
                    document.imagePath = current.path;
                    document.json = current.document;
                    return document;
                }
                return loadAnnotationDocument(file, m_outputDir,
                                              m_exifScanEnabled);
            },
            [this]() { openDirectory(); },
            [this](const QString& path) { return sidecarPathFor(path); },
            [this](const VqaDialog::Transform& transform, QString* error) {
                if (m_sidecarBlocked || m_inferenceRunning || m_batchRunning) {
                    *error = chatTr("Annotation data is unavailable or busy.");
                    return false;
                }
                QStringList labels;
                if (!m_outputDir.isEmpty()) {
                    const auto entries =
                            QDir(m_outputDir)
                                    .entryList({"*.json"}, QDir::Files,
                                               QDir::Name);
                    for (const auto& name : entries)
                        labels.append(QDir(m_outputDir).filePath(name));
                } else {
                    for (int i = 0; i < m_files->count(); ++i)
                        labels.append(
                                sidecarPathFor(m_files->item(i)
                                                       ->data(Qt::UserRole)
                                                       .toString()));
                }
                labels.removeDuplicates();
                struct Pending {
                    QString path;
                    QByteArray bytes;
                };
                QVector<Pending> pending;
                // O(n) preflight. Memory is the changed JSON bytes only; no
                // decoded images. Reject every malformed sidecar before
                // writing.
                for (const auto& path : labels) {
                    QFile file(path);
                    if (!file.exists()) continue;
                    if (!file.open(QIODevice::ReadOnly)) {
                        *error = path + ": " + file.errorString();
                        return false;
                    }
                    QJsonParseError parse;
                    const auto json =
                            QJsonDocument::fromJson(file.readAll(), &parse);
                    auto object = json.object();
                    const auto value = object.value("vqaData");
                    if (parse.error != QJsonParseError::NoError ||
                        !json.isObject() ||
                        (!value.isUndefined() && !value.isObject())) {
                        *error = chatTr("Invalid annotation: %1").arg(path);
                        return false;
                    }
                    // The active image is owned by the in-memory document,
                    // including unsaved geometry; setDirty handles its save.
                    if (!m_imagePath.isEmpty() &&
                        QFileInfo(path).absoluteFilePath() ==
                                QFileInfo(sidecarPathFor(m_imagePath))
                                        .absoluteFilePath())
                        continue;
                    if (value.isUndefined()) continue;
                    const auto updated = transform(value.toObject());
                    if (updated == value.toObject()) continue;
                    object.insert("vqaData", updated);
                    pending.append({path, QJsonDocument(object).toJson()});
                }
                const auto currentValue = m_document.value("vqaData");
                if (!currentValue.isUndefined() && !currentValue.isObject()) {
                    *error = chatTr("Invalid current VQA data.");
                    return false;
                }
                for (const auto& item : pending) {
                    QSaveFile file(item.path);
                    if (!file.open(QIODevice::WriteOnly) ||
                        file.write(item.bytes) != item.bytes.size() ||
                        !file.commit()) {
                        *error = item.path + ": " + file.errorString();
                        return false;
                    }
                }
                const auto updated = transform(currentValue.toObject());
                if (!m_imagePath.isEmpty() &&
                    updated != currentValue.toObject()) {
                    m_document.insert("vqaData", updated);
                    setDirty(true);
                }
                return true;
            });
    connect(this, &AnnotationWindow::annotationSourceChanged, dialog,
            [dialog]() { dialog->sourceChanged(); });
    dialog->show();
}
