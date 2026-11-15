// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "PpocrDialog.h"

#include <QApplication>
#include <QClipboard>
#include <QCloseEvent>
#include <QComboBox>
#include <QDateTime>
#include <QDir>
#include <QDragEnterEvent>
#include <QDropEvent>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QGraphicsItem>
#include <QGraphicsScene>
#include <QGraphicsView>
#include <QHBoxLayout>
#include <QImageReader>
#include <QInputDialog>
#include <QJsonArray>
#include <QJsonDocument>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QMessageBox>
#include <QMimeData>
#include <QMouseEvent>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QSaveFile>
#include <QScrollBar>
#include <QSignalBlocker>
#include <QSpinBox>
#include <QSplitter>
#include <QTabWidget>
#include <QTextBrowser>
#include <QTimer>
#include <QVBoxLayout>
#include <functional>

#include "PpocrDocument.h"
#include "PpocrRichTextEditor.h"
#include "PpocrServiceClient.h"
#include "PpocrTableEditor.h"
#include "QtCompat.h"

class PpocrPreview final : public QGraphicsView {
public:
    explicit PpocrPreview(QWidget* parent) : QGraphicsView(parent) {
        setScene(new QGraphicsScene(this));
        setDragMode(ScrollHandDrag);
        setMouseTracking(true);
        setAlignment(Qt::AlignCenter);
        setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
        setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    }
    std::function<void(const QString&)> selected;
    void showPage(const QImage& image,
                  const QVector<PpocrDocument::Block>& blocks,
                  int page) {
        scene()->clear();
        m_overlays.clear();
        if (!image.isNull()) scene()->addPixmap(QPixmap::fromImage(image));
        for (const auto& block : blocks) {
            if (block.page != page || block.polygon.isEmpty()) continue;
            const QColor color =
                    block.edited ? QColor(255, 156, 40)
                                 : PpocrDocument::categoryColor(block.label);
            QPen pen(color, 2);
            pen.setCosmetic(true);
            auto* item = scene()->addPolygon(block.polygon, pen);
            item->setData(0, block.key);
            item->setToolTip(block.text);
            item->setZValue(1);
            m_overlays.append(item);
        }
        scene()->setSceneRect(scene()->itemsBoundingRect());
        fitInView(sceneRect(), Qt::KeepAspectRatio);
        m_zoom = 1;
    }
    void highlight(const QString& key) {
        for (auto* item : m_overlays) {
            const bool match = item->data(0).toString() == key;
            item->setBrush(match ? QBrush(QColor(133, 144, 255, 60))
                                 : QBrush(Qt::NoBrush));
            if (match) ensureVisible(item);
        }
    }
    void zoom(double delta) {
        const double next = qBound(0.2, m_zoom + delta, 5.0);
        scale(next / m_zoom, next / m_zoom);
        m_zoom = next;
    }

protected:
    void resizeEvent(QResizeEvent* event) override {
        QGraphicsView::resizeEvent(event);
        if (!sceneRect().isEmpty()) {
            fitInView(sceneRect(), Qt::KeepAspectRatio);
            scale(m_zoom, m_zoom);
        }
    }
    void mousePressEvent(QMouseEvent* event) override {
        auto* item = itemAt(event->pos());
        if (item && !item->data(0).toString().isEmpty() && selected)
            selected(item->data(0).toString());
        QGraphicsView::mousePressEvent(event);
    }

private:
    QVector<QGraphicsPolygonItem*> m_overlays;
    double m_zoom = 1;
};

PpocrDialog::PpocrDialog(QWidget* parent, const QString& workDirectory)
    : QDialog(parent) {
    setObjectName("AnnotationPaddleOCR");
    setWindowTitle(tr("PaddleOCR"));
    if (parent) {
        setPalette(parent->palette());
        setStyleSheet(parent->styleSheet());
    }
    setWindowFlags(Qt::Window | Qt::WindowMinMaxButtonsHint |
                   Qt::WindowCloseButtonHint);
    setAttribute(Qt::WA_DeleteOnClose);
    setAcceptDrops(true);
    resize(1560, 920);
    setMinimumSize(1200, 760);
    m_root = QDir(workDirectory.isEmpty() ? QDir::homePath() : workDirectory)
                     .filePath("xanylabeling_data/paddleocr");
    QDir().mkpath(m_root + "/files");
    QDir().mkpath(m_root + "/jsons");
    m_state = load(m_root + "/ui_state.json");
    m_settings = load(m_root + "/api_settings.json");
    m_client = new PpocrServiceClient(this);
    auto* root = new QHBoxLayout(this);
    auto* sidebar = new QWidget(this);
    sidebar->setFixedWidth(240);
    auto* left = new QVBoxLayout(sidebar);
    root->addWidget(sidebar);
    auto button = [this](QBoxLayout* layout, const QString& text,
                         const char* name, std::function<void()> callback) {
        auto* result = new QPushButton(text, this);
        result->setObjectName(name);
        layout->addWidget(result);
        connect(result, &QPushButton::clicked, this, callback);
        return result;
    };
    button(left, tr("New parsing"), "PpocrImport", [this]() {
        confirmImport(QFileDialog::getOpenFileNames(
                this, tr("New parsing"), {},
                tr("Documents (*.bmp *.cif *.gif *.jpeg *.jpg *.png *.tif "
                   "*.tiff *.webp *.pdf)")));
    });
    m_search = new QLineEdit(sidebar);
    m_search->setObjectName("PpocrSearch");
    m_search->setPlaceholderText(tr("Search"));
    left->addWidget(m_search);
    m_tab = new QComboBox(sidebar);
    m_tab->addItems({tr("Recents"), tr("Favorites")});
    left->addWidget(m_tab);
    m_statusFilter = new QComboBox(sidebar);
    m_statusFilter->setObjectName("PpocrStatusFilter");
    m_statusFilter->addItem(tr("All"), "");
    for (const auto& state : {"pending", "parsed", "error"})
        m_statusFilter->addItem(QString::fromLatin1(state), state);
    left->addWidget(m_statusFilter);
    m_typeFilter = new QComboBox(sidebar);
    m_typeFilter->setObjectName("PpocrTypeFilter");
    m_typeFilter->addItem(tr("All file types"), "all");
    m_typeFilter->addItem(tr("Document"), "pdf");
    m_typeFilter->addItem(tr("Image"), "image");
    left->addWidget(m_typeFilter);
    m_sort = new QComboBox(sidebar);
    m_sort->setObjectName("PpocrSort");
    m_sort->addItem(tr("Newest First"), "newest");
    m_sort->addItem(tr("Oldest First"), "oldest");
    left->addWidget(m_sort);
    m_records = new QListWidget(sidebar);
    m_records->setObjectName("PpocrRecords");
    m_records->setSelectionMode(QAbstractItemView::ExtendedSelection);
    left->addWidget(m_records, 1);
    button(left, tr("Favorite / unfavorite"), "PpocrFavorite", [this]() {
        if (m_name.isEmpty()) return;
        auto values = m_state.value("favorites").toArray();
        if (values.contains(m_name)) {
            for (int i = values.size() - 1; i >= 0; --i)
                if (values[i] == m_name) values.removeAt(i);
        } else
            values.append(m_name);
        m_state.insert("favorites", values);
        QString error;
        if (!PpocrDocument::save(m_root + "/ui_state.json", m_state, &error))
            m_status->setText(error);
        refresh(m_name);
    });
    button(left, tr("Delete"), "PpocrDelete", [this]() {
        if (m_name.isEmpty() || m_name == m_active) return;
        if (QMessageBox::question(
                    this, tr("Delete"),
                    tr("Delete %1 and its parsing result?").arg(m_name)) !=
            QMessageBox::Yes)
            return;
        if (!QFile::remove(m_root + "/files/" + m_name)) {
            m_status->setText(tr("Could not delete source file"));
            return;
        }
        QFile::remove(jsonPath(m_name));
        refresh();
    });
    auto* workspace = new QWidget(this);
    auto* main = new QVBoxLayout(workspace);
    root->addWidget(workspace, 1);
    auto* controls = new QHBoxLayout;
    main->addLayout(controls);
    m_model = new QComboBox(this);
    m_model->setObjectName("PpocrModel");
    m_model->addItem("PaddleOCR-VL-1.6 (API)",
                     "__ppocr_api__:PaddleOCR-VL-1.6");
    m_model->addItem("PaddleOCR-VL (API)", "__ppocr_api__:PaddleOCR-VL");
    const QString apiModel =
            m_settings.value("api_model").toString("PaddleOCR-VL-1.6");
    const QString selected =
            m_settings.value("pipeline_model")
                    .toString(QStringLiteral("__ppocr_api__:%1").arg(apiModel));
    m_model->setCurrentIndex(qMax(0, m_model->findData(selected)));
    if (!selected.isEmpty() && m_model->findData(selected) < 0) {
        m_model->addItem(selected, selected);
        m_model->setCurrentIndex(m_model->count() - 1);
    }
    controls->addWidget(m_model);
    button(controls, tr("Refresh service"), "PpocrRefreshService",
           [this]() { probeService(); });
    button(controls, tr("API settings"), "PpocrSettings",
           [this]() { settings(); });
    button(controls, tr("Parse / retry selected"), "PpocrParse",
           [this]() { parseSelection(); });
    button(controls, tr("Cancel"), "PpocrCancel", [this]() {
        m_queue.clear();
        m_client->cancel();
    });
    controls->addStretch();
    m_status = new QLabel(this);
    m_status->setObjectName("PpocrStatus");
    m_status->setWordWrap(true);
    main->addWidget(m_status);
    auto* splitter = new QSplitter(Qt::Horizontal, this);
    main->addWidget(splitter, 1);
    auto* preview = new QWidget(splitter);
    auto* previewLayout = new QVBoxLayout(preview);
    m_preview = new PpocrPreview(preview);
    m_preview->setObjectName("PpocrPreview");
    previewLayout->addWidget(m_preview, 1);
    auto* pages = new QHBoxLayout;
    previewLayout->addLayout(pages);
    button(pages, tr("Previous"), "PpocrPrevious",
           [this]() { m_page->setValue(m_page->value() - 1); });
    m_page = new QSpinBox(this);
    m_page->setObjectName("PpocrPage");
    m_page->setMinimum(1);
    pages->addWidget(m_page);
    button(pages, tr("Next"), "PpocrNext",
           [this]() { m_page->setValue(m_page->value() + 1); });
    button(pages, "−", "PpocrZoomOut", [this]() { m_preview->zoom(-0.1); });
    button(pages, "+", "PpocrZoomIn", [this]() { m_preview->zoom(0.1); });
    button(pages, tr("Reset"), "PpocrZoomReset", [this]() { renderPage(); });
    auto* results = new QWidget(splitter);
    auto* right = new QVBoxLayout(results);
    auto* tabs = new QTabWidget(results);
    right->addWidget(tabs, 1);
    m_blocks = new QListWidget(tabs);
    m_blocks->setObjectName("PpocrBlocks");
    m_blocks->setWordWrap(true);
    tabs->addTab(m_blocks, tr("Document"));
    m_markdown = new QTextBrowser(tabs);
    m_markdown->setObjectName("PpocrMarkdown");
    tabs->addTab(m_markdown, tr("Markdown"));
    m_json = new QPlainTextEdit(tabs);
    m_json->setObjectName("PpocrJson");
    m_json->setReadOnly(true);
    tabs->addTab(m_json, tr("JSON"));
    auto* actions = new QHBoxLayout;
    right->addLayout(actions);
    button(actions, tr("Copy"), "PpocrCopy", [this, tabs]() {
        QString text = tabs->currentIndex() == 2
                               ? m_json->toPlainText()
                               : PpocrDocument::markdown(m_document);
        if (tabs->currentIndex() == 0 && m_blocks->currentItem())
            text = m_blocks->currentItem()->data(Qt::UserRole + 1).toString();
        QApplication::clipboard()->setText(text);
    });
    button(actions, tr("Correct block"), "PpocrEdit",
           [this]() { editBlock(); });
    button(actions, tr("Import JSON"), "PpocrImportJson", [this]() {
        if (m_name.isEmpty() || m_name == m_active) return;
        const auto path = QFileDialog::getOpenFileName(
                this, tr("Import JSON"), {},
                tr("OCR results (*.json *.jsonl)"));
        if (path.isEmpty()) return;
        QFile file(path);
        if (!file.open(QIODevice::ReadOnly)) {
            m_status->setText(file.errorString());
            return;
        }
        QString error;
        const auto document = PpocrDocument::parse(file.readAll(), &error);
        if (!error.isEmpty()) {
            m_status->setText(error);
            return;
        }
        m_document = document;
        if (saveCurrent()) render();
    });
    button(actions, tr("Download document"), "PpocrExportMarkdown",
           [this]() { exportResult(false); });
    button(actions, tr("Download JSON"), "PpocrExportJson",
           [this]() { exportResult(true); });
    splitter->setSizes({650, 650});
    splitter->setChildrenCollapsible(false);
    connect(m_search, &QLineEdit::textChanged, this,
            [this]() { refresh(m_name); });
    connect(m_tab, QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            [this]() { refresh(m_name); });
    connect(m_statusFilter, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, [this]() { refresh(m_name); });
    connect(m_typeFilter, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, [this]() { refresh(m_name); });
    connect(m_sort, QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            [this]() { refresh(m_name); });
    connect(m_records, &QListWidget::currentItemChanged, this,
            [this]() { selectRecord(); });
    connect(m_page, QOverload<int>::of(&QSpinBox::valueChanged), this,
            [this]() { renderPage(); });
    connect(m_blocks, &QListWidget::itemDoubleClicked, this,
            [this]() { editBlock(); });
    connect(m_blocks, &QListWidget::currentItemChanged, this, [this]() {
        if (!m_blocks->currentItem()) return;
        const auto key = m_blocks->currentItem()->data(Qt::UserRole).toString();
        for (const auto& block : PpocrDocument::blocks(m_document))
            if (block.key == key) {
                m_page->setValue(block.page + 1);
                break;
            }
        m_preview->highlight(key);
    });
    m_preview->selected = [this](const QString& key) {
        for (int i = 0; i < m_blocks->count(); ++i)
            if (m_blocks->item(i)->data(Qt::UserRole) == key) {
                m_blocks->setCurrentRow(i);
                m_blocks->scrollToItem(m_blocks->item(i));
                break;
            }
    };
    connect(m_client, &PpocrServiceClient::progress, m_status,
            &QLabel::setText);
    connect(m_model, QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            [this]() {
                m_settings.insert("pipeline_model",
                                  m_model->currentData().toString());
                if (m_model->currentIndex() < 2)
                    m_settings.insert(
                            "api_model",
                            m_model->currentData().toString().section(':', 1));
                saveSettings();
            });
    connect(m_client, &PpocrServiceClient::serviceProbed, this,
            [this](const QJsonArray& models, const QString& selected,
                   const QString& base, const QString& error) {
                if (!error.isEmpty()) {
                    m_status->setText(tr("Service offline: %1").arg(error));
                    return;
                }
                const QSignalBlocker blocker(m_model);
                while (m_model->count() > 2) m_model->removeItem(2);
                for (const auto& value : models) {
                    const auto model = value.toObject();
                    const auto id =
                            model.value("model_id").toString().trimmed();
                    if (id.isEmpty() || id == "ppocr_api" ||
                        m_model->findData(id) >= 0)
                        continue;
                    m_model->addItem(model.value("display_name").toString(),
                                     id);
                }
                m_model->setCurrentIndex(qMax(0, m_model->findData(selected)));
                m_settings.insert("pipeline_model", selected);
                saveSettings();
                m_status->setText(tr("Service online: %1").arg(base));
            });
    connect(m_client, &PpocrServiceClient::finished, this,
            [this](QJsonObject document, const QString& error) {
                if (document.isEmpty()) document = load(jsonPath(m_active));
                auto meta = document.value("_ppocr_meta").toObject();
                meta.insert("status", error.isEmpty() ? "parsed" : "error");
                meta.insert("error_message", error);
                meta.insert("source_path",
                            QStringLiteral("files/%1").arg(m_active));
                meta.insert("updated_at", QDateTime::currentDateTime().toString(
                                                  "yyyy-MM-dd HH:mm:ss"));
                document.insert("_ppocr_meta", meta);
                QString failure;
                const bool saved = PpocrDocument::save(jsonPath(m_active),
                                                       document, &failure);
                const auto selected = m_name;
                m_active.clear();
                refresh(selected);
                m_status->setText(
                        saved ? error.isEmpty() ? tr("Parsing complete") : error
                              : failure);
                QTimer::singleShot(0, this, [this]() { nextQueued(); });
            });
    refresh();
}
PpocrDialog::~PpocrDialog() {
    m_client->disconnect(this);
    m_client->cancel();
}
QString PpocrDialog::jsonPath(const QString& name) const {
    return m_root + "/jsons/" + name + ".json";
}
QJsonObject PpocrDialog::load(const QString& path) const {
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) return {};
    return QJsonDocument::fromJson(file.readAll()).object();
}
bool PpocrDialog::saveCurrent() {
    if (m_name.isEmpty()) return false;
    auto meta = m_document.value("_ppocr_meta").toObject();
    meta.insert("status", "parsed");
    meta.insert("source_path", QStringLiteral("files/%1").arg(m_name));
    meta.insert("updated_at",
                QDateTime::currentDateTime().toString("yyyy-MM-dd HH:mm:ss"));
    m_document.insert("_ppocr_meta", meta);
    QString error;
    if (!PpocrDocument::save(jsonPath(m_name), m_document, &error)) {
        m_status->setText(error);
        return false;
    }
    return true;
}
void PpocrDialog::refresh(const QString& selected) {
    const QSignalBlocker blocked(m_records);
    m_records->clear();
    const auto favorites = m_state.value("favorites").toArray();
    const auto files =
            QDir(m_root + "/files")
                    .entryInfoList(QDir::Files,
                                   m_sort->currentData() == "oldest"
                                           ? QDir::Time | QDir::Reversed
                                           : QDir::Time);
    for (const auto& file : files) {
        const auto name = file.fileName();
        const auto suffix = file.suffix().toLower();
        if (!QStringList{"bmp", "cif", "gif", "jpeg", "jpg", "png", "tif",
                         "tiff", "webp", "pdf"}
                     .contains(suffix))
            continue;
        const auto type = suffix == "pdf" ? QStringLiteral("pdf")
                                          : QStringLiteral("image");
        if (m_typeFilter->currentData() != "all" &&
            m_typeFilter->currentData() != type)
            continue;
        if (!name.contains(m_search->text(), Qt::CaseInsensitive) ||
            (m_tab->currentIndex() == 1 && !favorites.contains(name)))
            continue;
        const auto document = load(jsonPath(name));
        auto state =
                document.value("_ppocr_meta")
                        .toObject()
                        .value("status")
                        .toString(document.isEmpty() ? "pending" : "parsed");
        if (!QStringList{"pending", "parsed", "error"}.contains(state))
            state = "pending";
        if (!m_statusFilter->currentData().toString().isEmpty() &&
            state != m_statusFilter->currentData())
            continue;
        auto* item = new QListWidgetItem(
                (favorites.contains(name) ? QStringLiteral("★ ") : QString()) +
                        name + "\n" +
                        (name == m_active ? tr("parsing") : state),
                m_records);
        item->setData(Qt::UserRole, name);
        if (name == selected) m_records->setCurrentItem(item);
    }
    if (!m_records->currentItem() && m_records->count())
        m_records->setCurrentRow(0);
    selectRecord();
}
void PpocrDialog::selectRecord() {
    m_name = m_records->currentItem()
                     ? m_records->currentItem()->data(Qt::UserRole).toString()
                     : QString();
    m_document = m_name.isEmpty() ? QJsonObject() : load(jsonPath(m_name));
    render();
}
void PpocrDialog::render() {
    ++m_renderGeneration;
    m_editor.clear();
    const QSignalBlocker blocked(m_blocks);
    m_blocks->clear();
    for (const auto& block : PpocrDocument::blocks(m_document)) {
        auto* item = new QListWidgetItem(
                QStringLiteral("%1 · %2%3\n%4")
                        .arg(block.page + 1)
                        .arg(block.label,
                             block.edited ? tr(" · edited") : QString(),
                             block.text),
                m_blocks);
        item->setData(Qt::UserRole, block.key);
        item->setData(Qt::UserRole + 1, block.text);
    }
    m_json->setPlainText(
            m_document.isEmpty()
                    ? QString()
                    : QString::fromUtf8(QJsonDocument(m_document).toJson()));
    qtCompatSetMarkdown(*m_markdown->document(),
                        PpocrDocument::markdown(m_document));
    m_page->setMaximum(
            qMax(1, m_document.value("layoutParsingResults").toArray().size()));
    m_page->setValue(1);
    renderPage();
}
void PpocrDialog::renderPage() {
    QImage image;
    if (!m_name.isEmpty()) {
        QString path = m_root + "/files/" + m_name;
        if (QFileInfo(path).suffix().compare("pdf", Qt::CaseInsensitive) == 0) {
            // Display cached upstream PDF renders when available. Upload and
            // document editing also work on Qt builds without QtPdf.
            path = m_root + "/files/__PDF_" +
                   QFileInfo(m_name).completeBaseName() +
                   QStringLiteral("/page_%1.png")
                           .arg(m_page->value(), 3, 10, QLatin1Char('0'));
            if (!QFileInfo::exists(path))
                m_status->setText(tr(
                        "PDF preview requires rendered page images; document "
                        "parsing and text results remain available."));
        }
        QImageReader reader(path);
        reader.setAutoTransform(true);
        image = reader.read();
    }
    auto blocks = PpocrDocument::blocks(m_document);
    const auto pages = m_document.value("layoutParsingResults").toArray();
    if (!image.isNull() && m_page->value() <= pages.size()) {
        const auto dimensions = pages[m_page->value() - 1]
                                        .toObject()
                                        .value("prunedResult")
                                        .toObject();
        const double width = dimensions.value("width").toDouble(image.width());
        const double height =
                dimensions.value("height").toDouble(image.height());
        if (width > 0 && height > 0)
            for (auto& block : blocks)
                for (auto& point : block.polygon)
                    point = QPointF(point.x() * image.width() / width,
                                    point.y() * image.height() / height);
    }
    m_preview->showPage(image, blocks, m_page->value() - 1);
}
QStringList PpocrDialog::importFiles(const QStringList& paths) {
    QStringList imported;
    for (const auto& path : paths) {
        const QFileInfo source(path);
        const auto suffix = source.suffix().toLower();
        if (!QStringList{"bmp", "cif", "gif", "jpeg", "jpg", "png", "tif",
                         "tiff", "webp", "pdf"}
                     .contains(suffix))
            continue;
        QString name = source.fileName();
        int copy = 1;
        while (QFileInfo::exists(m_root + "/files/" + name))
            name = source.completeBaseName() +
                   QStringLiteral("_%1.").arg(copy++) + suffix;
        if (!QFile::copy(path, m_root + "/files/" + name)) {
            m_status->setText(tr("Cannot import %1").arg(path));
            continue;
        }
        imported.append(name);
    }
    refresh(imported.isEmpty() ? m_name : imported.first());
    return imported;
}
void PpocrDialog::parseSelection() {
    QStringList selected;
    for (const auto* item : m_records->selectedItems())
        selected.append(item->data(Qt::UserRole).toString());
    startParsing(selected);
}
void PpocrDialog::startParsing(const QStringList& names) {
    if (m_client->busy()) return;
    if (names.isEmpty()) return;
    if (m_model->currentIndex() < 2 &&
        m_settings.value("api_key").toString().trimmed().isEmpty()) {
        settings();
        if (m_settings.value("api_key").toString().trimmed().isEmpty()) return;
    }
    m_queue = names;
    nextQueued();
}
void PpocrDialog::confirmImport(const QStringList& paths) {
    if (paths.isEmpty()) return;
    QMessageBox question(QMessageBox::Question, tr("Confirm Parsing"),
                         tr("Start parsing selected files?"),
                         QMessageBox::Ok | QMessageBox::Cancel, this);
    question.setDefaultButton(QMessageBox::Cancel);
    question.setEscapeButton(QMessageBox::Cancel);
    QStringList preview;
    for (const auto& path : paths.mid(0, 5))
        preview.append(QFileInfo(path).fileName());
    if (paths.size() > 5)
        preview.append(tr("...and %1 more").arg(paths.size() - 5));
    question.setInformativeText(preview.join('\n'));
    question.button(QMessageBox::Ok)->setText(tr("Start Parsing"));
    if (question.exec() == QMessageBox::Ok) startParsing(importFiles(paths));
}
void PpocrDialog::nextQueued() {
    if (m_client->busy() || m_queue.isEmpty()) return;
    m_active = m_queue.takeFirst();
    PpocrServiceClient::Settings settings;
    settings.apiKey = m_settings.value("api_key").toString();
    settings.model = m_model->currentData().toString();
    if (m_model->currentIndex() < 2)
        settings.model = settings.model.section(':', 1);
    settings.jobsUrl = QUrl(
            m_settings.value("api_url").toString(settings.jobsUrl.toString()));
    const auto selected = m_name;
    refresh(selected);
    if (m_model->currentIndex() < 2) {
        m_client->parseFile(m_root + "/files/" + m_active, settings);
    } else {
        const auto configured =
                m_settings.value("remote_server_settings").toObject();
        PpocrServiceClient::RemoteSettings remote;
        remote.serverUrl =
                configured.value("server_url").toString(remote.serverUrl);
        remote.apiKey = configured.value("api_key").toString();
        remote.timeoutMs = configured.value("timeout").toInt(180) * 1000;
        m_client->parseRemoteFile(m_root + "/files/" + m_active, remote,
                                  settings.model);
    }
}
void PpocrDialog::editBlock() {
    if (!m_blocks->currentItem() || m_name == m_active) return;
    const auto key = m_blocks->currentItem()->data(Qt::UserRole).toString();
    bool ok = false;
    const auto original =
            m_blocks->currentItem()->data(Qt::UserRole + 1).toString();
    QString label, imagePath;
    for (const auto& block : PpocrDocument::blocks(m_document))
        if (block.key == key) {
            label = block.label;
            if (!block.imagePath.isEmpty())
                imagePath = QDir(m_root).filePath(block.imagePath);
            break;
        }
    QString text;
    const auto normalized = label.trimmed().toLower();
    const bool formula = QStringList{"display_formula", "formula_number",
                                     "algorithm", "formula"}
                                 .contains(normalized);
    if (!formula) {
        // Replace only this block with an inline editor. A refresh or source
        // switch invalidates its callbacks before the old widget is deleted.
        if (m_editor) render();
        QListWidgetItem* item = nullptr;
        for (int i = 0; i < m_blocks->count(); ++i)
            if (m_blocks->item(i)->data(Qt::UserRole).toString() == key) {
                item = m_blocks->item(i);
                break;
            }
        if (!item) return;
        auto* table =
                PpocrTable::isTable(label, original)
                        ? new PpocrTableEditor(original, m_blocks->viewport())
                        : nullptr;
        auto* rich = table ? nullptr
                           : new PpocrRichTextEditor(
                                     original, m_blocks->viewport(), imagePath);
        QWidget* editor = table ? static_cast<QWidget*>(table) : rich;
        m_editor = editor;
        const auto generation = m_renderGeneration;
        const auto source = m_name;
        const auto current = [this, editor, generation, source]() {
            return m_editor == editor && generation == m_renderGeneration &&
                   source == m_name && m_name != m_active;
        };
        const auto save = [this, current, key](const QString& value) {
            if (current()) commitBlockEdit(key, value);
        };
        const auto cancel = [this, current, key]() {
            if (!current()) return;
            render();
            for (int i = 0; i < m_blocks->count(); ++i)
                if (m_blocks->item(i)->data(Qt::UserRole) == key) {
                    m_blocks->setCurrentRow(i);
                    break;
                }
        };
        const auto geometry = [current, item, editor]() {
            if (!current()) return;
            const QSize size(0, editor->sizeHint().height());
            if (item->sizeHint() != size) item->setSizeHint(size);
        };
        if (table) {
            connect(table, &PpocrTableEditor::saveRequested, this, save);
            connect(table, &PpocrTableEditor::cancelRequested, this, cancel);
            connect(table, &PpocrTableEditor::geometryChanged, this, geometry);
        } else {
            connect(rich, &PpocrRichTextEditor::saveRequested, this, save);
            connect(rich, &PpocrRichTextEditor::cancelRequested, this, cancel);
            connect(rich, &PpocrRichTextEditor::geometryChanged, this,
                    geometry);
        }
        item->setText({});
        item->setSizeHint(QSize(0, editor->sizeHint().height()));
        m_blocks->setItemWidget(item, editor);
        m_blocks->setCurrentItem(item);
        m_blocks->scrollToItem(item);
        editor->show();
        return;
    } else {
        text = QInputDialog::getMultiLineText(this, tr("Correct block"),
                                              tr("Content"), original, &ok);
    }
    if (!ok) return;
    commitBlockEdit(key, text);
}
void PpocrDialog::commitBlockEdit(const QString& key, const QString& text) {
    auto updated = m_document;
    if (!PpocrDocument::edit(&updated, key, text)) return;
    const auto previous = m_document;
    m_document = updated;
    if (!saveCurrent()) {
        m_document = previous;
        return;
    }
    render();
    for (int i = 0; i < m_blocks->count(); ++i)
        if (m_blocks->item(i)->data(Qt::UserRole) == key) {
            m_blocks->setCurrentRow(i);
            break;
        }
}
void PpocrDialog::settings() {
    QDialog dialog(this);
    dialog.setObjectName("PpocrApiSettings");
    dialog.setWindowTitle(tr("PaddleOCR API settings"));
    dialog.setMinimumWidth(600);
    QFormLayout layout(&dialog);
    QLineEdit key(m_settings.value("api_key").toString(), &dialog);
    key.setObjectName("PpocrApiKey");
    key.setEchoMode(QLineEdit::Password);
    QLineEdit url(m_settings.value("api_url").toString(
                          PpocrServiceClient::Settings().jobsUrl.toString()),
                  &dialog);
    url.setObjectName("PpocrApiUrl");
    layout.addRow(tr("Jobs URL"), &url);
    layout.addRow(tr("API key"), &key);
    const auto remote = m_settings.value("remote_server_settings").toObject();
    QLineEdit server(
            remote.value("server_url").toString("http://127.0.0.1:8000"),
            &dialog);
    server.setObjectName("PpocrServerUrl");
    QLineEdit token(remote.value("api_key").toString(), &dialog);
    token.setObjectName("PpocrServerToken");
    token.setEchoMode(QLineEdit::Password);
    QSpinBox timeout(&dialog);
    timeout.setObjectName("PpocrServerTimeout");
    timeout.setRange(1, 3600);
    timeout.setValue(remote.value("timeout").toInt(180));
    timeout.setSuffix(tr(" s"));
    layout.addRow(tr("Remote server URL"), &server);
    layout.addRow(tr("Remote server token (optional)"), &token);
    layout.addRow(tr("Remote timeout"), &timeout);
    QPushButton save(tr("Save"), &dialog);
    save.setObjectName("PpocrSettingsSave");
    layout.addRow(&save);
    connect(&save, &QPushButton::clicked, &dialog, &QDialog::accept);
    if (dialog.exec() != QDialog::Accepted) return;
    m_settings.insert("api_key", key.text().trimmed());
    if (m_model->currentIndex() < 2)
        m_settings.insert("api_model",
                          m_model->currentData().toString().section(':', 1));
    m_settings.insert("remote_server_settings",
                      QJsonObject{{"server_url", server.text().trimmed()},
                                  {"api_key", token.text().trimmed()},
                                  {"timeout", timeout.value()}});
    if (!url.text().trimmed().isEmpty())
        m_settings.insert("api_url", url.text().trimmed());
    m_settings.insert("api_mode", "async_jobs");
    saveSettings();
}
void PpocrDialog::saveSettings() {
    QString error;
    if (!PpocrDocument::save(m_root + "/api_settings.json", m_settings, &error))
        m_status->setText(error);
    QFile::setPermissions(m_root + "/api_settings.json",
                          QFileDevice::ReadOwner | QFileDevice::WriteOwner);
}
void PpocrDialog::probeService() {
    if (m_client->busy()) return;
    const auto configured =
            m_settings.value("remote_server_settings").toObject();
    PpocrServiceClient::RemoteSettings remote;
    remote.serverUrl =
            configured.value("server_url").toString(remote.serverUrl);
    remote.apiKey = configured.value("api_key").toString();
    remote.timeoutMs = configured.value("timeout").toInt(180) * 1000;
    m_client->probeService(remote, m_model->currentData().toString());
}
void PpocrDialog::exportResult(bool json) {
    if (m_document.isEmpty()) return;
    if (!json) {
        auto path = QFileDialog::getSaveFileName(
                this, tr("Download Document Parsing"),
                QFileInfo(m_name).completeBaseName() + "_document_parsing.zip",
                tr("ZIP Files (*.zip)"));
        if (path.isEmpty()) return;
        if (!path.endsWith(".zip", Qt::CaseInsensitive)) path += ".zip";
        QStringList pagePaths;
        const QString source = m_root + "/files/" + m_name;
        if (QFileInfo(m_name).suffix().compare("pdf", Qt::CaseInsensitive) != 0)
            pagePaths.append(source);
        else
            for (int p = 1;
                 p <= m_document.value("layoutParsingResults").toArray().size();
                 ++p) {
                const QString rendered =
                        m_root + "/files/__PDF_" +
                        QFileInfo(m_name).completeBaseName() +
                        QStringLiteral("/page_%1.png")
                                .arg(p, 3, 10, QLatin1Char('0'));
                pagePaths.append(QFileInfo::exists(rendered) ? rendered
                                                             : QString());
            }
        QString error;
        if (!PpocrDocument::exportBundle(m_document, m_root, pagePaths, path,
                                         &error))
            m_status->setText(error);
        else
            m_status->setText(tr("Download Successful"));
        return;
    }
    const auto path = QFileDialog::getSaveFileName(
            this, tr("Download result"), m_name + (json ? ".json" : ".md"),
            json ? tr("JSON (*.json)") : tr("Markdown (*.md)"));
    if (path.isEmpty()) return;
    QSaveFile file(path);
    const auto bytes = json ? QJsonDocument(m_document).toJson()
                            : PpocrDocument::markdown(m_document).toUtf8();
    if (!file.open(QIODevice::WriteOnly) || file.write(bytes) != bytes.size() ||
        !file.commit())
        m_status->setText(file.errorString());
}
void PpocrDialog::dragEnterEvent(QDragEnterEvent* event) {
    if (event->mimeData()->hasUrls()) event->acceptProposedAction();
}
void PpocrDialog::dropEvent(QDropEvent* event) {
    QStringList paths;
    for (const auto& url : event->mimeData()->urls())
        if (url.isLocalFile()) paths.append(url.toLocalFile());
    confirmImport(paths);
    event->acceptProposedAction();
}
void PpocrDialog::closeEvent(QCloseEvent* event) {
    m_queue.clear();
    m_client->cancel();
    QDialog::closeEvent(event);
}
