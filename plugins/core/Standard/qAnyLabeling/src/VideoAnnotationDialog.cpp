// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "VideoAnnotationDialog.h"

#include <QAbstractItemView>
#include <QCheckBox>
#include <QCloseEvent>
#include <QColorDialog>
#include <QComboBox>
#include <QCoreApplication>
#include <QDialogButtonBox>
#include <QDir>
#include <QDoubleSpinBox>
#include <QDragEnterEvent>
#include <QDragMoveEvent>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QFutureWatcher>
#include <QHBoxLayout>
#include <QHash>
#include <QHeaderView>
#include <QInputDialog>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QMenu>
#include <QMessageBox>
#include <QMimeData>
#include <QMouseEvent>
#include <QPainter>
#include <QPlainTextEdit>
#include <QProgressDialog>
#include <QPushButton>
#include <QRegularExpression>
#include <QSet>
#include <QShortcut>

// Qt 5.14 compatibility: the QShortcut(QKeySequence, QObject*, Functor)
// overload only exists from Qt 5.15. Route every shortcut through this
// helper, which connects activated() to the functor explicitly.
static QShortcut* addShortcut(QWidget* parent,
                              const QKeySequence& key,
                              std::function<void()> fn) {
    QShortcut* sc = new QShortcut(key, parent);
    QObject::connect(sc, &QShortcut::activated, parent, std::move(fn));
    return sc;
}
#include <QSpinBox>
#include <QSplitter>
#include <QTableWidget>
#include <QTableWidgetItem>
#include <QTextStream>
#include <QUuid>
#include <QVBoxLayout>
#include <QtConcurrent>
#include <algorithm>
#include <cmath>
#include <functional>

#include "VideoDatasetExporter.h"
#include "VideoPlaybackWidget.h"

namespace {
const QStringList kVideoExtensions = {
        QStringLiteral("mp4"), QStringLiteral("avi"),  QStringLiteral("mov"),
        QStringLiteral("mkv"), QStringLiteral("webm"), QStringLiteral("m4v")};

bool acceptsVideoDrop(const QMimeData* mime) {
    if (!mime->hasUrls()) return false;
    for (const QUrl& url : mime->urls()) {
        const QFileInfo info(url.toLocalFile());
        if (info.isFile() && kVideoExtensions.contains(info.suffix().toLower()))
            return true;
    }
    return false;
}

QString formatTime(qint64 ms) {
    const qint64 seconds = std::max<qint64>(0, ms) / 1000;
    return QStringLiteral("%1:%2:%3.%4")
            .arg(seconds / 3600, 2, 10, QLatin1Char('0'))
            .arg((seconds / 60) % 60, 2, 10, QLatin1Char('0'))
            .arg(seconds % 60, 2, 10, QLatin1Char('0'))
            .arg(std::max<qint64>(0, ms) % 1000, 3, 10, QLatin1Char('0'));
}

QString colorFor(int index) {
    static const char* colors[] = {"#EF476F", "#FFD166", "#06D6A0", "#118AB2",
                                   "#8338EC", "#FF7F50", "#3A86FF", "#FB5607",
                                   "#2EC4B6", "#9D4EDD", "#FF006E", "#43AA8B"};
    return QString::fromLatin1(colors[index % 12]);
}

QString safeSegmentToken(QString value) {
    value.replace(QRegularExpression(QStringLiteral("[^A-Za-z0-9._-]+")),
                  QStringLiteral("_"));
    while (!value.isEmpty() && (value.front() == QLatin1Char('.') ||
                                value.front() == QLatin1Char('_')))
        value.remove(0, 1);
    return value.isEmpty() ? QStringLiteral("segment") : value;
}

QVector<QImage> videoThumbnails(const QString& path) {
    QVector<QImage> images;
#ifdef HAS_OPENCV_FACE_CAPTURE
    try {
        cv::VideoCapture capture(path.toStdString());
        if (!capture.isOpened()) return images;
        const int frameCount = int(capture.get(cv::CAP_PROP_FRAME_COUNT));
        const int count = std::min(16, frameCount);
        if (count <= 0) return images;
        images.reserve(count);
        for (int i = 0; i < count; ++i) {
            const int frameIndex =
                    count == 1 ? 0
                               : int(std::llround(double(i) * (frameCount - 1) /
                                                  (count - 1)));
            if (!capture.set(cv::CAP_PROP_POS_FRAMES, frameIndex)) continue;
            cv::Mat frame;
            if (!capture.read(frame) || frame.empty()) continue;
            const QImage image = VideoPlaybackWidget::cvMatToQImage(frame);
            if (!image.isNull())
                images.append(
                        image.scaledToWidth(160, Qt::SmoothTransformation));
        }
    } catch (const cv::Exception&) {
        images.clear();
    }
#else
    Q_UNUSED(path)
#endif
    return images;
}

bool editVideoLabels(QWidget* parent,
                     const QStringList& currentLabels,
                     const QJsonObject& currentColors,
                     QStringList* labels,
                     QJsonObject* colors,
                     QStringList* deletedLabels,
                     QHash<QString, QString>* renamedLabels) {
    QDialog dialog(parent);
    dialog.setWindowTitle(QObject::tr("Label Settings"));
    dialog.resize(600, 360);
    auto* layout = new QVBoxLayout(&dialog);
    auto* table = new QTableWidget(&dialog);
    table->setObjectName(QStringLiteral("VideoLabelSettingsTable"));
    table->setColumnCount(4);
    table->setHorizontalHeaderLabels(
            {QObject::tr("Category"), QObject::tr("Delete"),
             QObject::tr("Name"), QObject::tr("Color")});
    table->setSelectionMode(QAbstractItemView::NoSelection);
    table->verticalHeader()->hide();
    table->horizontalHeader()->setSectionResizeMode(
            0, QHeaderView::ResizeToContents);
    table->horizontalHeader()->setSectionResizeMode(
            1, QHeaderView::ResizeToContents);
    table->horizontalHeader()->setSectionResizeMode(2, QHeaderView::Stretch);
    table->horizontalHeader()->setSectionResizeMode(
            3, QHeaderView::ResizeToContents);
    const auto addRow = [table, &dialog](const QString& original,
                                         const QString& name,
                                         const QString& color) {
        const int row = table->rowCount();
        table->insertRow(row);
        auto* originalItem = new QTableWidgetItem(
                original.isEmpty() ? QObject::tr("(new label)") : original);
        originalItem->setData(Qt::UserRole, original);
        originalItem->setFlags(originalItem->flags() & ~Qt::ItemIsEditable);
        table->setItem(row, 0, originalItem);
        auto* deleted = new QCheckBox(table);
        deleted->setEnabled(!original.isEmpty());
        auto* deleteCell = new QWidget(table);
        auto* deleteLayout = new QHBoxLayout(deleteCell);
        deleteLayout->setContentsMargins(0, 0, 0, 0);
        deleteLayout->setAlignment(Qt::AlignCenter);
        deleteLayout->addWidget(deleted);
        table->setCellWidget(row, 1, deleteCell);
        auto* nameEdit = new QLineEdit(name, table);
        nameEdit->setObjectName(QStringLiteral("VideoLabelName"));
        table->setCellWidget(row, 2, nameEdit);
        auto* colorButton = new QPushButton(table);
        colorButton->setObjectName(QStringLiteral("VideoLabelColor"));
        colorButton->setFixedSize(30, 24);
        colorButton->setProperty("labelColor", color);
        colorButton->setStyleSheet(
                QStringLiteral("background-color: %1;").arg(color));
        QObject::connect(colorButton, &QPushButton::clicked, &dialog,
                         [colorButton, &dialog]() {
                             const QColor selected = QColorDialog::getColor(
                                     QColor(colorButton->property("labelColor")
                                                    .toString()),
                                     &dialog, QObject::tr("Label color"));
                             if (!selected.isValid()) return;
                             colorButton->setProperty("labelColor",
                                                      selected.name());
                             colorButton->setStyleSheet(
                                     QStringLiteral("background-color: %1;")
                                             .arg(selected.name()));
                         });
        auto* colorCell = new QWidget(table);
        auto* colorLayout = new QHBoxLayout(colorCell);
        colorLayout->setContentsMargins(2, 0, 2, 0);
        colorLayout->setAlignment(Qt::AlignCenter);
        colorLayout->addWidget(colorButton);
        table->setCellWidget(row, 3, colorCell);
        QObject::connect(deleted, &QCheckBox::toggled, nameEdit,
                         &QLineEdit::setDisabled);
        QObject::connect(deleted, &QCheckBox::toggled, colorButton,
                         &QPushButton::setDisabled);
    };
    int colorIndex = 0;
    for (const QString& label : currentLabels) {
        addRow(label, label,
               currentColors.value(label).toString(colorFor(colorIndex++)));
    }
    layout->addWidget(table, 1);
    auto* actions = new QHBoxLayout;
    auto* import = new QPushButton(QObject::tr("Import labels..."), &dialog);
    import->setObjectName(QStringLiteral("ImportVideoLabels"));
    actions->addWidget(import);
    actions->addStretch();
    auto* buttons = new QDialogButtonBox(
            QDialogButtonBox::Ok | QDialogButtonBox::Cancel, &dialog);
    actions->addWidget(buttons);
    layout->addLayout(actions);
    QObject::connect(import, &QPushButton::clicked, &dialog, [&, table]() {
        const QString path = QFileDialog::getOpenFileName(
                &dialog, QObject::tr("Import labels"), {},
                QObject::tr("Text files (*.txt);;All files (*)"));
        if (path.isEmpty()) return;
        QFile file(path);
        if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
            QMessageBox::warning(&dialog, QObject::tr("Import labels"),
                                 file.errorString());
            return;
        }
        QTextStream stream(&file);
#if QT_VERSION < QT_VERSION_CHECK(6, 0, 0)
        stream.setCodec("UTF-8");
#endif
        QSet<QString> existing;
        for (int row = 0; row < table->rowCount(); ++row) {
            existing.insert(table->item(row, 0)->data(Qt::UserRole).toString());
            existing.insert(qobject_cast<QLineEdit*>(table->cellWidget(row, 2))
                                    ->text()
                                    .trimmed());
        }
        int added = 0;
        while (!stream.atEnd()) {
            QString name = stream.readLine().trimmed();
            if (!name.isEmpty() && name.front() == QChar(0xfeff))
                name.remove(0, 1);
            if (name.isEmpty() || existing.contains(name)) continue;
            existing.insert(name);
            addRow({}, name, colorFor(table->rowCount()));
            ++added;
        }
        if (!added)
            QMessageBox::information(&dialog, QObject::tr("Import labels"),
                                     QObject::tr("No new labels found."));
    });
    QObject::connect(buttons, &QDialogButtonBox::rejected, &dialog,
                     &QDialog::reject);
    QObject::connect(buttons, &QDialogButtonBox::accepted, &dialog, [&]() {
        QStringList finalNames;
        QStringList deleted;
        QStringList nextLabels;
        QJsonObject nextColors;
        QHash<QString, QString> renamed;
        for (int row = 0; row < table->rowCount(); ++row) {
            const QString original =
                    table->item(row, 0)->data(Qt::UserRole).toString();
            auto* deleteCell = table->cellWidget(row, 1);
            auto* check = deleteCell->findChild<QCheckBox*>();
            if (check && check->isChecked()) {
                if (!original.isEmpty()) deleted.append(original);
                continue;
            }
            const QString name =
                    qobject_cast<QLineEdit*>(table->cellWidget(row, 2))
                            ->text()
                            .trimmed();
            if (name.isEmpty()) {
                QMessageBox::information(
                        &dialog, QObject::tr("Label Settings"),
                        QObject::tr("Label name cannot be empty."));
                return;
            }
            if (finalNames.contains(name)) {
                QMessageBox::information(
                        &dialog, QObject::tr("Label Settings"),
                        QObject::tr("Label names must be unique."));
                return;
            }
            finalNames.append(name);
            nextLabels.append(name);
            if (!original.isEmpty() && original != name)
                renamed.insert(original, name);
            auto* colorButton = qobject_cast<QPushButton*>(
                    table->cellWidget(row, 3)->findChild<QPushButton*>());
            nextColors.insert(name,
                              colorButton->property("labelColor").toString());
        }
        if (!deleted.isEmpty() &&
            QMessageBox::warning(
                    &dialog, QObject::tr("Delete labels"),
                    QObject::tr("Deleting these labels also removes their "
                                "segments. Continue?"),
                    QMessageBox::Yes | QMessageBox::No,
                    QMessageBox::No) != QMessageBox::Yes)
            return;
        std::sort(nextLabels.begin(), nextLabels.end(),
                  [](const QString& left, const QString& right) {
                      return left.compare(right, Qt::CaseInsensitive) < 0;
                  });
        *labels = nextLabels;
        *colors = nextColors;
        *deletedLabels = deleted;
        *renamedLabels = renamed;
        dialog.accept();
    });
    return dialog.exec() == QDialog::Accepted;
}
}  // namespace

class SegmentTimeline final : public QWidget {
public:
    using QWidget::QWidget;
    std::function<void(qint64)> seekRequested;
    std::function<void(qint64, qint64)> rangeCreated;
    std::function<void(int)> segmentSelected;
    std::function<void(int, qint64, qint64)> segmentEdited;
    std::function<void()> segmentEditFinished;

    void setData(const VideoSidecar& data, qint64 playhead, int selected) {
        m_data = data;
        m_playhead = playhead;
        m_selected = selected;
        if (!m_dragging) m_mode = Mode::Idle;
        update();
    }

    void setThumbnails(QVector<QImage> images) {
        m_thumbnails = std::move(images);
        setProperty("thumbnailCount", m_thumbnails.size());
        update();
    }

    QSize sizeHint() const override { return {600, 100}; }

protected:
    void paintEvent(QPaintEvent*) override {
        QPainter painter(this);
        painter.fillRect(rect(), palette().window());
        const QRect track = rect().adjusted(12, 30, -12, -12);
        painter.fillRect(track, QColor("#252b30"));
        if (m_data.durationMs <= 0) return;
        if (!m_thumbnails.isEmpty()) {
            painter.save();
            painter.setClipRect(track);
            const QImage& first = m_thumbnails.first();
            const int tileWidth = std::max(
                    24,
                    int(std::llround(double(track.height()) * first.width() /
                                     std::max(1, first.height()))));
            for (int x = track.left(); x <= track.right(); x += tileWidth) {
                const qint64 center = timeAt(track, x + tileWidth / 2);
                const int index = std::clamp(
                        int(std::llround(double(center) / m_data.durationMs *
                                         (m_thumbnails.size() - 1))),
                        0, m_thumbnails.size() - 1);
                const QImage& image = m_thumbnails[index];
                if (image.isNull()) continue;
                const QRect target(x, track.top(), tileWidth + 1,
                                   track.height());
                const double ratio = double(target.width()) / target.height();
                QRect source = image.rect();
                if (double(image.width()) / image.height() > ratio) {
                    source.setWidth(std::max(
                            1, int(std::llround(image.height() * ratio))));
                    source.moveLeft((image.width() - source.width()) / 2);
                } else {
                    source.setHeight(std::max(
                            1, int(std::llround(image.width() / ratio))));
                    source.moveTop((image.height() - source.height()) / 2);
                }
                painter.drawImage(target, image, source);
            }
            painter.fillRect(track, QColor(20, 24, 28, 90));
            painter.restore();
        }
        painter.setPen(palette().text().color());
        for (int tick = 0; tick <= 4; ++tick) {
            const int x = track.left() + track.width() * tick / 4;
            painter.drawLine(x, 20, x, 27);
            painter.drawText(
                    x - 25, 15,
                    formatTime(m_data.durationMs * tick / 4).mid(3, 8));
        }
        for (int i = 0; i < m_data.segments.size(); ++i) {
            const VideoSegment& segment = m_data.segments[i];
            const int left = xAt(track, segment.startMs);
            const int right = xAt(track, segment.endMs);
            const QColor color(m_data.labelColors.value(segment.label)
                                       .toString("#118AB2"));
            QColor overlay = color;
            overlay.setAlpha(160);
            painter.fillRect(QRect(QPoint(left, track.top() + 4),
                                   QPoint(std::max(left + 2, right),
                                          track.bottom() - 4)),
                             overlay);
            if (i == m_selected) {
                painter.setPen(QPen(palette().text().color(), 2));
                painter.drawRect(QRect(
                        QPoint(left, track.top() + 4),
                        QPoint(std::max(left + 2, right), track.bottom() - 4)));
            }
            painter.setPen(QColor("#111111"));
            painter.drawText(
                    QRect(left + 4, track.top() + 5,
                          std::max(0, right - left - 8), track.height() - 10),
                    Qt::AlignVCenter, segment.label);
            if (i == m_selected) {
                painter.setPen(QPen(palette().text().color(), 2));
                painter.drawLine(left, track.top(), left, track.bottom());
                painter.drawLine(right, track.top(), right, track.bottom());
            }
        }
        if (m_mode == Mode::Create) {
            painter.fillRect(
                    QRect(QPoint(std::min(m_pressX, m_dragX), track.top()),
                          QPoint(std::max(m_pressX, m_dragX), track.bottom())),
                    QColor(255, 255, 255, 50));
        }
        painter.setPen(QPen(QColor("#f6f6f6"), 2));
        const int playX = xAt(track, m_playhead);
        painter.drawLine(playX, track.top(), playX, track.bottom());
    }

    void mousePressEvent(QMouseEvent* event) override {
        const QRect track = rect().adjusted(12, 30, -12, -12);
        const QPoint pos = event->pos();
        if (event->button() == Qt::LeftButton && pos.y() < track.top() &&
            pos.x() >= track.left() && pos.x() <= track.right()) {
            m_mode = Mode::Seek;
            m_dragging = true;
            if (seekRequested) seekRequested(timeAt(track, pos.x()));
            return;
        }
        if (event->button() == Qt::RightButton &&
            (pos.y() < track.top() || track.contains(pos))) {
            beginCreate(track, pos.x());
            return;
        }
        if (event->button() != Qt::LeftButton || !track.contains(pos)) return;
        const Hit hit = hitTest(track, pos.x());
        if (hit.kind == HitKind::None) {
            beginCreate(track, pos.x());
            return;
        }
        m_dragging = true;
        m_mode = hit.kind == HitKind::Left    ? Mode::ResizeLeft
                 : hit.kind == HitKind::Right ? Mode::ResizeRight
                                              : Mode::Move;
        m_active = hit.index;
        m_pressX = pos.x();
        m_dragX = m_pressX;
        m_initialStart = m_data.segments[m_active].startMs;
        m_initialEnd = m_data.segments[m_active].endMs;
        m_dragOffset = timeAt(track, pos.x()) - m_initialStart;
        if (segmentSelected) segmentSelected(m_active);
    }

    void mouseMoveEvent(QMouseEvent* event) override {
        const QRect track = rect().adjusted(12, 30, -12, -12);
        if (m_mode == Mode::Seek) {
            if (seekRequested) seekRequested(timeAt(track, event->pos().x()));
            return;
        }
        if (m_mode == Mode::Create) {
            m_dragX = event->pos().x();
            update();
            return;
        }
        if (!m_dragging || m_active < 0 || m_active >= m_data.segments.size())
            return;
        const qint64 ms = snapTime(timeAt(track, event->pos().x()));
        VideoSegment& segment = m_data.segments[m_active];
        const qint64 oldStart = segment.startMs;
        const qint64 oldEnd = segment.endMs;
        const qint64 minimum = minSegmentMs();
        if (m_mode == Mode::Move) {
            const qint64 length = m_initialEnd - m_initialStart;
            const auto bounds = moveBounds(m_active, length);
            const qint64 start = std::clamp(snapTime(ms - m_dragOffset),
                                            bounds.first, bounds.second);
            segment.startMs = start;
            segment.endMs = start + length;
        } else if (m_mode == Mode::ResizeLeft) {
            const qint64 lower = adjacentBounds(m_active).first;
            segment.startMs =
                    std::clamp(ms, std::min(lower, segment.endMs - minimum),
                               segment.endMs - minimum);
        } else if (m_mode == Mode::ResizeRight) {
            const qint64 upper = adjacentBounds(m_active).second;
            segment.endMs =
                    std::clamp(ms, segment.startMs + minimum,
                               std::max(upper, segment.startMs + minimum));
        }
        if (segmentEdited &&
            (segment.startMs != oldStart || segment.endMs != oldEnd))
            segmentEdited(m_active, segment.startMs, segment.endMs);
        update();
    }

    void mouseReleaseEvent(QMouseEvent* event) override {
        const QRect track = rect().adjusted(12, 30, -12, -12);
        if (m_mode == Mode::Create) {
            if (event->button() != Qt::LeftButton &&
                event->button() != Qt::RightButton)
                return;
            m_dragging = false;
            const qint64 first = snapTime(timeAt(track, m_pressX));
            const qint64 last = snapTime(timeAt(track, event->pos().x()));
            m_mode = Mode::Idle;
            const auto range = clampCreatedRange(first, last);
            if (range.second - range.first >= minSegmentMs() && rangeCreated)
                rangeCreated(range.first, range.second);
            update();
            return;
        }
        if (m_mode == Mode::Seek && event->button() == Qt::LeftButton) {
            m_mode = Mode::Idle;
            m_dragging = false;
            return;
        }
        if (!m_dragging || event->button() != Qt::LeftButton) return;
        m_dragging = false;
        const bool edited = m_mode == Mode::Move ||
                            m_mode == Mode::ResizeLeft ||
                            m_mode == Mode::ResizeRight;
        m_mode = Mode::Idle;
        if (edited && segmentEditFinished) segmentEditFinished();
        m_active = -1;
        update();
    }

private:
    enum class Mode { Idle, Seek, Create, Move, ResizeLeft, ResizeRight };
    enum class HitKind { None, Body, Left, Right };
    struct Hit {
        HitKind kind = HitKind::None;
        int index = -1;
    };

    int xAt(const QRect& track, qint64 ms) const {
        return track.left() +
               int(double(track.width()) *
                   std::clamp(
                           double(ms) / std::max<qint64>(1, m_data.durationMs),
                           0.0, 1.0));
    }
    qint64 timeAt(const QRect& track, int x) const {
        return qint64(double(m_data.durationMs) *
                      std::clamp(double(x - track.left()) /
                                         std::max(1, track.width()),
                                 0.0, 1.0));
    }
    qint64 snapTime(qint64 ms) const {
        if (m_data.fps <= 0)
            return std::clamp(ms, qint64(0), m_data.durationMs);
        const double frame = 1000.0 / m_data.fps;
        return std::clamp(
                qint64(std::llround(std::llround(ms / frame) * frame)),
                qint64(0), m_data.durationMs);
    }
    qint64 minSegmentMs() const {
        return m_data.fps > 0
                       ? std::max<qint64>(
                                 1, qint64(std::llround(1000.0 / m_data.fps)))
                       : 50;
    }
    Hit hitTest(const QRect& track, int x) const {
        for (int i = m_data.segments.size() - 1; i >= 0; --i) {
            const VideoSegment& segment = m_data.segments[i];
            const int left = xAt(track, segment.startMs);
            const int right = xAt(track, segment.endMs);
            if (i == m_selected && std::abs(x - left) <= 5)
                return {HitKind::Left, i};
            if (i == m_selected && std::abs(x - right) <= 5)
                return {HitKind::Right, i};
            if (x >= left && x <= std::max(left + 2, right))
                return {HitKind::Body, i};
        }
        return {};
    }
    std::pair<qint64, qint64> adjacentBounds(int index) const {
        qint64 lower = 0;
        qint64 upper = m_data.durationMs;
        if (index < 0 || index >= m_data.segments.size()) return {lower, upper};
        for (int i = 0; i < m_data.segments.size(); ++i) {
            if (i == index) continue;
            const VideoSegment& other = m_data.segments[i];
            if (other.endMs <= m_initialStart)
                lower = std::max(lower, other.endMs);
            if (other.startMs >= m_initialEnd)
                upper = std::min(upper, other.startMs);
        }
        return {lower, upper};
    }
    std::pair<qint64, qint64> moveBounds(int index, qint64 length) const {
        const auto bounds = adjacentBounds(index);
        return {bounds.first, std::max(bounds.first, bounds.second - length)};
    }
    std::pair<qint64, qint64> clampCreatedRange(qint64 first,
                                                qint64 last) const {
        const qint64 anchor = first;
        if (first > last) std::swap(first, last);
        for (int i = 0; i < m_data.segments.size(); ++i) {
            const VideoSegment& other = m_data.segments[i];
            if (anchor > other.startMs && anchor < other.endMs)
                return {anchor, anchor};
        }
        if (last == first) return {first, last};
        if (last > anchor) {
            qint64 limit = m_data.durationMs;
            for (const VideoSegment& other : m_data.segments)
                if (other.startMs >= anchor)
                    limit = std::min(limit, other.startMs);
            return {first, std::min(last, limit)};
        }
        qint64 limit = 0;
        for (const VideoSegment& other : m_data.segments)
            if (other.endMs <= anchor) limit = std::max(limit, other.endMs);
        return {std::max(first, limit), last};
    }
    void beginCreate(const QRect&, int x) {
        m_mode = Mode::Create;
        m_dragging = true;
        m_pressX = x;
        m_dragX = x;
    }
    VideoSidecar m_data;
    QVector<QImage> m_thumbnails;
    qint64 m_playhead = 0;
    int m_selected = -1;
    int m_pressX = 0;
    int m_dragX = 0;
    bool m_dragging = false;
    Mode m_mode = Mode::Idle;
    int m_active = -1;
    qint64 m_initialStart = 0;
    qint64 m_initialEnd = 0;
    qint64 m_dragOffset = 0;
};

class AnnotationVideoPlayer final : public VideoPlaybackWidget {
public:
    using VideoPlaybackWidget::VideoPlaybackWidget;
    std::function<void(int)> frameChanged;
    std::function<void(int, int)> sizeChanged;
    void hideSourceSelectors() {
        if (sourceCombo()) sourceCombo()->parentWidget()->hide();
        if (videoRow()) videoRow()->hide();
    }

protected:
#ifdef HAS_OPENCV_FACE_CAPTURE
    void onFrameDecoded(cv::Mat& frame, int) override {
        if (sizeChanged && !frame.empty()) sizeChanged(frame.cols, frame.rows);
    }
#endif
    void onDisplayFrame(QImage&, int index) override {
        if (frameChanged) frameChanged(index);
    }
};

VideoAnnotationDialog::VideoAnnotationDialog(QWidget* parent)
    : QDialog(parent) {
    setObjectName(QStringLiteral("qAnyLabelingVideoWorkspace"));
    setWindowTitle(tr("Video Classifier[*]"));
    resize(1280, 800);
    setMinimumSize(850, 560);
    setAcceptDrops(true);
    auto* root = new QVBoxLayout(this);
    auto* commands = new QHBoxLayout;
    auto* open = new QPushButton(
            QIcon(QStringLiteral(":/CC/plugin/qAnyLabeling/images/open.png")),
            tr("Open video"), this);
    auto* saveButton = new QPushButton(
            QIcon(QStringLiteral(":/CC/plugin/qAnyLabeling/images/save.png")),
            tr("Save"), this);
    m_undoButton = new QPushButton(
            QIcon(QStringLiteral(":/CC/plugin/qAnyLabeling/images/undo.svg")),
            QString(), this);
    m_undoButton->setObjectName(QStringLiteral("VideoUndo"));
    m_undoButton->setToolTip(tr("Undo segment edit (Ctrl+Z)"));
    m_redoButton = new QPushButton(
            QIcon(QStringLiteral(":/CC/plugin/qAnyLabeling/images/redo.svg")),
            QString(), this);
    m_redoButton->setObjectName(QStringLiteral("VideoRedo"));
    m_redoButton->setToolTip(tr("Redo segment edit (Ctrl+Shift+Z)"));
    m_info = new QLabel(tr("Open a video"), this);
    commands->addWidget(open);
    commands->addWidget(saveButton);
    commands->addWidget(m_undoButton);
    commands->addWidget(m_redoButton);
    commands->addWidget(m_info, 1);
    auto* shortcutsButton = new QPushButton(
            QIcon(QStringLiteral(
                    ":/CC/plugin/qAnyLabeling/images/shortcuts.svg")),
            QString(), this);
    shortcutsButton->setObjectName(QStringLiteral("VideoShortcuts"));
    shortcutsButton->setToolTip(tr("Show shortcuts"));
    commands->addWidget(shortcutsButton);
    connect(shortcutsButton, &QPushButton::clicked, this,
            [this, shortcutsButton]() {
                QMenu menu(this);
                const QStringList rows = {
                        tr("Space\tPlay / pause"),
                        tr("Left / Right\tPrevious / next frame"),
                        tr("Shift+Left / Right\tBack / forward 1 second"),
                        tr(", / .\tPrevious / next segment"),
                        tr("I / O\tMark in / out"),
                        tr("Enter\tCreate segment from marks"),
                        tr("Delete / Backspace\tDelete selected segment"),
                        tr("Ctrl+S\tSave annotations"),
                        tr("Ctrl+Z / Ctrl+Shift+Z\tUndo / redo"),
                        tr("0-9\tSelect label by number")};
                for (const QString& row : rows) menu.addAction(row);
                menu.exec(shortcutsButton->mapToGlobal(
                        shortcutsButton->rect().bottomRight()));
            });
    root->addLayout(commands);

    auto* splitter = new QSplitter(this);
    auto* media = new QWidget(splitter);
    auto* mediaLayout = new QVBoxLayout(media);
    auto* videoPlayer = new AnnotationVideoPlayer(media);
    m_player = videoPlayer;
    m_player->setAcceptDrops(true);
    m_player->installEventFilter(this);
    m_player->setInputSource(VideoPlaybackWidget::InputSource::VideoFile);
    videoPlayer->hideSourceSelectors();
    mediaLayout->addWidget(m_player, 1);
    m_timeline = new SegmentTimeline(media);
    m_timeline->setAcceptDrops(true);
    m_timeline->installEventFilter(this);
    m_timeline->setObjectName(QStringLiteral("VideoSegmentTimeline"));
    m_timeline->setMinimumHeight(100);
    mediaLayout->addWidget(m_timeline);
    auto* markRow = new QHBoxLayout;
    auto* markIn = new QPushButton(tr("Mark in"), media);
    auto* markOut = new QPushButton(tr("Mark out"), media);
    auto* add = new QPushButton(tr("Add segment"), media);
    auto* split = new QPushButton(
            QIcon(QStringLiteral(
                    ":/CC/plugin/qAnyLabeling/images/scissors.svg")),
            tr("Split at playhead"), media);
    split->setObjectName(QStringLiteral("VideoSplitSegment"));
    split->setToolTip(tr("Split the selected segment at the current frame"));
    auto* playPause = new QPushButton(
            QIcon(QStringLiteral(":/CC/plugin/qAnyLabeling/images/pause.svg")),
            tr("Pause"), media);
    m_activeLabel = new QComboBox(media);
    m_activeLabel->setEditable(false);
    markRow->addWidget(playPause);
    markRow->addWidget(markIn);
    markRow->addWidget(markOut);
    markRow->addWidget(m_activeLabel, 1);
    markRow->addWidget(add);
    markRow->addWidget(split);
    mediaLayout->addLayout(markRow);
    splitter->addWidget(media);

    auto* inspector = new QWidget(splitter);
    auto* inspectorLayout = new QVBoxLayout(inspector);
    auto* labelRow = new QHBoxLayout;
    labelRow->addWidget(new QLabel(tr("Labels"), inspector));
    auto* addLabel = new QPushButton(tr("+"), inspector);
    addLabel->setToolTip(tr("Add label"));
    auto* renameLabel = new QPushButton(tr("Rename"), inspector);
    auto* colorLabel = new QPushButton(
            QIcon(QStringLiteral(":/CC/plugin/qAnyLabeling/images/color.png")),
            QString(), inspector);
    colorLabel->setToolTip(tr("Label color"));
    auto* deleteLabel = new QPushButton(
            QIcon(QStringLiteral(":/CC/plugin/qAnyLabeling/images/delete.png")),
            QString(), inspector);
    deleteLabel->setToolTip(tr("Delete label and its segments"));
    labelRow->addStretch();
    labelRow->addWidget(addLabel);
    labelRow->addWidget(renameLabel);
    labelRow->addWidget(colorLabel);
    auto* settingsLabel = new QPushButton(
            QIcon(QStringLiteral(
                    ":/CC/plugin/qAnyLabeling/images/settings.svg")),
            QString(), inspector);
    settingsLabel->setObjectName(QStringLiteral("VideoLabelSettings"));
    settingsLabel->setToolTip(tr("Label settings and import"));
    labelRow->addWidget(settingsLabel);
    labelRow->addWidget(deleteLabel);
    inspectorLayout->addLayout(labelRow);
    m_labels = new QListWidget(inspector);
    m_labels->setObjectName(QStringLiteral("VideoLabels"));
    inspectorLayout->addWidget(m_labels, 1);
    auto* segmentRow = new QHBoxLayout;
    segmentRow->addWidget(new QLabel(tr("Segments"), inspector));
    auto* exportVideo = new QPushButton(
            QIcon(QStringLiteral(":/CC/plugin/qAnyLabeling/images/export.svg")),
            tr("Clip"), inspector);
    exportVideo->setObjectName(QStringLiteral("ExportVideoSegment"));
    exportVideo->setToolTip(tr("Export selected segment as a video clip"));
    auto* exportFrames = new QPushButton(
            QIcon(QStringLiteral(":/CC/plugin/qAnyLabeling/images/export.svg")),
            tr("Frames"), inspector);
    exportFrames->setObjectName(QStringLiteral("ExportVideoFrames"));
    exportFrames->setToolTip(tr("Export selected segment as raw frames"));
    auto* exportDataset = new QPushButton(
            QIcon(QStringLiteral(":/CC/plugin/qAnyLabeling/images/export.svg")),
            tr("Dataset"), inspector);
    exportDataset->setObjectName(QStringLiteral("ExportVideoDataset"));
    exportDataset->setToolTip(tr("Export all labeled segments as a dataset"));
    m_exportDataset = exportDataset;
    auto* remove = new QPushButton(
            QIcon(QStringLiteral(":/CC/plugin/qAnyLabeling/images/delete.png")),
            QString(), inspector);
    remove->setToolTip(tr("Delete selected segment"));
    segmentRow->addStretch();
    segmentRow->addWidget(remove);
    inspectorLayout->addLayout(segmentRow);
    m_segments = new QListWidget(inspector);
    m_segments->setObjectName(QStringLiteral("VideoSegments"));
    inspectorLayout->addWidget(m_segments, 2);
    auto* segmentExports = new QHBoxLayout;
    segmentExports->addWidget(exportVideo);
    segmentExports->addWidget(exportFrames);
    segmentExports->addWidget(exportDataset);
    inspectorLayout->addLayout(segmentExports);
    exportVideo->setEnabled(false);
    exportFrames->setEnabled(false);
    exportDataset->setEnabled(false);
    auto* range = new QHBoxLayout;
    m_start = new QDoubleSpinBox(inspector);
    m_end = new QDoubleSpinBox(inspector);
    for (QDoubleSpinBox* spin : {m_start, m_end}) {
        spin->setRange(0, 86400);
        spin->setDecimals(3);
        spin->setSuffix(tr(" s"));
        spin->setSingleStep(0.1);
    }
    range->addWidget(new QLabel(tr("In"), inspector));
    range->addWidget(m_start);
    range->addWidget(new QLabel(tr("Out"), inspector));
    range->addWidget(m_end);
    inspectorLayout->addLayout(range);
    m_description = new QPlainTextEdit(inspector);
    m_description->setPlaceholderText(tr("Segment description"));
    m_description->setMaximumHeight(84);
    inspectorLayout->addWidget(m_description);
    splitter->addWidget(inspector);
    splitter->setStretchFactor(0, 3);
    splitter->setStretchFactor(1, 1);
    root->addWidget(splitter, 1);

    connect(open, &QPushButton::clicked, this, [this]() {
        const QString path = QFileDialog::getOpenFileName(
                this, tr("Open video"), QFileInfo(m_videoPath).absolutePath(),
                tr("Videos (*.mp4 *.mov *.mkv *.avi *.webm *.m4v *.flv "
                   "*.wmv)"));
        if (!path.isEmpty()) openVideo(path);
    });
    connect(saveButton, &QPushButton::clicked, this,
            &VideoAnnotationDialog::save);
    connect(m_undoButton, &QPushButton::clicked, this,
            &VideoAnnotationDialog::undoLastEdit);
    connect(m_redoButton, &QPushButton::clicked, this,
            &VideoAnnotationDialog::redoLastEdit);
    connect(playPause, &QPushButton::clicked, this, [this, playPause]() {
        if (m_videoPath.isEmpty()) return;
        if (m_player->isActive())
            m_player->stopStream();
        else
            m_player->resumePlayback();
        const bool playing = m_player->isActive();
        playPause->setText(playing ? tr("Pause") : tr("Play"));
        playPause->setIcon(
                QIcon(QStringLiteral(":/CC/plugin/qAnyLabeling/images/") +
                      (playing ? QStringLiteral("pause.svg")
                               : QStringLiteral("play.svg"))));
    });
    connect(markIn, &QPushButton::clicked, this, [this]() {
        m_markStart = m_currentMs;
        m_info->setText(tr("In: %1").arg(formatTime(m_markStart)));
    });
    connect(markOut, &QPushButton::clicked, this, [this]() {
        m_markEnd = m_currentMs;
        m_info->setText(tr("Out: %1").arg(formatTime(m_markEnd)));
    });
    connect(add, &QPushButton::clicked, this, [this]() {
        addSegment(m_markStart >= 0 ? m_markStart : m_currentMs,
                   m_markEnd >= 0 ? m_markEnd : m_currentMs);
    });
    connect(split, &QPushButton::clicked, this,
            &VideoAnnotationDialog::splitSelectedSegment);
    connect(addLabel, &QPushButton::clicked, this, [this]() {
        bool ok = false;
        const QString label =
                QInputDialog::getText(this, tr("Add label"), tr("Name"),
                                      QLineEdit::Normal, {}, &ok)
                        .trimmed();
        if (!ok || label.isEmpty() || m_data.labels.contains(label)) return;
        pushUndoSnapshot();
        m_data.labels.append(label);
        m_data.labelColors.insert(label, colorFor(m_data.labels.size() - 1));
        refreshLabels();
        m_activeLabel->setCurrentText(label);
        setDirty();
    });
    connect(renameLabel, &QPushButton::clicked, this, [this]() {
        const int row = m_labels->currentRow();
        if (row < 0) return;
        const QString old = m_data.labels[row];
        bool ok = false;
        const QString label =
                QInputDialog::getText(this, tr("Rename label"), tr("Name"),
                                      QLineEdit::Normal, old, &ok)
                        .trimmed();
        if (!ok || label.isEmpty() ||
            (label != old && m_data.labels.contains(label)))
            return;
        pushUndoSnapshot();
        m_data.labels[row] = label;
        m_data.labelColors.insert(label, m_data.labelColors.take(old));
        for (VideoSegment& segment : m_data.segments)
            if (segment.label == old) segment.label = label;
        refreshLabels();
        refreshSegments();
        setDirty();
    });
    connect(colorLabel, &QPushButton::clicked, this, [this]() {
        const int row = m_labels->currentRow();
        if (row < 0) return;
        const QString label = m_data.labels[row];
        const QColor color = QColorDialog::getColor(
                QColor(m_data.labelColors.value(label).toString()), this,
                tr("Label color"));
        if (!color.isValid()) return;
        pushUndoSnapshot();
        m_data.labelColors.insert(label, color.name());
        refreshLabels();
        refreshTimeline();
        setDirty();
    });
    connect(deleteLabel, &QPushButton::clicked, this, [this]() {
        const int row = m_labels->currentRow();
        if (row < 0) return;
        const QString label = m_data.labels[row];
        if (QMessageBox::question(
                    this, tr("Delete label"),
                    tr("Delete %1 and all its segments?").arg(label)) !=
            QMessageBox::Yes)
            return;
        pushUndoSnapshot();
        m_data.labels.removeAt(row);
        m_data.labelColors.remove(label);
        auto& segments = m_data.segments;
        segments.erase(std::remove_if(segments.begin(), segments.end(),
                                      [&label](const VideoSegment& segment) {
                                          return segment.label == label;
                                      }),
                       segments.end());
        refreshLabels();
        refreshSegments();
        setDirty();
    });
    connect(settingsLabel, &QPushButton::clicked, this, [this]() {
        QStringList labels;
        QJsonObject colors;
        QStringList deleted;
        QHash<QString, QString> renamed;
        if (!editVideoLabels(this, m_data.labels, m_data.labelColors, &labels,
                             &colors, &deleted, &renamed))
            return;
        if (labels == m_data.labels && colors == m_data.labelColors &&
            deleted.isEmpty() && renamed.isEmpty())
            return;
        pushUndoSnapshot();
        for (const QString& old : deleted) {
            m_data.segments.erase(
                    std::remove_if(m_data.segments.begin(),
                                   m_data.segments.end(),
                                   [&old](const VideoSegment& segment) {
                                       return segment.label == old;
                                   }),
                    m_data.segments.end());
        }
        for (VideoSegment& segment : m_data.segments)
            if (renamed.contains(segment.label))
                segment.label = renamed.value(segment.label);
        m_data.labels = labels;
        m_data.labelColors = colors;
        refreshLabels();
        refreshSegments();
        setDirty();
    });
    connect(m_labels, &QListWidget::currentRowChanged, this, [this](int row) {
        if (row >= 0 && row < m_data.labels.size())
            m_activeLabel->setCurrentText(m_data.labels[row]);
    });
    connect(remove, &QPushButton::clicked, this, [this]() {
        const int row = m_segments->currentRow();
        if (row < 0 || row >= m_data.segments.size()) return;
        pushUndoSnapshot();
        m_data.segments.removeAt(row);
        refreshSegments();
        setDirty();
    });
    connect(exportVideo, &QPushButton::clicked, this, [this]() {
        const int row = m_segments->currentRow();
        if (row < 0 || row >= m_data.segments.size() || m_videoPath.isEmpty())
            return;
        const VideoSegment& segment = m_data.segments[row];
        const QString stem = QFileInfo(m_videoPath).completeBaseName();
        const QString path = QFileDialog::getSaveFileName(
                this, tr("Export video clip"),
                QFileInfo(m_videoPath)
                        .absoluteDir()
                        .filePath(QStringLiteral("%1_%2.avi")
                                          .arg(safeSegmentToken(stem),
                                               safeSegmentToken(segment.id))),
                tr("Video clips (*.avi *.mp4)"));
        if (path.isEmpty()) return;
        QString error;
        if (!exportSelectedSegmentVideo(path, &error))
            QMessageBox::warning(this, tr("Export video clip"), error);
        else
            m_info->setText(tr("Exported %1").arg(QFileInfo(path).fileName()));
    });
    connect(exportFrames, &QPushButton::clicked, this, [this]() {
        const int row = m_segments->currentRow();
        if (row < 0 || row >= m_data.segments.size() || m_videoPath.isEmpty())
            return;
        const VideoSegment& segment = m_data.segments[row];
        const QString root = QFileDialog::getExistingDirectory(
                this, tr("Export raw frames"),
                QFileInfo(m_videoPath).absolutePath());
        if (root.isEmpty()) return;
        const QString path = QDir(root).filePath(QStringLiteral("%1_%2").arg(
                safeSegmentToken(QFileInfo(m_videoPath).completeBaseName()),
                safeSegmentToken(segment.id)));
        QString error;
        if (!exportSelectedSegmentFrames(path, &error))
            QMessageBox::warning(this, tr("Export raw frames"), error);
        else
            m_info->setText(tr("Exported frames to %1").arg(path));
    });
    connect(exportDataset, &QPushButton::clicked, this, [this]() {
        if (m_videoPath.isEmpty()) return;
        QDialog settings(this);
        settings.setWindowTitle(tr("Export video dataset"));
        settings.resize(560, 240);
        auto* layout = new QVBoxLayout(&settings);
        auto* outputRow = new QHBoxLayout;
        outputRow->addWidget(new QLabel(tr("Output folder"), &settings));
        auto* output = new QLineEdit(
                QFileInfo(m_videoPath)
                        .absoluteDir()
                        .filePath(
                                safeSegmentToken(QFileInfo(m_videoPath)
                                                         .completeBaseName()) +
                                QStringLiteral("_dataset")),
                &settings);
        output->setObjectName(QStringLiteral("VideoDatasetOutput"));
        outputRow->addWidget(output, 1);
        auto* browse = new QPushButton(tr("Browse..."), &settings);
        outputRow->addWidget(browse);
        layout->addLayout(outputRow);
        connect(browse, &QPushButton::clicked, &settings, [&, output]() {
            const QString parent = QFileDialog::getExistingDirectory(
                    &settings, tr("Choose dataset parent folder"),
                    QFileInfo(output->text()).absolutePath());
            if (!parent.isEmpty())
                output->setText(QDir(parent).filePath(
                        QFileInfo(output->text()).fileName()));
        });
        auto* clips = new QCheckBox(tr("Export video clips"), &settings);
        auto* frames = new QCheckBox(tr("Export raw frames"), &settings);
        auto* reEncode = new QCheckBox(
                tr("Re-encode clips for frame-accurate cuts"), &settings);
        reEncode->setToolTip(
                tr("When off, copy the video stream quickly around keyframes; "
                   "re-encode automatically if copying fails."));
        auto* zip = new QCheckBox(tr("Create ZIP archive"), &settings);
        clips->setChecked(true);
        zip->setChecked(true);
        auto* fpsRow = new QHBoxLayout;
        fpsRow->addWidget(
                new QLabel(tr("Raw frame FPS (0 = source)"), &settings));
        auto* fps = new QSpinBox(&settings);
        fps->setRange(0, 240);
        fps->setValue(0);
        fps->setEnabled(false);
        fpsRow->addWidget(fps);
        connect(frames, &QCheckBox::toggled, fps, &QSpinBox::setEnabled);
        connect(clips, &QCheckBox::toggled, reEncode, &QCheckBox::setEnabled);
        layout->addWidget(clips);
        layout->addWidget(frames);
        layout->addLayout(fpsRow);
        layout->addWidget(reEncode);
        layout->addWidget(zip);
        auto* buttons = new QDialogButtonBox(
                QDialogButtonBox::Ok | QDialogButtonBox::Cancel, &settings);
        layout->addWidget(buttons);
        connect(buttons, &QDialogButtonBox::accepted, &settings, [&]() {
            if (output->text().trimmed().isEmpty() ||
                (!clips->isChecked() && !frames->isChecked())) {
                QMessageBox::warning(
                        &settings, tr("Export video dataset"),
                        tr("Choose an output folder and a format."));
                return;
            }
            settings.accept();
        });
        connect(buttons, &QDialogButtonBox::rejected, &settings,
                &QDialog::reject);
        if (settings.exec() != QDialog::Accepted) return;
        VideoDatasetExportOptions options;
        options.videoPath = m_videoPath;
        options.outputDirectory =
                QFileInfo(output->text().trimmed()).absoluteFilePath();
        options.includeVideo = clips->isChecked();
        options.includeRawFrames = frames->isChecked();
        options.rawFrameFps = fps->value();
        options.zipOutput = zip->isChecked();
        options.reEncodeVideo = reEncode->isChecked();
        QProgressDialog progress(tr("Exporting video dataset"), tr("Cancel"), 0,
                                 1, this);
        progress.setWindowModality(Qt::WindowModal);
        progress.setAutoClose(false);
        QString error;
        QString archive;
        const bool exported = exportVideoDataset(
                options,
                [&progress](int current, int total) {
                    progress.setRange(0, std::max(1, total));
                    progress.setValue(current);
                    QCoreApplication::processEvents();
                    return !progress.wasCanceled();
                },
                &error, &archive, &m_data);
        progress.close();
        if (!exported) {
            QMessageBox::warning(this, tr("Export video dataset"), error);
            return;
        }
        m_info->setText(archive.isEmpty()
                                ? tr("Dataset exported to %1")
                                          .arg(options.outputDirectory)
                                : tr("Dataset and ZIP exported"));
    });
    connect(m_segments, &QListWidget::currentRowChanged, this,
            &VideoAnnotationDialog::selectSegment);
    connect(m_segments, &QListWidget::currentRowChanged, this,
            [this, exportVideo, exportFrames](int row) {
                const bool enabled = !m_videoPath.isEmpty() && row >= 0 &&
                                     row < m_data.segments.size();
                exportVideo->setEnabled(enabled);
                exportFrames->setEnabled(enabled);
            });
    auto editRange = [this]() {
        const int row = m_segments->currentRow();
        if (m_updating || row < 0 || row >= m_data.segments.size()) return;
        VideoSegment& segment = m_data.segments[row];
        const qint64 start = qint64(std::llround(m_start->value() * 1000));
        const qint64 end = qint64(std::llround(m_end->value() * 1000));
        if (end <= start || (m_data.durationMs > 0 && end > m_data.durationMs))
            return;
        for (int i = 0; i < m_data.segments.size(); ++i) {
            if (i == row) continue;
            const VideoSegment& other = m_data.segments[i];
            if (start < other.endMs && end > other.startMs) return;
        }
        if (start == segment.startMs && end == segment.endMs) return;
        if (m_rangeUndoId != segment.id) {
            pushUndoSnapshot();
            m_rangeUndoId = segment.id;
        }
        segment.startMs = start;
        segment.endMs = end;
        segment.startFrame = videoFrameAt(start, m_data.fps);
        segment.endFrame = videoFrameAt(end, m_data.fps);
        refreshSegments();
        m_segments->setCurrentRow(row);
        setDirty();
    };
    connect(m_start, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
            editRange);
    connect(m_end, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
            editRange);
    connect(m_start, &QDoubleSpinBox::editingFinished, this,
            [this]() { m_rangeUndoId.clear(); });
    connect(m_end, &QDoubleSpinBox::editingFinished, this,
            [this]() { m_rangeUndoId.clear(); });
    connect(m_description, &QPlainTextEdit::textChanged, this, [this]() {
        const int row = m_segments->currentRow();
        if (m_updating || row < 0 || row >= m_data.segments.size()) return;
        if (m_descriptionUndoId != m_data.segments[row].id) {
            pushUndoSnapshot();
            m_descriptionUndoId = m_data.segments[row].id;
        }
        m_data.segments[row].description = m_description->toPlainText();
        setDirty();
    });
    videoPlayer->frameChanged = [this](int frame) {
        m_currentMs =
                m_data.fps > 0
                        ? qint64(std::llround(1000.0 * frame / m_data.fps))
                        : 0;
        refreshTimeline();
    };
    videoPlayer->sizeChanged = [this](int width, int height) {
        if (m_data.width != width || m_data.height != height) {
            m_data.width = width;
            m_data.height = height;
        }
    };
    m_timeline->seekRequested = [this](qint64 ms) { seek(ms); };
    m_timeline->rangeCreated = [this](qint64 start, qint64 end) {
        addSegment(start, end);
    };
    m_timeline->segmentSelected = [this](int row) {
        m_segments->setCurrentRow(row);
    };
    m_timeline->segmentEdited = [this](int row, qint64 start, qint64 end) {
        if (row < 0 || row >= m_data.segments.size()) return;
        if (!m_timelineUndoActive) {
            pushUndoSnapshot();
            m_timelineUndoActive = true;
        }
        VideoSegment& segment = m_data.segments[row];
        segment.startMs = start;
        segment.endMs = end;
        segment.startFrame = videoFrameAt(start, m_data.fps);
        segment.endFrame = videoFrameAt(end, m_data.fps);
        setDirty();
    };
    m_timeline->segmentEditFinished = [this]() {
        m_timelineUndoActive = false;
        refreshSegments();
    };
    addShortcut(this, QKeySequence::Save, [this]() { save(); });
    addShortcut(this, QKeySequence(Qt::Key_I), [markIn]() { markIn->click(); });
    addShortcut(this, QKeySequence(Qt::Key_O),
                [markOut]() { markOut->click(); });
    addShortcut(this, QKeySequence(Qt::Key_N), [add]() { add->click(); });
    addShortcut(this, QKeySequence(Qt::Key_Space),
                [playPause]() { playPause->click(); });
    addShortcut(this, QKeySequence(Qt::Key_Delete), [remove]() {
        if (remove->isEnabled()) remove->click();
    });
    addShortcut(this, QKeySequence(Qt::Key_Backspace), [remove]() {
        if (remove->isEnabled()) remove->click();
    });
    addShortcut(this, QKeySequence(Qt::Key_Return), [add]() { add->click(); });
    addShortcut(this, QKeySequence(Qt::Key_Enter), [add]() { add->click(); });
    addShortcut(this, QKeySequence(Qt::CTRL | Qt::Key_Z),
                [this]() { undoLastEdit(); });
    addShortcut(this, QKeySequence(Qt::CTRL | Qt::SHIFT | Qt::Key_Z),
                [this]() { redoLastEdit(); });
    addShortcut(this, QKeySequence(Qt::Key_Right), [this]() {
        if (m_player->totalVideoFrames() <= 0) return;
        m_player->seekToFrame(
                std::min(m_player->totalVideoFrames() - 1,
                         videoFrameAt(m_currentMs, m_data.fps) + 1));
    });
    addShortcut(this, QKeySequence(Qt::Key_Left), [this]() {
        if (m_player->totalVideoFrames() <= 0) return;
        m_player->seekToFrame(
                std::max(0, videoFrameAt(m_currentMs, m_data.fps) - 1));
    });
    addShortcut(this, QKeySequence(Qt::SHIFT | Qt::Key_Right),
                [this]() { seek(m_currentMs + 1000); });
    addShortcut(this, QKeySequence(Qt::SHIFT | Qt::Key_Left),
                [this]() { seek(std::max<qint64>(0, m_currentMs - 1000)); });
    addShortcut(this, QKeySequence(Qt::Key_Comma), [this]() {
        const int row = m_segments->currentRow();
        if (row > 0) m_segments->setCurrentRow(row - 1);
    });
    addShortcut(this, QKeySequence(Qt::Key_Period), [this]() {
        const int row = m_segments->currentRow();
        if (row + 1 < m_segments->count()) m_segments->setCurrentRow(row + 1);
    });
    for (int digit = 0; digit <= 9; ++digit) {
        addShortcut(this, QKeySequence(Qt::Key_0 + digit), [this, digit]() {
            if (digit < m_labels->count()) m_labels->setCurrentRow(digit);
        });
    }
    refreshLabels();
    refreshSegments();
    selectSegment(-1);
    refreshHistoryActions();
}

bool VideoAnnotationDialog::openVideo(const QString& path) {
    if (!confirmSave()) return false;
    VideoSidecar loaded;
    QString error;
    if (!loadVideoSidecar(path, &loaded, &error)) {
        QMessageBox::warning(
                this, tr("Open video"),
                tr("Unable to load %1: %2").arg(videoSidecarPath(path), error));
        return false;
    }
    if (!m_player->startVideoFile(path)) {
        QMessageBox::warning(this, tr("Open video"),
                             tr("Unable to decode %1").arg(path));
        return false;
    }
    m_videoPath = path;
    m_data = loaded;
    m_data.video = QFileInfo(path).fileName();
    m_data.fps = m_player->videoFps();
    m_data.durationMs =
            m_player->totalVideoFrames() > 0
                    ? qint64(std::llround(1000.0 *
                                          m_player->totalVideoFrames() /
                                          m_data.fps))
                    : loaded.durationMs;
    m_currentMs = 0;
    m_markStart = -1;
    m_markEnd = -1;
    m_dirty = false;
    m_undoStack.clear();
    m_redoStack.clear();
    m_descriptionUndoId.clear();
    m_rangeUndoId.clear();
    m_timelineUndoActive = false;
    setWindowTitle(tr("Video Classifier - %1[*]").arg(m_data.video));
    m_info->setText(m_data.video);
    const quint64 thumbnailRequest = ++m_thumbnailRequest;
    m_timeline->setThumbnails({});
    auto* thumbnailWatcher = new QFutureWatcher<QVector<QImage>>(this);
    connect(thumbnailWatcher, &QFutureWatcher<QVector<QImage>>::finished, this,
            [this, thumbnailWatcher, thumbnailRequest, path]() {
                if (thumbnailRequest == m_thumbnailRequest &&
                    path == m_videoPath)
                    m_timeline->setThumbnails(thumbnailWatcher->result());
                thumbnailWatcher->deleteLater();
            });
    thumbnailWatcher->setFuture(
            QtConcurrent::run([path]() { return videoThumbnails(path); }));
    refreshLabels();
    refreshSegments();
    selectSegment(-1);
    refreshHistoryActions();
    return true;
}

void VideoAnnotationDialog::dragEnterEvent(QDragEnterEvent* event) {
    if (acceptsVideoDrop(event->mimeData()))
        event->acceptProposedAction();
    else
        event->ignore();
}

void VideoAnnotationDialog::dragMoveEvent(QDragMoveEvent* event) {
    if (acceptsVideoDrop(event->mimeData()))
        event->acceptProposedAction();
    else
        event->ignore();
}

void VideoAnnotationDialog::dropEvent(QDropEvent* event) {
    for (const QUrl& url : event->mimeData()->urls()) {
        const QFileInfo info(url.toLocalFile());
        if (info.isFile() &&
            kVideoExtensions.contains(info.suffix().toLower())) {
            if (openVideo(info.absoluteFilePath()))
                event->acceptProposedAction();
            return;
        }
    }
    event->ignore();
}

bool VideoAnnotationDialog::eventFilter(QObject* watched, QEvent* event) {
    if ((watched == m_player || watched == m_timeline) && event) {
        if (event->type() == QEvent::DragEnter) {
            dragEnterEvent(static_cast<QDragEnterEvent*>(event));
            return event->isAccepted();
        }
        if (event->type() == QEvent::DragMove) {
            dragMoveEvent(static_cast<QDragMoveEvent*>(event));
            return event->isAccepted();
        }
        if (event->type() == QEvent::Drop) {
            dropEvent(static_cast<QDropEvent*>(event));
            return event->isAccepted();
        }
    }
    return QDialog::eventFilter(watched, event);
}

bool VideoAnnotationDialog::exportSelectedSegmentVideo(
        const QString& outputPath, QString* error) {
    return exportSelectedSegment(VideoSegmentExportFormat::Video, outputPath,
                                 error);
}

bool VideoAnnotationDialog::exportSelectedSegmentFrames(
        const QString& outputDirectory, QString* error) {
    return exportSelectedSegment(VideoSegmentExportFormat::RawFrames,
                                 outputDirectory, error);
}

bool VideoAnnotationDialog::exportSelectedSegment(
        VideoSegmentExportFormat format,
        const QString& outputPath,
        QString* error) {
    const int row = m_segments ? m_segments->currentRow() : -1;
    if (m_videoPath.isEmpty() || row < 0 || row >= m_data.segments.size()) {
        if (error) *error = tr("Select a video segment first");
        return false;
    }
    const VideoSegment segment = m_data.segments[row];
    QProgressDialog progress(format == VideoSegmentExportFormat::Video
                                     ? tr("Exporting video")
                                     : tr("Exporting frames"),
                             tr("Cancel"), 0, 100, this);
    progress.setWindowModality(Qt::WindowModal);
    progress.setAutoClose(true);
    QString localError;
    const bool ok = exportVideoSegment(
            {m_videoPath, outputPath, segment.startMs, segment.endMs,
             m_data.fps, format},
            [&progress](int current, int total) {
                progress.setValue(total > 0 ? current * 100 / total : 0);
                QCoreApplication::processEvents();
                return !progress.wasCanceled();
            },
            &localError);
    if (!ok && error) *error = localError;
    if (ok && error) error->clear();
    return ok;
}

void VideoAnnotationDialog::refreshLabels() {
    const QString selected = m_activeLabel->currentText();
    m_labels->clear();
    m_activeLabel->clear();
    for (const QString& label : m_data.labels) {
        auto* item = new QListWidgetItem(label, m_labels);
        item->setForeground(
                QColor(m_data.labelColors.value(label).toString("#118AB2")));
        m_activeLabel->addItem(label);
    }
    m_activeLabel->setCurrentText(selected);
    if (m_activeLabel->currentIndex() < 0 && m_activeLabel->count())
        m_activeLabel->setCurrentIndex(0);
}

void VideoAnnotationDialog::refreshSegments() {
    const int selected = m_segments->currentRow();
    m_segments->blockSignals(true);
    m_segments->clear();
    for (const VideoSegment& segment : m_data.segments) {
        auto* item = new QListWidgetItem(
                QStringLiteral("%1  %2 - %3")
                        .arg(segment.label, formatTime(segment.startMs),
                             formatTime(segment.endMs)),
                m_segments);
        item->setForeground(QColor(
                m_data.labelColors.value(segment.label).toString("#118AB2")));
    }
    m_segments->blockSignals(false);
    if (selected >= 0 && selected < m_segments->count())
        m_segments->setCurrentRow(selected);
    else
        selectSegment(-1);
    if (m_exportDataset)
        m_exportDataset->setEnabled(
                !m_videoPath.isEmpty() &&
                std::any_of(m_data.segments.cbegin(), m_data.segments.cend(),
                            [](const VideoSegment& segment) {
                                return !segment.label.trimmed().isEmpty() &&
                                       segment.endMs > segment.startMs;
                            }));
    refreshTimeline();
}

void VideoAnnotationDialog::refreshTimeline() {
    m_timeline->setData(m_data, m_currentMs, m_segments->currentRow());
}

void VideoAnnotationDialog::selectSegment(int row) {
    m_descriptionUndoId.clear();
    m_rangeUndoId.clear();
    m_updating = true;
    const bool valid = row >= 0 && row < m_data.segments.size();
    m_start->setEnabled(valid);
    m_end->setEnabled(valid);
    m_description->setEnabled(valid);
    if (valid) {
        const VideoSegment& segment = m_data.segments[row];
        m_start->setValue(segment.startMs / 1000.0);
        m_end->setValue(segment.endMs / 1000.0);
        m_description->setPlainText(segment.description);
    } else {
        m_start->setValue(0);
        m_end->setValue(0);
        m_description->clear();
    }
    m_updating = false;
    refreshTimeline();
}

void VideoAnnotationDialog::seek(qint64 ms) {
    if (m_data.fps <= 0) return;
    const qint64 clamped =
            std::clamp(ms, qint64(0), std::max<qint64>(0, m_data.durationMs));
    const int totalFrames = m_player->totalVideoFrames();
    const int lastFrame = std::max(0, totalFrames - 1);
    const int frame = totalFrames > 0
                              ? std::clamp(videoFrameAt(clamped, m_data.fps), 0,
                                           lastFrame)
                              : videoFrameAt(clamped, m_data.fps);
    m_currentMs = qint64(std::llround(1000.0 * frame / m_data.fps));
    m_player->seekToFrame(frame);
    refreshTimeline();
}

void VideoAnnotationDialog::addSegment(qint64 startMs, qint64 endMs) {
    const QString label = m_activeLabel->currentText();
    if (m_videoPath.isEmpty() || label.isEmpty()) return;
    if (endMs < startMs) std::swap(startMs, endMs);
    for (const VideoSegment& other : m_data.segments) {
        if (startMs >= other.startMs && startMs < other.endMs) return;
        if (other.startMs >= startMs) endMs = std::min(endMs, other.startMs);
    }
    if (endMs - startMs < 50) {
        m_info->setText(tr("Select at least 50 ms on the timeline"));
        return;
    }
    VideoSegment segment;
    segment.id = QStringLiteral("s") + QUuid::createUuid()
                                               .toString(QUuid::WithoutBraces)
                                               .remove(QLatin1Char('-'))
                                               .left(10);
    segment.label = label;
    segment.startMs = startMs;
    segment.endMs =
            m_data.durationMs > 0 ? std::min(endMs, m_data.durationMs) : endMs;
    if (segment.endMs - segment.startMs < 50) return;
    segment.startFrame = videoFrameAt(segment.startMs, m_data.fps);
    segment.endFrame = videoFrameAt(segment.endMs, m_data.fps);
    pushUndoSnapshot();
    m_data.segments.append(segment);
    refreshSegments();
    m_segments->setCurrentRow(m_data.segments.size() - 1);
    m_markStart = m_markEnd = -1;
    setDirty();
}

void VideoAnnotationDialog::splitSelectedSegment() {
    if (m_videoPath.isEmpty() || m_data.fps <= 0) return;

    const int selectedRow = m_segments->currentRow();
    int splitRow = -1;
    if (selectedRow >= 0 && selectedRow < m_data.segments.size()) {
        const VideoSegment& selected = m_data.segments[selectedRow];
        if (m_currentMs > selected.startMs && m_currentMs < selected.endMs)
            splitRow = selectedRow;
    }
    if (splitRow < 0) {
        for (int i = 0; i < m_data.segments.size(); ++i) {
            const VideoSegment& segment = m_data.segments[i];
            if (m_currentMs > segment.startMs && m_currentMs < segment.endMs) {
                splitRow = i;
                break;
            }
        }
    }
    if (splitRow < 0) {
        m_info->setText(tr("Place the playhead inside a segment to split it"));
        return;
    }

    const VideoSegment original = m_data.segments[splitRow];
    const int splitFrame = videoFrameAt(m_currentMs, m_data.fps);
    const qint64 splitMs =
            qint64(std::llround(1000.0 * splitFrame / m_data.fps));
    if (splitMs <= original.startMs || splitMs >= original.endMs) {
        m_info->setText(tr("The playhead must be between segment boundaries"));
        return;
    }

    VideoSegment first = original;
    first.id = QStringLiteral("s") + QUuid::createUuid()
                                             .toString(QUuid::WithoutBraces)
                                             .remove(QLatin1Char('-'))
                                             .left(10);
    first.endMs = splitMs;
    first.startFrame = videoFrameAt(first.startMs, m_data.fps);
    first.endFrame = videoFrameAt(first.endMs, m_data.fps);
    VideoSegment second = original;
    second.id = QStringLiteral("s") + QUuid::createUuid()
                                              .toString(QUuid::WithoutBraces)
                                              .remove(QLatin1Char('-'))
                                              .left(10);
    second.startMs = splitMs;
    second.startFrame = videoFrameAt(second.startMs, m_data.fps);
    second.endFrame = videoFrameAt(second.endMs, m_data.fps);

    pushUndoSnapshot();
    m_data.segments[splitRow] = first;
    m_data.segments.insert(splitRow + 1, second);
    refreshSegments();
    m_segments->setCurrentRow(splitRow + 1);
    setDirty();
    m_info->setText(tr("Segment split at %1").arg(formatTime(splitMs)));
}

void VideoAnnotationDialog::pushUndoSnapshot() {
    m_undoStack.append({m_data, m_segments->currentRow()});
    if (m_undoStack.size() > 100) m_undoStack.removeFirst();
    m_redoStack.clear();
    refreshHistoryActions();
}

void VideoAnnotationDialog::restoreSnapshot(const EditSnapshot& snapshot) {
    m_data = snapshot.data;
    m_descriptionUndoId.clear();
    m_rangeUndoId.clear();
    m_timelineUndoActive = false;
    refreshLabels();
    refreshSegments();
    m_segments->setCurrentRow(snapshot.selectedRow);
    selectSegment(snapshot.selectedRow);
    setDirty();
    refreshHistoryActions();
}

void VideoAnnotationDialog::undoLastEdit() {
    if (m_undoStack.isEmpty()) return;
    m_redoStack.append({m_data, m_segments->currentRow()});
    const EditSnapshot snapshot = m_undoStack.takeLast();
    restoreSnapshot(snapshot);
}

void VideoAnnotationDialog::redoLastEdit() {
    if (m_redoStack.isEmpty()) return;
    m_undoStack.append({m_data, m_segments->currentRow()});
    const EditSnapshot snapshot = m_redoStack.takeLast();
    restoreSnapshot(snapshot);
}

void VideoAnnotationDialog::refreshHistoryActions() {
    if (m_undoButton) m_undoButton->setEnabled(!m_undoStack.isEmpty());
    if (m_redoButton) m_redoButton->setEnabled(!m_redoStack.isEmpty());
}

void VideoAnnotationDialog::setDirty() {
    m_dirty = true;
    setWindowModified(true);
}

bool VideoAnnotationDialog::save() {
    if (m_videoPath.isEmpty()) return false;
    QString error;
    if (!saveVideoSidecar(m_videoPath, m_data, &error)) {
        QMessageBox::warning(this, tr("Save video annotations"), error);
        return false;
    }
    m_dirty = false;
    setWindowModified(false);
    m_info->setText(
            tr("Saved %1")
                    .arg(QFileInfo(videoSidecarPath(m_videoPath)).fileName()));
    return true;
}

bool VideoAnnotationDialog::confirmSave() {
    if (!m_dirty) return true;
    const QMessageBox::StandardButton answer = QMessageBox::question(
            this, tr("Unsaved video annotations"),
            tr("Save changes before closing?"),
            QMessageBox::Save | QMessageBox::Discard | QMessageBox::Cancel);
    if (answer == QMessageBox::Cancel) return false;
    return answer == QMessageBox::Discard || save();
}

void VideoAnnotationDialog::closeEvent(QCloseEvent* event) {
    if (confirmSave())
        event->accept();
    else
        event->ignore();
}
