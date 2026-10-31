// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "PointCloudCameraPanel.h"

#include <QButtonGroup>
#include <QCoreApplication>
#include <QDir>
#include <QFileInfo>
#include <QGraphicsItem>
#include <QGraphicsScene>
#include <QGraphicsView>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QHideEvent>
#include <QLabel>
#include <QMenu>
#include <QMouseEvent>
#include <QPaintEvent>
#include <QPainter>
#include <QPainterPath>
#include <QPolygonF>
#include <QRadioButton>
#include <QResizeEvent>
#include <QSettings>
#include <QSlider>
#include <QStyle>
#include <QToolButton>
#include <QVBoxLayout>
#include <QWheelEvent>
#include <QWidgetAction>
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>

#include "QtCompat.h"

namespace {
// Separate from the application's other plugins, using the same source keys
// within qAnyLabeling's settings namespace.
const QString cameraSettings =
        QStringLiteral("qAnyLabeling/pointcloud/camera_overlay/");

class ProjectionSettingsMenu final : public QMenu {
public:
    explicit ProjectionSettingsMenu(QWidget* parent) : QMenu(parent) {
        setWindowFlag(Qt::FramelessWindowHint);
        setAttribute(Qt::WA_TranslucentBackground);
        const bool dark = palette().color(QPalette::Window).lightness() < 128;
        const QString text = dark ? "#f5f5f7" : "#1d1d1f";
        const QString surface = dark ? "#2c2c2e" : "#f5f5f7";
        const QString secondary = dark ? "#aeaeb2" : "#86868b";
        const QString primary = dark ? "#0A84FF" : "#0071e3";
        setStyleSheet(
                QStringLiteral(
                        "QMenu { background: transparent; border: none; "
                        "padding: 4px; } "
                        "QLabel, QRadioButton { color: %1; background: "
                        "transparent; } "
                        "QRadioButton { spacing: 6px; } "
                        "QRadioButton::indicator { width: 14px; height: 14px; "
                        "border: 1px solid %2; border-radius: 8px; background: "
                        "%3; } "
                        "QRadioButton::indicator:checked { border-color: %4; "
                        "background: qradialgradient(cx:0.5, cy:0.5, "
                        "radius:0.5, "
                        "fx:0.5, fy:0.5, stop:0 %4, stop:0.5 %4, "
                        "stop:0.6 %3, stop:1 %3); } "
                        "QRadioButton::indicator:hover { border-color: %4; }")
                        .arg(text, secondary, surface, primary));
    }

protected:
    void paintEvent(QPaintEvent*) override {
        const bool dark = palette().color(QPalette::Window).lightness() < 128;
        QPainter painter(this);
        painter.setRenderHint(QPainter::Antialiasing);
        painter.setPen(QPen(QColor(dark ? "#3a3a3c" : "#E5E5E5"), 1));
        painter.setBrush(QColor(dark ? "#2c2c2e" : "#f5f5f7"));
        painter.drawRoundedRect(QRectF(rect()).adjusted(.5, .5, -.5, -.5), 12,
                                12);
    }
};

// Same QGraphicsView contract as upstream CameraView: hand drag,
// cursor-anchored 1.2x zoom, absolute scale bounds, and double-click fit.
class CameraImageView final : public QGraphicsView {
public:
    explicit CameraImageView(QWidget* parent) : QGraphicsView(parent) {
        setScene(new QGraphicsScene(this));
        setObjectName(QStringLiteral("PointCloudCameraView"));
        setFrameShape(QFrame::NoFrame);
        setBackgroundBrush(Qt::NoBrush);
        viewport()->setAutoFillBackground(false);
        setDragMode(QGraphicsView::ScrollHandDrag);
        setTransformationAnchor(QGraphicsView::AnchorUnderMouse);
        setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
        setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
        for (int index = 0; index < 2; ++index) {
            auto* button = new QToolButton(viewport());
            button->setObjectName(index ? "PointCloudCameraNavigateNext"
                                        : "PointCloudCameraNavigatePrevious");
            const QString title = QCoreApplication::translate(
                    "PointCloudCameraPanel",
                    index ? "Next camera" : "Previous camera");
            button->setToolTip(title);
            button->setAccessibleName(title);
            button->setFocusPolicy(Qt::NoFocus);
            button->setCursor(Qt::PointingHandCursor);
            button->setFixedSize(32, 32);
            button->setIconSize(QSize(20, 20));
            // Source icons.py camera-arrow geometry, in its 24-unit space.
            QPixmap pixmap(72, 72);
            pixmap.fill(Qt::transparent);
            QPainter painter(&pixmap);
            painter.setRenderHint(QPainter::Antialiasing);
            painter.scale(3, 3);
            painter.setPen(QPen(QColor("#e1e5ec"), 1.5, Qt::SolidLine,
                                Qt::RoundCap, Qt::RoundJoin));
            painter.translate(12, 12);
            painter.rotate(index ? 90 : -90);
            painter.drawLine(QPointF(0, 7), QPointF(0, -7));
            QPolygonF arrow;
            arrow << QPointF(-5, -2) << QPointF(0, -7) << QPointF(5, -2);
            painter.drawPolyline(arrow);
            painter.end();
            button->setIcon(QIcon(pixmap));
            button->setStyleSheet(QStringLiteral(
                    "QToolButton { background: rgba(32, 35, 42, 90); "
                    "border: none; border-radius: 16px; padding: 0; } "
                    "QToolButton:hover { background: rgba(90, 97, 110, 160); "
                    "}"));
            connect(button, &QToolButton::clicked, this, [this, index] {
                if (navigate) navigate(index ? 1 : -1);
            });
            button->hide();
            navigation.push_back(button);
        }
    }
    std::function<void(int)> navigate;
    void setCameraCount(int count) {
        cameraCount = count;
        setToolTip(count > 1 ? QCoreApplication::translate(
                                       "PointCloudCameraPanel",
                                       "Wheel: zoom · Arrows: switch camera")
                             : QString());
        updateNavigation();
    }
    void fitImage() {
        fitted = true;
        if (!sceneRect().isEmpty()) fitInView(sceneRect(), Qt::KeepAspectRatio);
    }

protected:
    bool event(QEvent* event) override {
        if (event->type() == QEvent::Enter || event->type() == QEvent::Leave) {
            hovered = event->type() == QEvent::Enter;
            updateNavigation();
        }
        return QGraphicsView::event(event);
    }
    void wheelEvent(QWheelEvent* event) override {
        if (event->angleDelta().y() && !sceneRect().isEmpty()) {
            const double factor = event->angleDelta().y() > 0 ? 1.2 : 1.0 / 1.2;
            const double next = transform().m11() * factor;
            if (next >= 0.02 && next <= 32.0) {
                fitted = false;
                scale(factor, factor);
            }
        }
        event->accept();
    }
    void mouseDoubleClickEvent(QMouseEvent* event) override {
        fitImage();
        event->accept();
    }
    void resizeEvent(QResizeEvent* event) override {
        QGraphicsView::resizeEvent(event);
        if (fitted) fitImage();
        updateNavigation();
    }

private:
    void updateNavigation() {
        for (int index = 0; index < navigation.size(); ++index) {
            auto* button = navigation[index];
            button->move(
                    index ? viewport()->width() - button->width() - 8 : 8,
                    std::max(0, (viewport()->height() - button->height()) / 2));
            button->setVisible(hovered && cameraCount > 1);
            button->raise();
        }
    }
    QVector<QToolButton*> navigation;
    int cameraCount = 0;
    bool hovered = false;
    bool fitted = true;
};

class CameraImageItem final : public QGraphicsItem {
public:
    std::function<void(QPainter*)> draw;
    QRectF boundingRect() const override { return bounds; }
    void setSize(const QSize& size) {
        prepareGeometryChange();
        bounds = QRectF(QPointF(), size);
        update();
    }
    void paint(QPainter* painter,
               const QStyleOptionGraphicsItem*,
               QWidget*) override {
        if (draw) draw(painter);
    }

private:
    QRectF bounds;
};

class CameraResizeHandle final : public QWidget {
public:
    explicit CameraResizeHandle(PointCloudCameraPanel* panel)
        : QWidget(panel), panel(panel) {
        setObjectName(QStringLiteral("PointCloudCameraResizeHandle"));
        setFixedSize(16, 16);
        setCursor(Qt::SizeBDiagCursor);
        setToolTip(QCoreApplication::translate("PointCloudCameraPanel",
                                               "Drag to resize camera image"));
    }

protected:
    void paintEvent(QPaintEvent*) override {
        QPainter painter(this);
        painter.setRenderHint(QPainter::Antialiasing);
        painter.setPen(QPen(QColor(203, 210, 223, 150), 1.2));
        for (int offset : {0, 4})
            painter.drawLine(3, 6 + offset, 10 - offset, 13);
    }
    void mousePressEvent(QMouseEvent* event) override {
        if (event->button() != Qt::LeftButton) return;
        dragging = true;
        origin = qtCompatMouseEventGlobalPosInt(event);
        originalSize = panel->size();
        event->accept();
    }
    void mouseMoveEvent(QMouseEvent* event) override {
        if (!dragging) return;
        if (!(event->buttons() & Qt::LeftButton)) {
            dragging = false;
            return;
        }
        const QPoint delta = qtCompatMouseEventGlobalPosInt(event) - origin;
        panel->resizeImageRegion(QSize(originalSize.width() - delta.x(),
                                       originalSize.height() + delta.y()));
        event->accept();
    }
    void mouseReleaseEvent(QMouseEvent* event) override {
        if (event->button() == Qt::LeftButton) {
            dragging = false;
            event->accept();
        }
    }

private:
    PointCloudCameraPanel* panel;
    QPoint origin;
    QSize originalSize;
    bool dragging = false;
};
}  // namespace

PointCloudCameraPanel::PointCloudCameraPanel(QWidget* parent)
    : QWidget(parent) {
    setObjectName(QStringLiteral("PointCloudCameraPanel"));
    setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Expanding);
    setAutoFillBackground(false);
    setStyleSheet(QStringLiteral(
            "QWidget#PointCloudCameraPanel, QGraphicsView, "
            "QGraphicsView > QWidget { background: transparent; border: none; "
            "padding: 0; } QLabel { background: transparent; color: #cbd2df; "
            "border: none; padding: 0; font-size: 10px; } "
            "QToolButton { background: transparent; border: none; padding: 0; "
            "min-width: 0; min-height: 0; border-radius: 3px; } "
            "QToolButton:hover, QToolButton:checked { "
            "background: rgba(203, 210, 223, 30); }"));
    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(0);
    m_cameraControls = new QWidget(this);
    m_cameraControls->setFixedHeight(24);
    auto* controls = new QHBoxLayout(m_cameraControls);
    controls->setContentsMargins(18, 2, 2, 2);
    controls->setSpacing(4);
    m_cameraName = new QLabel(m_cameraControls);
    m_cameraName->setObjectName(QStringLiteral("PointCloudCameraName"));
    m_cameraName->setTextFormat(Qt::PlainText);
    m_cameraName->setMinimumWidth(0);
    m_cameraName->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);
    controls->addWidget(m_cameraName, 1);
    const auto toggle = [&](const QString& text, const QString& name) {
        auto* button = new QToolButton(m_cameraControls);
        button->setText(text);
        button->setToolTip(text);
        button->setCheckable(true);
        button->setChecked(true);
        button->setObjectName(name);
        controls->addWidget(button);
        connect(button, &QToolButton::toggled, this, [this] { update(); });
        return button;
    };
    m_pointsToggle = toggle(tr("Projected points"),
                            QStringLiteral("PointCloudCameraPoints"));
    m_cuboidsToggle = toggle(tr("Projected 3D boxes"),
                             QStringLiteral("PointCloudCameraCuboids"));
    auto* settings = new QToolButton(m_cameraControls);
    settings->setText(tr("Projection settings"));
    settings->setToolTip(tr("Projection settings"));
    settings->setAccessibleName(tr("Projection settings"));
    const auto icon = [&](QToolButton* button, const QString& name,
                          QStyle::StandardPixmap fallback) {
        const QString path =
                QStringLiteral(":/CC/plugin/qAnyLabeling/images/") + name;
        button->setIcon(QFileInfo::exists(path)
                                ? QIcon(path)
                                : style()->standardIcon(fallback));
        button->setToolButtonStyle(Qt::ToolButtonIconOnly);
        button->setIconSize(QSize(13, 13));
        button->setFixedSize(20, 20);
    };
    icon(m_pointsToggle, QStringLiteral("point.png"), QStyle::SP_ComputerIcon);
    icon(m_cuboidsToggle, QStringLiteral("cuboid.png"),
         QStyle::SP_FileDialogDetailedView);
    icon(settings, QStringLiteral("settings.svg"),
         QStyle::SP_FileDialogContentsView);
    settings->setObjectName(QStringLiteral("PointCloudCameraSettings"));
    auto* menu = new ProjectionSettingsMenu(this);
    m_overlaySettings = menu;
    menu->setObjectName(QStringLiteral("PointCloudCameraSettingsMenu"));
    m_settings = new QSettings(this);
    auto* content = new QWidget(menu);
    auto* form = new QGridLayout(content);
    form->setContentsMargins(14, 12, 14, 12);
    form->setHorizontalSpacing(12);
    form->setVerticalSpacing(10);
    m_overlayColor = new QButtonGroup(content);
    m_overlayColor->setObjectName(QStringLiteral("PointCloudCameraColor"));
    auto* choices = new QHBoxLayout;
    choices->setSpacing(16);
    for (int index = 0; index < 2; ++index) {
        auto* button = new QRadioButton(
                index ? tr("Point cloud colors") : tr("Depth"), content);
        button->setProperty("color_mode", index ? "cloud" : "depth");
        m_overlayColor->addButton(button, index);
        choices->addWidget(button);
    }
    choices->addStretch();
    const QString mode =
            m_settings->value(cameraSettings + "color", "depth").toString();
    m_overlayColor->button(mode == "cloud" ? 1 : 0)->setChecked(true);
    auto* colorLabel = new QLabel(tr("Color"), content);
    colorLabel->setBuddy(m_overlayColor->button(0));
    form->addWidget(colorLabel, 0, 0);
    form->addLayout(choices, 0, 1, 1, 3);
    m_overlaySize = new QSlider(Qt::Horizontal, content);
    m_overlaySize->setObjectName(QStringLiteral("PointCloudCameraPointSize"));
    m_overlayOpacity = new QSlider(Qt::Horizontal, content);
    m_overlayOpacity->setObjectName(QStringLiteral("PointCloudCameraOpacity"));
    const auto sliderRow = [&](QSlider* slider, int row, const QString& title,
                               const QString& key, int minimum, int maximum,
                               int initial, const QString& suffix,
                               const QString& resetTitle) {
        slider->setRange(minimum, maximum);
        slider->setValue(
                m_settings->value(cameraSettings + key, initial).toInt());
        slider->setMinimumWidth(120);
        slider->setAccessibleName(title);
        auto* label = new QLabel(title, content);
        label->setBuddy(slider);
        auto* value =
                new QLabel(QString::number(slider->value()) + suffix, content);
        value->setObjectName(slider->objectName() + "Value");
        value->setMinimumWidth(
                QTCOMPAT_FONTMETRICS_WIDTH(value->fontMetrics(), "100 px"));
        value->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
        connect(slider, &QSlider::valueChanged, value,
                [value, suffix](int number) {
                    value->setText(QString::number(number) + suffix);
                });
        form->addWidget(label, row, 0);
        form->addWidget(slider, row, 1);
        form->addWidget(value, row, 2);
        auto* reset = new QToolButton(content);
        reset->setObjectName(slider->objectName() + "Reset");
        reset->setIcon(style()->standardIcon(QStyle::SP_BrowserReload));
        reset->setIconSize(QSize(14, 14));
        reset->setFixedSize(24, 24);
        reset->setToolTip(resetTitle);
        reset->setAccessibleName(resetTitle);
        connect(reset, &QToolButton::clicked, slider,
                [slider, initial] { slider->setValue(initial); });
        form->addWidget(reset, row, 3);
    };
    sliderRow(m_overlaySize, 1, tr("Point size"), "size", 1, 7, 1, " px",
              tr("Reset point size (1 px)"));
    sliderRow(m_overlayOpacity, 2, tr("Opacity"), "opacity", 0, 100, 65, "%",
              tr("Reset opacity (65%)"));
    m_depthLegend = new QLabel(tr("Near → Far"), content);
    m_depthLegend->setObjectName(QStringLiteral("PointCloudCameraDepthLegend"));
    m_depthLegend->setAlignment(Qt::AlignCenter);
    m_depthLegend->setStyleSheet(QStringLiteral(
            "color: #101828; padding: 3px 8px; border-radius: 3px; "
            "background: qlineargradient(x1:0, y1:0, x2:1, y2:0, "
            "stop:0 #ff6441, stop:0.25 #ffd740, stop:0.5 #67dc78, "
            "stop:0.75 #32bee6, stop:1 #5a64eb);"));
    m_depthLegend->setEnabled(m_overlayColor->checkedId() == 0);
    form->addWidget(m_depthLegend, 3, 0, 1, 4);
    auto* action = new QWidgetAction(menu);
    action->setDefaultWidget(content);
    menu->addAction(action);
    connect(settings, &QToolButton::clicked, menu, [settings, menu] {
        menu->popup(settings->mapToGlobal(
                QPoint(settings->width() - menu->sizeHint().width(),
                       settings->height())));
    });
    controls->addWidget(settings);
    for (auto* button : m_overlayColor->buttons())
        connect(button, &QAbstractButton::toggled, this,
                [this, button](bool checked) {
                    if (!checked) return;
                    m_settings->setValue(cameraSettings + "color",
                                         button->property("color_mode"));
                    m_depthLegend->setEnabled(m_overlayColor->checkedId() == 0);
                    refreshPointOverlay();
                });
    connect(m_overlaySize, &QSlider::valueChanged, this, [this](int value) {
        m_settings->setValue(cameraSettings + "size", value);
        refreshPointOverlay();
    });
    connect(m_overlayOpacity, &QSlider::valueChanged, this, [this](int value) {
        m_settings->setValue(cameraSettings + "opacity", value);
        update();
    });
    auto* view = new CameraImageView(this);
    view->navigate = [this](int delta) { stepCamera(delta); };
    m_cameraView = view;
    auto* messages = new QVBoxLayout(view->viewport());
    messages->setContentsMargins(42, 4, 42, 4);
    m_message = new QLabel(view->viewport());
    m_message->setObjectName(QStringLiteral("PointCloudCameraMessage"));
    m_message->setAlignment(Qt::AlignCenter);
    m_message->setWordWrap(true);
    m_message->setTextFormat(Qt::PlainText);
    m_message->setAttribute(Qt::WA_TransparentForMouseEvents);
    m_message->setStyleSheet(QStringLiteral(
            "background: transparent; color: #cbd2df; border: none;"));
    messages->addWidget(m_message);
    auto* imageItem = new CameraImageItem;
    imageItem->draw = [this](QPainter* painter) { paintCameraImage(painter); };
    m_imageItem = imageItem;
    view->scene()->addItem(imageItem);
    layout->addWidget(view, 1);
    auto* fit = new QToolButton(m_cameraControls);
    fit->setObjectName(QStringLiteral("PointCloudCameraFit"));
    fit->setToolTip(tr("Fit image"));
    fit->setAccessibleName(tr("Fit image"));
    icon(fit, QStringLiteral("fit.png"), QStyle::SP_TitleBarMaxButton);
    controls->addWidget(fit);
    connect(fit, &QToolButton::clicked, this,
            &PointCloudCameraPanel::fitCameraImage);
    layout->addWidget(m_cameraControls);
    m_resizeHandle = new CameraResizeHandle(this);
    m_resizeHandle->hide();
    m_cameraControls->hide();
}

void PointCloudCameraPanel::setSource(const PointCloudCameraSource& source) {
    setSources({source});
}

void PointCloudCameraPanel::setSources(
        const QVector<PointCloudCameraSource>& sources) {
    const QString previous = m_hasSource ? m_source.directory : QString();
    m_sources = sources;
    m_images.clear();
    static_cast<CameraImageView*>(m_cameraView)->setCameraCount(sources.size());
    int active = sources.isEmpty() ? -1 : 0;
    for (int index = 0; index < sources.size(); ++index) {
        if (sources[index].directory == previous) active = index;
    }
    m_cameraControls->setVisible(!sources.isEmpty());
    m_activeIndex = -1;
    if (active >= 0)
        setActiveCameraIndex(active);
    else {
        m_source = PointCloudCameraSource();
        m_hasSource = false;
        m_cameraName->clear();
        clearImage();
    }
}

void PointCloudCameraPanel::setActiveCameraIndex(int index) {
    if (index < 0 || index >= m_sources.size() || index == m_activeIndex)
        return;
    m_activeIndex = index;
    m_source = m_sources[index];
    m_hasSource = true;
    const QDir directory(m_source.directory);
    const QString name =
            directory.dirName() == QStringLiteral("data")
                    ? QFileInfo(directory.absolutePath()).dir().dirName()
                    : directory.dirName();
    m_cameraName->setText(name.isEmpty() ? directory.absolutePath() : name);
    refreshFrame();
}

void PointCloudCameraPanel::stepCamera(int delta) {
    if (m_sources.size() < 2) return;
    const qint64 count = m_sources.size();
    setActiveCameraIndex(
            int(((qint64(m_activeIndex) + delta) % count + count) % count));
}

void PointCloudCameraPanel::clearSource() { setSources({}); }

void PointCloudCameraPanel::fitCameraImage() {
    static_cast<CameraImageView*>(m_cameraView)->fitImage();
}

void PointCloudCameraPanel::attachToViewport(QWidget* viewport) {
    if (m_overlayHost) m_overlayHost->removeEventFilter(this);
    m_overlayHost = viewport;
    if (viewport) {
        setParent(viewport);
        viewport->installEventFilter(this);
    }
    m_resizeHandle->setVisible(viewport != nullptr);
    positionPanel();
}

void PointCloudCameraPanel::resizeImageRegion(const QSize& size) {
    m_userSize = size;
    positionPanel();
}

void PointCloudCameraPanel::positionPanel() {
    if (!m_overlayHost) return;
    // Source CameraPanel.minimum_image_size/position_panel. The requested
    // size survives a temporarily smaller viewport; geometry always fits it.
    const int areaWidth = m_overlayHost->width();
    const int areaHeight = m_overlayHost->height();
    const int minimumWidth =
            std::max(1, std::min({420, std::max(180, int(areaWidth * .42)),
                                  areaWidth - 16}));
    const double ratio =
            m_image.isNull() ? .6 : double(m_image.height()) / m_image.width();
    const int minimumHeight =
            std::max(1, std::min(int(minimumWidth * ratio) + 24,
                                 std::max(100, areaHeight / 2)));
    const QSize requested = m_userSize.isValid()
                                    ? m_userSize
                                    : QSize(minimumWidth, minimumHeight);
    const int width = std::min(std::max(minimumWidth, requested.width()),
                               std::max(1, areaWidth - 16));
    const int height = std::min(std::max(minimumHeight, requested.height()),
                                std::max(1, areaHeight - 16));
    setGeometry(areaWidth - width - 8, 8, width, height);
    raise();
}

bool PointCloudCameraPanel::eventFilter(QObject* watched, QEvent* event) {
    if (watched == m_overlayHost && event->type() == QEvent::Resize)
        positionPanel();
    return QWidget::eventFilter(watched, event);
}

void PointCloudCameraPanel::resizeEvent(QResizeEvent* event) {
    QWidget::resizeEvent(event);
    if (m_resizeHandle) {
        m_resizeHandle->move(0,
                             std::max(0, height() - m_resizeHandle->height()));
        m_resizeHandle->raise();
    }
}

void PointCloudCameraPanel::hideEvent(QHideEvent* event) {
    if (m_overlaySettings) m_overlaySettings->hide();
    QWidget::hideEvent(event);
}

void PointCloudCameraPanel::clearImage() {
    m_image = QImage();
    static_cast<CameraImageItem*>(m_imageItem)->setSize(QSize());
    m_cameraView->setSceneRect(QRectF());
    m_imagePath.clear();
    m_cameraName->setToolTip(
            m_frame.path.isEmpty()
                    ? QString()
                    : QFileInfo(m_frame.path).completeBaseName());
    m_projectionEnabled = false;
    m_statusMessage.clear();
    m_projectedPoints.clear();
    m_pointIndices.clear();
    m_depthColors.clear();
    m_depthLegend->setText(tr("Near → Far"));
    m_depthLegend->setToolTip(QString());
    m_pointOverlay = QImage();
    m_projectedCuboids.clear();
    m_projectedPreview.clear();
    m_pointsToggle->setEnabled(false);
    m_cuboidsToggle->setEnabled(false);
    update();
}

void PointCloudCameraPanel::setFrame(const PointCloudFrame* frame) {
    // Qt vectors share storage until edited; switching cameras does not copy
    // point/label arrays or borrow a frame that may be destroyed by its caller.
    if (frame && m_hasFrame && frame->path == m_frame.path &&
        frame->points.constData() == m_frame.points.constData() &&
        frame->labels.constData() == m_frame.labels.constData() &&
        frame->rgb.constData() == m_frame.rgb.constData() &&
        frame->cuboids == m_frame.cuboids)
        return;
    const bool sameGeometry =
            frame && m_hasFrame && frame->path == m_frame.path &&
            frame->points.constData() == m_frame.points.constData();
    if (!sameGeometry) m_hasEditPreview = false;
    m_hasFrame = frame != nullptr;
    m_frame = frame ? *frame : PointCloudFrame();
    // Annotation/color edits do not change the camera image or projected
    // point positions. Preserve the user's zoom/pan and the source's per-frame
    // projection cache while rebuilding only mutable overlays.
    if (sameGeometry && !m_image.isNull()) {
        if (m_projectionEnabled && !refreshCuboidProjection()) return;
        refreshPointOverlay();
        return;
    }
    refreshFrame();
}

bool PointCloudCameraPanel::refreshCuboidProjection() {
    m_projectedCuboids.clear();
    m_projectedPreview.clear();
    m_statusMessage.clear();
    for (const PointCloudCuboid& member : m_frame.cuboids) {
        const auto& cuboid = m_hasEditPreview && member.id == m_editPreview.id
                                     ? m_editPreview
                                     : member;
        QVector<QLineF> segments;
        QString error;
        if (!projectCuboid(cuboid, m_source.calibration, &segments, &error)) {
            m_statusMessage = error;
            m_projectedCuboids.clear();
            update();
            return false;
        }
        m_projectedCuboids.append(
                {cuboid.id, cuboid.classId, cuboid.occluded, segments});
    }
    if (m_hasCreationPreview) {
        QString error;
        if (!projectCuboid(m_creationPreview, m_source.calibration,
                           &m_projectedPreview, &error)) {
            m_statusMessage = error;
            update();
            return false;
        }
    }
    return true;
}

void PointCloudCameraPanel::refreshFrame() {
    clearImage();
    positionPanel();
    if (!m_hasSource || !m_hasFrame) return;
    const auto* frame = &m_frame;
    const QString stem = QFileInfo(frame->path).completeBaseName();
    const QString path = m_source.imagePaths.value(stem);
    if (path.isEmpty()) return;
    const QImage* cached = m_images.object(path);
    QImage image = cached ? *cached : QImage(path);
    if (image.isNull()) return;
    if (!cached) m_images.insert(path, new QImage(image));
    m_image = image.convertToFormat(QImage::Format_RGB32);
    static_cast<CameraImageItem*>(m_imageItem)->setSize(m_image.size());
    m_cameraView->setSceneRect(QRectF(m_image.rect()));
    m_imagePath = path;
    m_cameraName->setToolTip(path);
    positionPanel();
    fitCameraImage();
    if (m_source.hasCalibration) {
        if (m_image.size() != m_source.calibration.imageSize) {
            m_statusMessage =
                    tr("Image size %1 x %2 does not match calibration %3 x %4. "
                       "Projection disabled.")
                            .arg(m_image.width())
                            .arg(m_image.height())
                            .arg(m_source.calibration.imageSize.width())
                            .arg(m_source.calibration.imageSize.height());
            update();
            return;
        }
        QString error;
        if (!projectPointCloud(frame->points, m_source.calibration,
                               &m_pointIndices, &m_projectedPoints, &error)) {
            m_statusMessage = error;
            m_projectedPoints.clear();
            update();
            return;
        }
        if (!refreshCuboidProjection()) return;
        double nearDepth = 0, farDepth = 0;
        if (!pointCloudCameraDepthColors(frame->points, m_source.calibration,
                                         m_pointIndices, &m_depthColors, &error,
                                         &nearDepth, &farDepth)) {
            m_statusMessage = error;
            return;
        }
        if (!m_pointIndices.isEmpty()) {
            m_depthLegend->setText(
                    tr("Near %1 → Far %2")
                            .arg(QString::number(nearDepth, 'g', 3),
                                 QString::number(farDepth, 'g', 3)));
            m_depthLegend->setToolTip(
                    tr("Camera depth in point-cloud units. Colors use the "
                       "2nd–98th percentile of this frame; values outside the "
                       "range are clipped."));
        }
        m_projectionEnabled = true;
    }
    m_pointsToggle->setEnabled(m_projectionEnabled);
    m_cuboidsToggle->setEnabled(m_projectionEnabled && m_detection);
    refreshPointOverlay();
    update();
}

void PointCloudCameraPanel::setVisibility(const QSet<quint16>& hiddenSemantics,
                                          const QSet<quint16>& hiddenInstances,
                                          const QSet<int>& hiddenCuboids) {
    if (m_hiddenSemantics == hiddenSemantics &&
        m_hiddenInstances == hiddenInstances &&
        m_hiddenCuboids == hiddenCuboids)
        return;
    m_hiddenSemantics = hiddenSemantics;
    m_hiddenInstances = hiddenInstances;
    m_hiddenCuboids = hiddenCuboids;
    refreshPointOverlay();
}

void PointCloudCameraPanel::setCuboidStyle(const QHash<int, QColor>& colors,
                                           int selectedId,
                                           bool detection) {
    m_classColors = colors;
    m_selectedCuboid = selectedId;
    m_detection = detection;
    m_cuboidsToggle->setEnabled(m_projectionEnabled && detection);
    update();
}

void PointCloudCameraPanel::setPointColorStyle(
        PointCloudColorMode mode, const QHash<quint16, QColor>& colors) {
    if (m_pointColorMode == mode && m_pointColors == colors) return;
    m_pointColorMode = mode;
    m_pointColors = colors;
    refreshPointOverlay();
}

void PointCloudCameraPanel::setCreationPreview(
        const PointCloudCuboid* preview) {
    if (m_hasCreationPreview == bool(preview) &&
        (!preview || m_creationPreview == *preview))
        return;
    m_hasCreationPreview = preview != nullptr;
    if (preview) m_creationPreview = *preview;
    m_projectedPreview.clear();
    if (m_projectionEnabled) refreshCuboidProjection();
    update();
}
void PointCloudCameraPanel::setEditPreview(const PointCloudCuboid* preview) {
    if (m_hasEditPreview == bool(preview) &&
        (!preview || m_editPreview == *preview))
        return;
    m_hasEditPreview = preview != nullptr;
    if (preview) m_editPreview = *preview;
    if (m_projectionEnabled) refreshCuboidProjection();
    update();
}

void PointCloudCameraPanel::refreshPointOverlay() {
    m_pointOverlay = QImage();
    if (m_projectionEnabled && !m_image.isNull()) {
        // Const access preserves QVector's shared point storage across label
        // and color edits, so those edits cannot invalidate the projection.
        const PointCloudFrame& frame = m_frame;
        m_pointOverlay = QImage(m_image.size(), QImage::Format_ARGB32);
        m_pointOverlay.fill(Qt::transparent);
        const int size = m_overlaySize->value(), begin = -(size / 2);
        double low = std::numeric_limits<double>::infinity(), high = -low;
        if (m_pointColorMode == PointCloudColorMode::Intensity) {
            for (const auto& point : frame.points) {
                if (!std::isfinite(point.w())) continue;
                low = std::min(low, double(point.w()));
                high = std::max(high, double(point.w()));
            }
            if (!std::isfinite(low)) {
                low = 0;
                high = 1;
            }
        }
        for (int rank = 0; rank < m_pointIndices.size(); ++rank) {
            const int index = m_pointIndices[rank];
            const quint32 label =
                    index < frame.labels.size() ? frame.labels[index] : 0;
            if (m_hiddenSemantics.contains(quint16(label)) ||
                (quint16(label >> 16) &&
                 m_hiddenInstances.contains(quint16(label >> 16))))
                continue;
            const auto display = pointCloudDisplayColor(
                    label, frame.points[index].w(),
                    index < frame.rgb.size() ? &frame.rgb[index] : nullptr,
                    m_pointColorMode, m_pointColors, low, high);
            // Source converts the viewport's float32 RGB array to uint8 by
            // truncation, rather than QColor's floating-point rounding.
            const auto channel = [](float value) {
                return int(std::clamp(value * 255.f, 0.f, 255.f));
            };
            const QColor color =
                    m_overlayColor->checkedId() == 0
                            ? m_depthColors[rank]
                            : QColor(channel(display.x()), channel(display.y()),
                                     channel(display.z()));
            const QPoint pixel = m_projectedPoints[rank];
            // Farthest-to-nearest order gives the last visible point ownership
            // of each image pixel, including every offset of a larger splat.
            for (int dy = begin; dy < begin + size; ++dy) {
                const int y = pixel.y() + dy;
                if (y < 0 || y >= m_pointOverlay.height()) continue;
                auto* row = reinterpret_cast<QRgb*>(m_pointOverlay.scanLine(y));
                for (int dx = begin; dx < begin + size; ++dx) {
                    const int x = pixel.x() + dx;
                    if (x >= 0 && x < m_pointOverlay.width())
                        row[x] = color.rgb();
                }
            }
        }
    }
    update();
}

void PointCloudCameraPanel::setOverlayEnabled(bool enabled) {
    if (m_overlayEnabled == enabled) return;
    m_overlayEnabled = enabled;
    update();
}

void PointCloudCameraPanel::paintEvent(QPaintEvent* event) {
    Q_UNUSED(event);
    m_imageItem->update();
    QString message = m_statusMessage;
    if (m_image.isNull()) {
        if (message.isEmpty())
            message =
                    !m_hasSource ? tr("Configure a camera image directory")
                    : !m_hasFrame
                            ? tr("Open a point cloud to view its camera image.")
                            : tr("No matching camera image for this frame.");
    }
    m_message->setText(message);
    m_message->setVisible(!message.isEmpty());
}

void PointCloudCameraPanel::paintCameraImage(QPainter* targetPainter) {
    if (m_image.isNull()) return;
    QPainter& painter = *targetPainter;
    const QRectF target(m_image.rect());
    // Source QGraphicsPixmapItems use their default FastTransformation for
    // both the camera photo and point splats, including manual zoom.
    painter.setRenderHint(QPainter::SmoothPixmapTransform, false);
    painter.drawImage(QPointF(), m_image);
    if (!m_overlayEnabled || !m_source.hasCalibration || !m_projectionEnabled)
        return;
    painter.setRenderHint(QPainter::Antialiasing, true);
    if (m_pointsToggle->isChecked()) {
        painter.save();
        painter.setRenderHint(QPainter::SmoothPixmapTransform, false);
        painter.setOpacity(m_overlayOpacity->value() / 100.0);
        painter.drawImage(target, m_pointOverlay);
        painter.restore();
    }
    if (!m_detection || !m_cuboidsToggle->isChecked()) return;
    painter.setClipRect(target);
    // Selected objects are drawn last, as upstream does, with cosmetic widths
    // so zooming the image does not thicken the annotations.
    for (int selected = 0; selected < 2; ++selected)
        for (const auto& box : m_projectedCuboids) {
            if ((box.id == m_selectedCuboid) != bool(selected) ||
                m_hiddenCuboids.contains(box.id))
                continue;
            QPen pen(m_classColors.value(box.classId,
                                         QColor(QStringLiteral("#64b5f6"))),
                     selected ? 2.5 : 1.3);
            pen.setCapStyle(Qt::RoundCap);
            pen.setCosmetic(true);
            if (box.occluded) pen.setStyle(Qt::DashLine);
            painter.setPen(pen);
            QPainterPath path;
            for (const auto& segment : box.lines) {
                path.moveTo(segment.p1());
                path.lineTo(segment.p2());
            }
            painter.drawPath(path);
        }
    QPen previewPen(QColor(QStringLiteral("#ffd54f")), 2.5);
    previewPen.setCosmetic(true);
    previewPen.setCapStyle(Qt::RoundCap);
    painter.setPen(previewPen);
    QPainterPath previewPath;
    for (const auto& segment : m_projectedPreview) {
        previewPath.moveTo(segment.p1());
        previewPath.lineTo(segment.p2());
    }
    painter.drawPath(previewPath);
}
