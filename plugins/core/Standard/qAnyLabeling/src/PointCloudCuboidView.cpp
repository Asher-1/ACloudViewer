// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "PointCloudCuboidView.h"

#include <QKeyEvent>
#include <QMouseEvent>
#include <QPainter>
#include <QStyle>
#include <QToolButton>
#include <QVariantAnimation>
#include <QWheelEvent>
#include <algorithm>
#include <cmath>
#include <limits>

namespace {
// Source perimeter order also determines ties between equally near handles.
constexpr int kHandleSigns[8][2] = {{-1, -1}, {0, -1}, {1, -1}, {1, 0},
                                    {1, 1},   {0, 1},  {-1, 1}, {-1, 0}};
}  // namespace

PointCloudCuboidView::PointCloudCuboidView(int horizontal,
                                           int vertical,
                                           const QString& title,
                                           QWidget* parent)
    : QWidget(parent), m_h(horizontal), m_v(vertical), m_title(title) {
    setMinimumSize(160, 140);
    setFocusPolicy(Qt::StrongFocus);
    setMouseTracking(true);
    m_expandButton = new QToolButton(this);
    m_expandButton->setObjectName(
            QStringLiteral("PointCloudCuboidExpandButton"));
    m_expandButton->setFixedSize(24, 24);
    m_expandButton->setAutoRaise(true);
    m_expandButton->setFocusPolicy(Qt::NoFocus);
    m_expandButton->setStyleSheet(
            QStringLiteral("QToolButton { background: rgba(25, 27, 31, 210); "
                           "border: 1px solid #687483; border-radius: 2px; }"));
    connect(m_expandButton, &QToolButton::clicked, this, [this]() {
        if (onExpandRequested) onExpandRequested();
    });
    setExpanded(false);
    m_focusAnimation = new QVariantAnimation(this);
    m_focusAnimation->setObjectName(
            QStringLiteral("CuboidOrthographicFocusAnimation"));
    m_focusAnimation->setDuration(250);
    m_focusAnimation->setEasingCurve(QEasingCurve::InOutCubic);
}
void PointCloudCuboidView::setVisiblePointIndices(const QVector<int>* indices) {
    if (indices == m_visiblePointIndices) return;
    m_visiblePointIndices = indices;
    update();
}
void PointCloudCuboidView::setPointColorProvider(
        std::function<QColor(int)> provider) {
    m_pointColorProvider = std::move(provider);
    update();
}
void PointCloudCuboidView::setPointSize(int size) {
    size = qBound(1, size, 10);
    if (size == m_pointSize) return;
    m_pointSize = size;
    update();
}
void PointCloudCuboidView::setDepthCropEnabled(bool enabled) {
    if (m_depthCropEnabled == enabled) return;
    m_depthCropEnabled = enabled;
    update();
}
void PointCloudCuboidView::setExpanded(bool expanded) {
    m_expanded = expanded;
    const auto title = expanded ? tr("Restore view") : tr("Expand view");
    m_expandButton->setToolTip(title);
    m_expandButton->setAccessibleName(title);
    m_expandButton->setIcon(
            style()->standardIcon(expanded ? QStyle::SP_TitleBarNormalButton
                                           : QStyle::SP_TitleBarMaxButton));
    m_expandButton->setVisible(expanded || m_hovered);
    m_expandButton->move(width() - 32, 6);
    m_expandButton->raise();
}
const PointCloudCuboid* PointCloudCuboidView::selected() const {
    if (m_dragging && !m_creating) return &m_preview;
    if (m_hasEditPreview && m_editPreview.id == m_selected)
        return &m_editPreview;
    if (m_document)
        for (const auto& b : m_document->frame().cuboids)
            if (b.id == m_selected) return &b;
    return nullptr;
}
void PointCloudCuboidView::setDocument(const PointCloudDocument* d, int id) {
    const QString path = d ? d->frame().path : QString();
    const bool changed =
            d != m_document || id != m_selected || path != m_framePath;
    const auto* previous = selected();
    const CuboidVector previousOrigin = m_origin;
    const CuboidVector previousOrientation = m_orientation;
    const QPointF previousPan = m_pan;
    m_document = d;
    m_framePath = path;
    m_selected = id;
    m_dragging = false;
    m_panning = false;
    m_hasEditPreview = false;
    m_gestureButton = Qt::NoButton;
    updateHandleHover(-1);
    if (changed)
        focusCuboid();
    else if (const auto* box = selected()) {
        // Keep the world-space camera center stable when the selected cuboid
        // rotates.  This is the orthographic equivalent of upstream's
        // orientation.T @ (camera_center - box.center) adjustment.
        if (previous && previous->id == box->id &&
            previousOrientation != box->rotation) {
            CuboidVector localCamera{};
            localCamera[m_h] = previousPan.x() / m_scale;
            localCamera[m_v] = -previousPan.y() / m_scale;
            const auto cameraLocal = cuboidTransform(
                    cuboidRotationMatrix(previousOrientation), localCamera);
            CuboidVector camera{};
            for (int i = 0; i < 3; ++i)
                camera[i] = previousOrigin[i] + cameraLocal[i];
            m_origin = box->center;
            m_orientation = box->rotation;
            CuboidVector cameraOffset{};
            for (int i = 0; i < 3; ++i)
                cameraOffset[i] = camera[i] - m_origin[i];
            const auto local = cuboidTransform(
                    cuboidRotationMatrix(m_orientation), cameraOffset, true);
            m_pan = QPointF(local[m_h] * m_scale, -local[m_v] * m_scale);
        } else {
            m_orientation = box->rotation;
        }
    }
    update();
}
void PointCloudCuboidView::setEditPreview(const PointCloudCuboid* box,
                                          bool followTranslation) {
    const auto* previous = selected();
    const auto before = previous ? *previous : PointCloudCuboid();
    m_hasEditPreview = box != nullptr;
    if (box) m_editPreview = *box;
    const auto* after = selected();
    if (previous && after && before.id == after->id) {
        if (followTranslation && before.size == after->size &&
            before.rotation == after->rotation)
            for (int i = 0; i < 3; ++i)
                m_origin[i] += after->center[i] - before.center[i];
        if (m_orientation != after->rotation) {
            m_focusAnimation->stop();
            // Source rotates the camera offset with the selected box.
            auto offset = m_origin;
            for (int i = 0; i < 3; ++i) offset[i] -= after->center[i];
            offset = cuboidTransform(cuboidRotationMatrix(m_orientation),
                                     offset, true);
            offset = cuboidTransform(cuboidRotationMatrix(after->rotation),
                                     offset);
            for (int i = 0; i < 3; ++i)
                m_origin[i] = after->center[i] + offset[i];
            m_orientation = after->rotation;
        }
    }
    update();
}
void PointCloudCuboidView::setCreating(bool creating) {
    m_focusAnimation->stop();
    m_creating = creating;
    m_dragging = false;
    m_panning = false;
    m_gestureButton = Qt::NoButton;
    if (onEditPreview) onEditPreview(nullptr);
    updateHandleHover(-1);
    setCursor(creating ? Qt::CrossCursor : Qt::ArrowCursor);
    update();
}
void PointCloudCuboidView::focusCuboid(bool animate) {
    m_focusAnimation->stop();
    const auto startOrigin = m_origin;
    const auto startPan = m_pan;
    // Source interpolates world units per half-height, not pixels/metre.
    const double startScale = std::max(1, height()) / (2 * m_scale);
    m_pan = {};
    if (const auto* box = selected()) {
        m_origin = box->center;
        m_orientation = box->rotation;
        const double w = std::max(1, width()), h = std::max(1, height());
        const double padding = std::min(50.0, std::min(w, h) / 4);
        const double scale =
                std::clamp(std::max(box->size[m_v] * h / (h - 2 * padding),
                                    box->size[m_h] * h / (w - 2 * padding)) /
                                   2,
                           .025, 50.0);
        m_scale = h / (2 * scale);
    } else {
        m_origin = {};
        m_orientation = {};
    }
    if (animate && selected()) {
        const auto targetOrigin = m_origin;
        const double targetScale = std::max(1, height()) / (2 * m_scale);
        m_focusAnimation->disconnect(this);
        connect(m_focusAnimation, &QVariantAnimation::valueChanged, this,
                [this, startOrigin, targetOrigin, startScale, targetScale,
                 startPan](const QVariant& value) {
                    const double progress = value.toDouble();
                    for (int axis = 0; axis < 3; ++axis)
                        m_origin[axis] =
                                startOrigin[axis] +
                                (targetOrigin[axis] - startOrigin[axis]) *
                                        progress;
                    m_scale = std::max(1, height()) /
                              (2 * (startScale +
                                    (targetScale - startScale) * progress));
                    m_pan = startPan * (1 - progress);
                    update();
                });
        m_origin = startOrigin;
        m_pan = startPan;
        m_scale = std::max(1, height()) / (2 * startScale);
        m_focusAnimation->setStartValue(0.0);
        m_focusAnimation->setEndValue(1.0);
        m_focusAnimation->start();
    }
    update();
}
QPointF PointCloudCuboidView::project(const CuboidVector& world) const {
    CuboidVector p = world;
    for (int i = 0; i < 3; ++i) p[i] -= m_origin[i];
    p = cuboidTransform(cuboidRotationMatrix(m_orientation), p, true);
    return QPointF(width() / 2.0 + p[m_h] * m_scale,
                   height() / 2.0 - p[m_v] * m_scale) +
           m_pan;
}
CuboidVector PointCloudCuboidView::localDelta(const QPointF& screen) const {
    CuboidVector d{};
    d[m_h] = screen.x() / m_scale;
    d[m_v] = -screen.y() / m_scale;
    return d;
}
QPointF PointCloudCuboidView::boxCenter() const {
    if (const auto* box = selected()) return project(box->center);
    return QPointF(width() / 2.0, height() / 2.0) + m_pan;
}
QRectF PointCloudCuboidView::boxRect() const {
    if (const auto* b = selected())
        return QRectF(boxCenter() - QPointF(b->size[m_h] * m_scale / 2,
                                            b->size[m_v] * m_scale / 2),
                      QSizeF(b->size[m_h] * m_scale, b->size[m_v] * m_scale));
    return {};
}
QPointF PointCloudCuboidView::handlePosition(int index) const {
    if (index == 8) {
        const auto top = handlePosition(5);
        const auto direction = top - boxCenter();
        return top +
               direction * (28 / std::max(1e-6, std::hypot(direction.x(),
                                                           direction.y())));
    }
    if (const auto* box = selected()) {
        CuboidVector local{};
        local[m_h] = kHandleSigns[index][0] * box->size[m_h] / 2;
        local[m_v] = kHandleSigns[index][1] * box->size[m_v] / 2;
        auto world =
                cuboidTransform(cuboidRotationMatrix(box->rotation), local);
        for (int i = 0; i < 3; ++i) world[i] += box->center[i];
        return project(world);
    }
    return {};
}
int PointCloudCuboidView::hitHandle(const QPointF& point) const {
    const auto* box = selected();
    if (m_creating || !box || box->locked || !m_document ||
        m_document->frame().points.isEmpty())
        return -1;
    int hit = -1;
    double best = std::min(9.0, QLineF(point, boxCenter()).length());
    for (int i = 0; i < 8; ++i) {
        const double distance = QLineF(point, handlePosition(i)).length();
        if (distance < best) {
            best = distance;
            hit = i;
        }
    }
    if (hit >= 0) return hit;
    return QLineF(point, handlePosition(8)).length() < 10 ? 8 : -1;
}
void PointCloudCuboidView::updateHandleHover(int index) {
    if (index != m_hoverHandle) {
        m_hoverHandle = index;
        update();
    }
    auto cursor = m_creating ? Qt::CrossCursor : Qt::ArrowCursor;
    if (index == 8)
        cursor = m_dragging && m_rotating ? Qt::ClosedHandCursor
                                          : Qt::OpenHandCursor;
    else if (index >= 0) {
        const auto& signs = kHandleSigns[index];
        cursor = signs[0] == 0             ? Qt::SizeVerCursor
                 : signs[1] == 0           ? Qt::SizeHorCursor
                 : signs[0] * signs[1] < 0 ? Qt::SizeFDiagCursor
                                           : Qt::SizeBDiagCursor;
    }
    setCursor(cursor);
}
void PointCloudCuboidView::paintEvent(QPaintEvent*) {
    QPainter p(this);
    p.fillRect(rect(), QColor("#1d1d1f"));
    p.setRenderHint(QPainter::Antialiasing);
    p.setPen(QColor("#9ca6b4"));
    p.drawText(10, 20, m_title);
    if (!m_document) return;
    const auto* b = selected();
    const auto viewRotation = cuboidRotationMatrix(m_orientation);
    const auto boxRotation =
            b ? cuboidRotationMatrix(b->rotation) : viewRotation;
    // O(n), O(1) extra storage: views borrow the same frame and draw a local
    // slice. Rasterization clips to this widget, never duplicates a point VBO.
    p.setRenderHint(QPainter::Antialiasing, false);
    int visibleCursor = 0;
    const auto& points = m_document->frame().points;
    for (int pointIndex = 0; pointIndex < points.size(); ++pointIndex) {
        if (m_visiblePointIndices) {
            while (visibleCursor < m_visiblePointIndices->size() &&
                   m_visiblePointIndices->at(visibleCursor) < pointIndex)
                ++visibleCursor;
            if (visibleCursor >= m_visiblePointIndices->size() ||
                m_visiblePointIndices->at(visibleCursor) != pointIndex)
                continue;
            ++visibleCursor;
        }
        const auto& point = points[pointIndex];
        const QColor color = m_pointColorProvider
                                     ? m_pointColorProvider(pointIndex)
                                     : QColor(QStringLiteral("#809bb0"));
        const qreal devicePixelRatio = devicePixelRatioF();
        const qreal pixelSize =
                std::max(1.0, std::nearbyint(0.25 * m_pointSize *
                                             devicePixelRatio)) /
                devicePixelRatio;
        const CuboidVector world{{point.x(), point.y(), point.z()}};
        if (b && m_depthCropEnabled && !m_creating && !(m_h == 0 && m_v == 1)) {
            const int depth = 3 - m_h - m_v;
            double depthOffset = 0;
            CuboidVector forward{};
            for (int axis = 0; axis < 3; ++axis) {
                forward[axis] = viewRotation[axis][depth];
                depthOffset += (world[axis] - b->center[axis]) * forward[axis];
            }
            double halfDepth = 0;
            for (int boxAxis = 0; boxAxis < 3; ++boxAxis) {
                double alignment = 0;
                for (int worldAxis = 0; worldAxis < 3; ++worldAxis)
                    alignment += boxRotation[worldAxis][boxAxis] *
                                 forward[worldAxis];
                halfDepth += std::abs(alignment) * b->size[boxAxis] / 2;
            }
            if (std::abs(depthOffset) > halfDepth + 0.1) continue;
        }
        CuboidVector local{};
        for (int i = 0; i < 3; ++i) local[i] = world[i] - m_origin[i];
        local = cuboidTransform(viewRotation, local, true);
        const auto q = QPointF(width() / 2.0 + local[m_h] * m_scale,
                               height() / 2.0 - local[m_v] * m_scale) +
                       m_pan;
        if (rect().contains(q.toPoint()))
            p.fillRect(QRectF(q.x() - pixelSize / 2, q.y() - pixelSize / 2,
                              pixelSize, pixelSize),
                       color);
    }
    p.setRenderHint(QPainter::Antialiasing, true);
    if (b) {
        const auto r = boxRect();
        p.setPen(QPen(Qt::white, 1));
        p.setBrush(QColor(75, 169, 255, 20));
        const auto corners = b->corners();
        const int edges[12][2] = {{0, 1}, {1, 2}, {2, 3}, {3, 0},
                                  {4, 5}, {5, 6}, {6, 7}, {7, 4},
                                  {0, 4}, {1, 5}, {2, 6}, {3, 7}};
        for (const auto& edge : edges)
            p.drawLine(project(corners[edge[0]]), project(corners[edge[1]]));
        if (!b->locked) {
            for (int i = 0; i < 8; ++i) {
                const bool active = m_hoverHandle == i;
                const double radius = active ? 4 : 3;
                p.setBrush(active ? Qt::white : QColor("#4ba9ff"));
                p.drawRect(QRectF(handlePosition(i) - QPointF(radius, radius),
                                  QSizeF(2 * radius, 2 * radius)));
            }
            p.setPen(m_hoverHandle == 8 ? Qt::white : QColor("#4ba9ff"));
            p.setBrush(Qt::NoBrush);
            p.drawLine(QPointF(r.center().x(), r.top()), handlePosition(8));
            p.drawEllipse(handlePosition(8), 5, 5);
        }
        p.setPen(QColor("#d4dae3"));
        p.drawText(10, height() - 10,
                   QStringLiteral("#%1  %2 × %3")
                           .arg(b->id)
                           .arg(b->size[m_h], 0, 'f', 2)
                           .arg(b->size[m_v], 0, 'f', 2));
    }
    if (m_creating && m_dragging) {
        p.setPen(QPen(QColor("#4ba9ff"), 1, Qt::DashLine));
        p.setBrush(Qt::NoBrush);
        p.drawRect(QRectF(m_start, m_last).normalized());
    }
}
void PointCloudCuboidView::mousePressEvent(QMouseEvent* e) {
    if (m_dragging || m_panning) return;
    m_focusAnimation->stop();
    setFocus();
    m_start = m_last = e->localPos();
    if (e->button() == Qt::MiddleButton ||
        (e->button() == Qt::LeftButton &&
         e->modifiers().testFlag(Qt::ControlModifier))) {
        m_panning = true;
        m_gestureButton = e->button();
        return;
    }
    if (e->button() != Qt::LeftButton && e->button() != Qt::RightButton) return;
    if (!m_document || m_document->frame().points.isEmpty()) return;
    if (m_creating) {
        if (e->button() != Qt::LeftButton) return;
        m_dragging = true;
        m_gestureButton = e->button();
        return;
    }
    const auto* b = selected();
    // In the orthographic editor, a click outside the selected box or on a
    // locked box is ignored. Navigation uses the explicit middle/Ctrl-left
    // path above, matching upstream's locked-shape guard.
    if (!b || b->locked) return;
    m_before = m_preview = *b;
    m_signs = {};
    m_rotating = false;
    const auto r = boxRect();
    const int handle = hitHandle(m_start);
    if (handle == 8)
        m_rotating = true;
    else if (handle >= 0) {
        m_signs[m_h] = kHandleSigns[handle][0];
        m_signs[m_v] = kHandleSigns[handle][1];
    } else if (!r.contains(m_start))
        return;
    m_dragging = true;
    m_gestureButton = e->button();
    updateHandleHover(handle);
}
void PointCloudCuboidView::mouseMoveEvent(QMouseEvent* e) {
    const auto current = e->localPos();
    if (m_panning)
        m_pan += current - m_last;
    else if (m_dragging && !m_creating) {
        if (!QRectF(rect()).contains(current) ||
            !e->buttons().testFlag(m_gestureButton)) {
            finishEdit();
            return;
        }
        updateDrag(current, e->modifiers());
    } else if (!m_dragging) {
        updateHandleHover(hitHandle(current));
    }
    m_last = current;
    update();
}
void PointCloudCuboidView::updateDrag(const QPointF& current,
                                      Qt::KeyboardModifiers modifiers) {
    const auto delta = localDelta(current - m_start);
    if (m_rotating) {
        const auto c = project(m_before.center);
        double angle = std::atan2(-(current.y() - c.y()), current.x() - c.x()) -
                       std::atan2(-(m_start.y() - c.y()), m_start.x() - c.x());
        const int axis = 3 - m_h - m_v;
        // (X,Y), (Y,Z), (X,Z) have respectively +Z,+X,-Y normals.
        if (m_h == 0 && m_v == 2) angle = -angle;
        if (modifiers.testFlag(Qt::ShiftModifier)) {
            const double step = std::acos(-1.0) / 12;
            // Match Python round's ties-to-even rule, independently of
            // the process floating-point rounding mode.
            const double units = angle / step;
            const double lower = std::floor(units);
            const double fraction = units - lower;
            const double snapped = fraction < .5   ? lower
                                   : fraction > .5 ? lower + 1
                                   : std::fmod(std::abs(lower), 2) == 0
                                           ? lower
                                           : lower + 1;
            angle = snapped * step;
        }
        m_preview = m_before.rotated(axis, angle);
    } else if (m_signs[m_h] || m_signs[m_v])
        m_preview = m_before.resized(m_signs, delta);
    else {
        m_preview = m_before;
        const auto world =
                cuboidTransform(cuboidRotationMatrix(m_before.rotation), delta);
        for (int i = 0; i < 3; ++i) m_preview.center[i] += world[i];
    }
    if (onEditPreview) onEditPreview(&m_preview);
}
void PointCloudCuboidView::finishEdit() {
    if (!m_dragging || m_creating) return;
    const auto box = m_preview;
    m_dragging = false;
    m_gestureButton = Qt::NoButton;
    updateHandleHover(-1);
    update();
    // Clear the gesture before calling the owner, which may setDocument.
    if (box != m_before && onEdited) onEdited(box);
    if (onEditPreview) onEditPreview(nullptr);
}
void PointCloudCuboidView::mouseReleaseEvent(QMouseEvent* e) {
    if (e->button() != m_gestureButton) return;
    if (m_panning) {
        m_pan += e->localPos() - m_last;
        m_panning = false;
        m_gestureButton = Qt::NoButton;
        update();
        return;
    }
    if (!m_dragging) return;
    if (m_creating) {
        const QRectF rect(m_start, e->localPos());
        if (QLineF(m_start, e->localPos()).length() >= 5 && onCreated) {
            CuboidVector center{}, size{{1.6, 1.6, 1.6}};
            center = localDelta(rect.center() - boxCenter());
            const auto originLocal = cuboidTransform(
                    cuboidRotationMatrix(m_orientation), m_origin, true);
            for (int i = 0; i < 3; ++i) center[i] += originLocal[i];
            size[m_h] = std::max(0.01, rect.normalized().width() / m_scale);
            size[m_v] = std::max(0.01, rect.normalized().height() / m_scale);
            const int depth = 3 - m_h - m_v;
            double low = std::numeric_limits<double>::infinity(), high = -low;
            if (m_document)
                for (const auto& point : m_document->frame().points) {
                    const CuboidVector world{{point.x(), point.y(), point.z()}};
                    if (!rect.normalized().contains(project(world))) continue;
                    const auto local = cuboidTransform(
                            cuboidRotationMatrix(m_orientation), world, true);
                    low = std::min(low, local[depth]);
                    high = std::max(high, local[depth]);
                }
            if (std::isfinite(low)) {
                center[depth] = (low + high) / 2;
                size[depth] = std::max(0.01, high - low);
            }
            center = cuboidTransform(cuboidRotationMatrix(m_orientation),
                                     center);
            m_dragging = false;
            onCreated(center, size);
        }
    } else {
        // A release can contain a newer position than the last MouseMove.
        if (QRectF(rect()).contains(e->localPos()))
            updateDrag(e->localPos(), e->modifiers());
        finishEdit();
        return;
    }
    m_dragging = false;
    m_panning = false;
    m_gestureButton = Qt::NoButton;
    update();
}
void PointCloudCuboidView::leaveEvent(QEvent* e) {
    m_hovered = false;
    m_expandButton->setVisible(m_expanded);
    finishEdit();
    updateHandleHover(-1);
    QWidget::leaveEvent(e);
}
bool PointCloudCuboidView::event(QEvent* e) {
    if (e->type() == QEvent::Enter) {
        m_hovered = true;
        m_expandButton->show();
        m_expandButton->raise();
    }
    return QWidget::event(e);
}
void PointCloudCuboidView::wheelEvent(QWheelEvent* e) {
    m_focusAnimation->stop();
    if (m_dragging) {
        e->accept();
        return;
    }
    if (const auto* box = selected()) {
        m_origin = box->center;
        m_orientation = box->rotation;
        m_pan = {};
    }
    const double h = std::max(1, height());
    const double scale = std::clamp(
            h / (2 * m_scale) * std::exp(-e->angleDelta().y() / 900.0), .025,
            50.0);
    m_scale = h / (2 * scale);
    update();
    e->accept();
}
void PointCloudCuboidView::keyPressEvent(QKeyEvent* e) {
    if (e->key() == Qt::Key_Escape) {
        m_dragging = false;
        setCreating(false);
        if (onCancelled) onCancelled();
        return;
    }
    const auto* b = selected();
    if (b && !b->locked && onEdited && !m_dragging && !m_creating &&
        e->modifiers() == Qt::NoModifier) {
        CuboidVector delta{};
        const double step = 0.1;
        if (e->key() == Qt::Key_Left)
            delta[m_h] = -step;
        else if (e->key() == Qt::Key_Right)
            delta[m_h] = step;
        else if (e->key() == Qt::Key_Up)
            delta[m_v] = step;
        else if (e->key() == Qt::Key_Down)
            delta[m_v] = -step;
        else {
            QWidget::keyPressEvent(e);
            return;
        }
        auto box = *b;
        const auto world =
                cuboidTransform(cuboidRotationMatrix(box.rotation), delta);
        for (int i = 0; i < 3; ++i) box.center[i] += world[i];
        onEdited(box);
        return;
    }
    QWidget::keyPressEvent(e);
}
void PointCloudCuboidView::resizeEvent(QResizeEvent* e) {
    m_dragging = false;
    m_panning = false;
    m_gestureButton = Qt::NoButton;
    if (onEditPreview) onEditPreview(nullptr);
    updateHandleHover(-1);
    QWidget::resizeEvent(e);
    m_expandButton->move(width() - 32, 6);
    if (selected()) focusCuboid();
}
