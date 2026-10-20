// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once
#include <QColor>
#include <QPointF>
#include <QWidget>
#include <functional>

#include "PointCloudDocument.h"

class QVariantAnimation;
class QToolButton;

// Orthographic local-axis editor. Borrows the current frame; no point copy.
class PointCloudCuboidView final : public QWidget {
public:
    explicit PointCloudCuboidView(int horizontal,
                                  int vertical,
                                  const QString& title,
                                  QWidget* parent = nullptr);
    std::function<void(const PointCloudCuboid&)> onEdited;
    std::function<void(const PointCloudCuboid*)> onEditPreview;
    std::function<void(const CuboidVector&, const CuboidVector&)> onCreated;
    std::function<void()> onCancelled;
    std::function<void()> onExpandRequested;
    void setDocument(const PointCloudDocument* document, int selectedId);
    void setVisiblePointIndices(const QVector<int>* indices);
    void setPointColorProvider(std::function<QColor(int)> provider);
    void setPointSize(int size);
    int pointSize() const { return m_pointSize; }
    void setDepthCropEnabled(bool enabled);
    bool depthCropEnabled() const { return m_depthCropEnabled; }
    void setExpanded(bool expanded);
    bool isExpanded() const { return m_expanded; }
    void setEditPreview(const PointCloudCuboid* box, bool followTranslation);
    const PointCloudCuboid* editPreview() const {
        return m_hasEditPreview ? &m_editPreview : nullptr;
    }
    void setCreating(bool creating);
    void focusCuboid(bool animate = false);

protected:
    bool event(QEvent*) override;
    void paintEvent(QPaintEvent*) override;
    void mousePressEvent(QMouseEvent*) override;
    void mouseMoveEvent(QMouseEvent*) override;
    void mouseReleaseEvent(QMouseEvent*) override;
    void leaveEvent(QEvent*) override;
    void wheelEvent(QWheelEvent*) override;
    void keyPressEvent(QKeyEvent*) override;
    void resizeEvent(QResizeEvent*) override;

private:
    const PointCloudCuboid* selected() const;
    QPointF project(const CuboidVector& world) const;
    CuboidVector localDelta(const QPointF& screen) const;
    QPointF boxCenter() const;
    QRectF boxRect() const;
    QPointF handlePosition(int index) const;
    int hitHandle(const QPointF& point) const;
    void updateHandleHover(int index);
    void updateDrag(const QPointF& point, Qt::KeyboardModifiers modifiers);
    void finishEdit();
    const PointCloudDocument* m_document = nullptr;
    int m_selected = 0;
    int m_h, m_v;
    QString m_title;
    double m_scale = 35;
    CuboidVector m_origin{};
    CuboidVector m_orientation{};
    QString m_framePath;
    QPointF m_pan;
    QPointF m_start, m_last;
    bool m_creating = false;
    bool m_depthCropEnabled = true;
    bool m_expanded = false;
    bool m_hovered = false;
    int m_pointSize = 2;
    const QVector<int>* m_visiblePointIndices = nullptr;
    std::function<QColor(int)> m_pointColorProvider;
    bool m_dragging = false;
    bool m_panning = false;
    bool m_rotating = false;
    Qt::MouseButton m_gestureButton = Qt::NoButton;
    int m_hoverHandle = -1;
    std::array<int, 3> m_signs{};
    PointCloudCuboid m_before, m_preview;
    PointCloudCuboid m_editPreview;
    bool m_hasEditPreview = false;
    QVariantAnimation* m_focusAnimation = nullptr;
    QToolButton* m_expandButton = nullptr;
};
