// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QCache>
#include <QImage>
#include <QLineF>
#include <QPoint>
#include <QSet>
#include <QWidget>

#include "PointCloudCamera.h"
#include "PointCloudColors.h"
#include "PointCloudDocument.h"

class QButtonGroup;
class QToolButton;
class QSlider;
class QGraphicsView;
class QGraphicsItem;
class QPainter;
class QLabel;
class QMenu;
class QSettings;

class PointCloudCameraPanel final : public QWidget {
    Q_OBJECT

public:
    explicit PointCloudCameraPanel(QWidget* parent = nullptr);

    void setSource(const PointCloudCameraSource& source);
    void setSources(const QVector<PointCloudCameraSource>& sources);
    const QVector<PointCloudCameraSource>& sources() const { return m_sources; }
    int activeCameraIndex() const { return m_activeIndex; }
    void setActiveCameraIndex(int index);
    void stepCamera(int delta);
    void fitCameraImage();
    void attachToViewport(QWidget* viewport);
    void resizeImageRegion(const QSize& size);
    void clearSource();
    void setFrame(const PointCloudFrame* frame);
    void setOverlayEnabled(bool enabled);
    void setVisibility(const QSet<quint16>& hiddenSemantics,
                       const QSet<quint16>& hiddenInstances,
                       const QSet<int>& hiddenCuboids);
    void setCuboidStyle(const QHash<int, QColor>& colors,
                        int selectedId,
                        bool detection);
    void setPointColorStyle(PointCloudColorMode mode,
                            const QHash<quint16, QColor>& colors);
    void setCreationPreview(const PointCloudCuboid* preview);
    void setEditPreview(const PointCloudCuboid* preview);
    const PointCloudCuboid* editPreview() const {
        return m_hasEditPreview ? &m_editPreview : nullptr;
    }

    bool hasImage() const { return !m_image.isNull(); }
    QString imagePath() const { return m_imagePath; }
    bool overlayEnabled() const { return m_overlayEnabled; }
    bool projectionEnabled() const { return m_projectionEnabled; }
    QString statusMessage() const { return m_statusMessage; }

protected:
    void paintEvent(QPaintEvent* event) override;
    void resizeEvent(QResizeEvent* event) override;
    void hideEvent(QHideEvent* event) override;
    bool eventFilter(QObject* watched, QEvent* event) override;

private:
    void clearImage();
    void refreshFrame();
    bool refreshCuboidProjection();
    void refreshPointOverlay();
    void paintCameraImage(QPainter* painter);
    void positionPanel();

    QVector<PointCloudCameraSource> m_sources;
    int m_activeIndex = -1;
    QLabel* m_cameraName = nullptr;
    QWidget* m_overlayHost = nullptr;
    QWidget* m_resizeHandle = nullptr;
    QSize m_userSize;
    QWidget* m_cameraControls = nullptr;
    QGraphicsView* m_cameraView = nullptr;
    QGraphicsItem* m_imageItem = nullptr;
    QLabel* m_message = nullptr;
    QToolButton* m_pointsToggle = nullptr;
    QToolButton* m_cuboidsToggle = nullptr;
    QButtonGroup* m_overlayColor = nullptr;
    QMenu* m_overlaySettings = nullptr;
    QSettings* m_settings = nullptr;
    QLabel* m_depthLegend = nullptr;
    QSlider* m_overlaySize = nullptr;
    QSlider* m_overlayOpacity = nullptr;
    PointCloudFrame m_frame;
    bool m_hasFrame = false;
    PointCloudCameraSource m_source;
    bool m_hasSource = false;
    bool m_overlayEnabled = true;
    bool m_projectionEnabled = false;
    QImage m_image;
    QCache<QString, QImage> m_images{4};
    QString m_imagePath;
    QString m_statusMessage;
    QVector<QPoint> m_projectedPoints;
    QVector<int> m_pointIndices;
    QVector<QColor> m_depthColors;
    PointCloudColorMode m_pointColorMode = PointCloudColorMode::Semantic;
    QHash<quint16, QColor> m_pointColors;
    QImage m_pointOverlay;
    struct ProjectedCuboid {
        int id, classId;
        bool occluded;
        QVector<QLineF> lines;
    };
    QVector<ProjectedCuboid> m_projectedCuboids;
    QVector<QLineF> m_projectedPreview;
    PointCloudCuboid m_creationPreview;
    bool m_hasCreationPreview = false;
    PointCloudCuboid m_editPreview;
    bool m_hasEditPreview = false;
    QSet<quint16> m_hiddenSemantics, m_hiddenInstances;
    QSet<int> m_hiddenCuboids;
    QHash<int, QColor> m_classColors;
    int m_selectedCuboid = 0;
    bool m_detection = true;
};
