// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QColor>
#include <QHash>
#include <QImage>
#include <QJsonArray>
#include <QJsonObject>
#include <QPointF>
#include <QRectF>
#include <QSet>
#include <QVector>
#include <QWidget>
#include <algorithm>
#include <cstdint>

#include "AnnotationCuboid.h"
#include "RelateOptions.h"

struct AnnotationShape {
    QString label;
    QString type;
    QVector<QPointF> points;
    QImage mask;
    QJsonObject extra;
    bool visible = true;
    bool locked = false;
    // Display cache: the source paints masks as a binary silhouette filled
    // with the shape color at mask_opacity; the tinted image is rebuilt
    // only when the mask or color changes. It is display-only and never
    // serialized.
    mutable QImage displaySilhouette;
    mutable quint64 displaySilhouetteSource = 0;
    mutable QColor displaySilhouetteColor;

    static AnnotationShape fromJson(const QJsonObject& object);
    QJsonObject toJson() const;
};

QColor annotationShapeColor(const QString& label, const QJsonObject& extra);
QColor annotationLabelColor(int labelId);

struct AnnotationPrompt {
    QVector<QPointF> positive;
    QVector<QPointF> negative;
    QRectF box;
    bool useBox = false;
    QString label;
    QString text;
    QString textModelPath;
    double iouThreshold = 0.7;
    // YOLO-World -> RAM: tag each detected crop into its object description.
    QString objectTaggingModelPath;
    // RAM's output configuration: filtering uses English model labels before
    // translation, matching the upstream adapter's delete/filter precedence.
    QString ramTagMode = QStringLiteral("en");
    QStringList ramDeleteTags;
    QStringList ramFilterTags;
    // Explicit optional OCR layout head. Cache contents never enable stages.
    QString ocrLayoutModelPath;
    QString ocrClassifierModelPath;
    // Original selected text polygons, not their axis-aligned bounding boxes.
    QVector<QVector<QPointF>> ocrQuads;
    // Selected box proposals for GKD or SAM3 in source-image pixels.
    QVector<QRectF> rois;
    QVector<int> roiGroupIds;
    QVector<int> roiSourceIndices;
    QVector<QString> roiLabels;
    // YOLOE SAVPE visual prompts in original-image pixel coordinates.
    QVector<QRectF> visualPrompts;
    // GKD 1-shot support prompt. The image is borrowed by the worker only
    // for the duration of the AICore call; points use its pixel coordinates.
    QImage visualSupportImage;
    QVector<QPointF> visualSupportPoints;

    // RelateAnything: thresholds, edge caps, custom predicates and the
    // detector stage used when fewer than two rectangles are selected.
    RelateOptions relate;
    // Detector (YOLO / RF-DETR GGUF) for that stage; empty = none, in which
    // case at least two rectangles must be selected.
    QString relateDetectorPath;
    QStringList relateDetectorClasses;  // vocabulary detectors only
    QString relateDetectorTextModel;
    QString relateStudentPath;
    QString relateTokenizerPath;

    // SAHI-compatible tiled detection options. These stay in the plugin
    // prompt because tiling is orchestration around the existing AICore
    // detector, not a new inference ABI.
    bool slicedDetection = false;
    int sliceWidth = 512;
    int sliceHeight = 512;
    double sliceOverlap = 0.2;
    double sliceMergeThreshold = 0.45;

    bool isEmpty() const {
        return positive.isEmpty() && negative.isEmpty() && !useBox &&
               text.isEmpty() && rois.isEmpty() && visualPrompts.isEmpty() &&
               visualSupportPoints.isEmpty();
    }
};

Q_DECLARE_METATYPE(AnnotationPrompt)

class AnnotationCanvas final : public QWidget {
    Q_OBJECT
public:
    explicit AnnotationCanvas(QWidget* parent = nullptr);
    void setImage(const QImage& image);
    void setCompareImage(const QImage& image);
    void clearCompareImage();
    void setCompareSplit(double split);
    void setShapeOpacity(int value);
    int shapeOpacity() const { return m_shapeOpacity; }
    void setMaskOpacity(int value);
    int maskOpacity() const { return m_maskOpacity; }
    void setMagicWandOptions(int threshold,
                             double dragSensitivity,
                             double luminanceWeight);
    void setMagicWandOpacity(double opacity);
    void setMagicWandSimplifyEpsilon(double epsilon);
    void setShapeStyle(int pointSize, double lineWidth);
    void setHitEpsilon(double epsilon);
    void setCrosshair(bool visible,
                      double width,
                      const QColor& color,
                      double opacity);
    void setDoubleClickClose(bool enabled) { m_doubleClickClose = enabled; }
    void setDoubleClickEditLabel(bool enabled) {
        m_doubleClickEditLabel = enabled;
    }
    void setWheelRectangleEditing(bool enabled,
                                  double adjustStep,
                                  double scaleStep);
    void setBrushPointDistance(double distance);
    void setBrushHistoryLimits(int maxSteps, int maxMemoryMb);
    void setRotationIncrements(double largeDegrees, double smallDegrees);
    void setCuboidDefaults(const QPointF& depthVector, double minDepth);
    /** Shape color styling, matching the source shape.* settings group. */
    enum class ShapeColorMode { Auto, Manual };
    void setShapeColorMode(ShapeColorMode mode);
    void setDefaultShapeColor(const QColor& color);
    void setShiftAutoShapeColor(int shift);
    void setLabelOrder(const QStringList& labels);
    QColor shapeColor(const AnnotationShape& shape) const {
        return effectiveLineColor(shape, false);
    }
    void setShapeStyleColors(const QColor& line,
                             const QColor& fill,
                             const QColor& vertex,
                             const QColor& selectLine,
                             const QColor& selectFill,
                             const QColor& hoverVertex);
    void setBrightness(int value);
    void setContrast(int value);
    void setBrightnessContrast(int brightness, int contrast);
    int brightness() const { return m_brightness; }
    int contrast() const { return m_contrast; }
    const QImage& displayImage() const { return m_displayImage; }
    void setShapes(const QVector<AnnotationShape>& shapes);
    /** Transient RelateAnything overlay. Each entry carries subject_index /
     *  object_index (canvas shape indices), predicate, score and type. It is
     *  display-only and cleared whenever the shapes or image change, because
     *  the indices would otherwise go stale. */
    void setRelations(const QJsonArray& relations);
    void clearRelations();
    void setHighlightedRelation(int row);
    // Render the current source image and annotation overlays at source
    // resolution for export. Selection handles and interactive prompts are
    // intentionally excluded.
    QImage renderVisualizationImage(
            const QVector<AnnotationShape>& shapes) const;
    struct VisualizationOptions {
        bool showLabels, showScores, showGroups, showTexts, showMasks;
        bool showShapes, showLinking, showDegrees, showAttributes;
    };
    VisualizationOptions visualizationOptions() const;
    QImage renderVisualizationImage(const QImage& source,
                                    const QVector<AnnotationShape>& shapes,
                                    const VisualizationOptions& options) const;
    void setShapeDescription(int index, const QString& description);
    void setShapeRecognition(int index,
                             const QString& description,
                             double score,
                             const QString& language);
    const QVector<AnnotationShape>& shapes() const { return m_shapes; }
    void setTool(const QString& type);
    QString tool() const { return m_tool; }
    bool creating() const { return m_creating; }
    int selectedIndex() const { return m_selected; }
    const QSet<int>& selectedIndices() const { return m_selectedIndices; }
    void selectShape(int index);
    void setSelectedShapes(const QVector<int>& indices);
    void setCurrentLabel(const QString& label) { m_label = label; }
    QString currentLabel() const { return m_label; }
    AnnotationPrompt prompt() const;
    void clearPrompt();
    void undoLastPoint();
    void addPointToEdge();
    void removeSelectedPoint();
    void removeLastPromptPoint(bool negative);
    void deleteSelected();
    void discardLastShape();
    void fitImage();
    void fitWidth();
    void zoomOriginal();
    void zoomBy(double factor);
    double zoomFactor() const { return m_zoom; }
    void setZoomFactor(double factor);
    /** Zoom so imageRect fills 1/widthDivisor of the viewport width, then
     *  center it; matches the source label-review zoom formula. */
    void zoomToRect(const QRectF& imageRect, double widthDivisor);
    void setFillDrawing(bool enabled);
    bool fillDrawing() const { return m_fillDrawing; }
    void setShowShapes(bool visible);
    bool showShapes() const { return m_showShapes; }
    void setShowMasks(bool visible);
    bool showMasks() const { return m_showMasks; }
    void setShowLabels(bool visible);
    bool showLabels() const { return m_showLabels; }
    void setShowScores(bool visible);
    bool showScores() const { return m_showScores; }
    void setShowGroups(bool visible);
    bool showGroups() const { return m_showGroups; }
    void setShowTexts(bool visible);
    bool showTexts() const { return m_showTexts; }
    void setShowAttributes(bool visible);
    bool showAttributes() const { return m_showAttributes; }
    void setShowDegrees(bool visible);
    bool showDegrees() const { return m_showDegrees; }
    void setShowLinking(bool visible);
    bool showLinking() const { return m_showLinking; }
    QRectF viewportImageRect();
    void navigateTo(const QPointF& imageRatio);
    bool brushEditing() const { return m_brushEditing; }
    bool setBrushEditing(bool enabled, bool cancel = false);
    void brushUndo();
    void brushRedo();
    void setBrushRadius(int radius);
    void setBrushErase(bool erase) { setBrushOperation(erase ? 1 : 0); }
    // 0 = paint, 1 = erase, 2 = invert the mask under the brush.
    void setBrushOperation(int operation) {
        const int clamped = std::max(0, std::min(2, operation));
        if (m_brushOperation == clamped) return;
        m_brushOperation = clamped;
        update();
        emit brushOperationChanged(m_brushOperation);
    }
    void setBrushLabel(const QString& label) { m_brushLabel = label; }
    void setBrushClassId(int id) { m_brushClassId = id; }
    QSize minimumSizeHint() const override { return QSize(440, 320); }

signals:
    void editStarted();
    void shapesChanged();
    void shapeSelected(int index);
    /** Hover moved onto/out of a shape vertex, matching the source
     *  canvas.vertex_selected signal used to enable point deletion. */
    void vertexHovered(bool hovered);
    void shapeFinished(int index);
    void shapeDoubleClicked(int index);
    void creationCanceled();
    void brushEditingChanged(bool enabled);
    void brushHistoryChanged(bool canUndo, bool canRedo);
    void brushRadiusChanged(int radius);
    void brushOperationChanged(int operation);
    void promptChanged();
    void viewChanged();
    void displayAdjustmentChanged();

protected:
    void paintEvent(QPaintEvent*) override;
    QColor effectiveLineColor(const AnnotationShape& shape,
                              bool selected) const;
    /** Source mask compositing: a binary silhouette filled with the shape
     *  color, cached for shape masks. */
    QImage shapeSilhouette(const AnnotationShape& shape,
                           const QColor& color) const;
    QImage tintedSilhouette(const QImage& mask,
                            const QColor& color,
                            QImage& cache,
                            quint64& sourceKey,
                            QColor& cacheColor) const;
    // Scratch cache for masks that are not shape-owned (brush previews).
    mutable QImage m_brushSilhouetteCache;
    mutable quint64 m_brushSilhouetteKey = 0;
    mutable QColor m_brushSilhouetteColor;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;
    void mouseDoubleClickEvent(QMouseEvent* event) override;
    void wheelEvent(QWheelEvent* event) override;
    void keyPressEvent(QKeyEvent* event) override;
    void keyReleaseEvent(QKeyEvent* event) override;

private:
    QPointF toImage(const QPointF& point) const;
    QPointF clamp(const QPointF& point) const;
    void updateTransform();
    void finishShape();
    void cancelShape();
    int hitShape(const QPointF& point,
                 int* vertex,
                 annotationCuboid::Face* face = nullptr) const;
    int hitGroup(const QPointF& point) const;
    QPointF rotationHandle(const AnnotationShape& shape) const;
    void moveVertex(AnnotationShape& shape, int vertex, const QPointF& point);
    void startMagicWand(const QPointF& imagePoint, const QPointF& screenPoint);
    void updateMagicWand(int threshold);
    void clearMagicWand();
    void finishMagicWand();
    void paintBrush(const QPointF& point, int operation);
    int effectiveBrushOperation(Qt::KeyboardModifiers modifiers) const;
    void finishBrushStroke();
    void rotateSelectedBy(double degrees);

    QImage m_image;
    QImage m_compareImage;
    QVector<AnnotationShape> m_shapes;
    QJsonArray m_relations;
    int m_highlightedRelation = -1;
    QString m_tool = QStringLiteral("edit");
    QString m_label;
    int m_selected = -1;
    QSet<int> m_selectedIndices;
    int m_dragVertex = -1;
    annotationCuboid::Face m_dragFace = annotationCuboid::Face::None;
    bool m_dragging = false;
    bool m_dragChanged = false;
    QVector<int> m_dragIndices;
    QVector<QVector<QPointF>> m_dragOriginalPoints;
    QVector<QImage> m_dragOriginalMasks;
    QPointF m_dragStartPoint;
    bool m_rotating = false;
    bool m_panning = false;
    bool m_spaceHeld = false;
    bool m_creating = false;
    AnnotationPrompt m_prompt;
    bool m_promptBoxDragging = false;
    QPointF m_promptBoxStart;
    bool m_magicActive = false;
    QPointF m_magicAnchor;
    QVector<float> m_magicDistances;
    QVector<uint8_t> m_magicMask;
    QVector<QPointF> m_magicPolygon;
    int m_magicThreshold = 15;
    int m_magicDefaultThreshold = 15;
    double m_magicDragSensitivity = 3.0;
    double m_magicLuminanceWeight = 0.5;
    double m_magicOpacity = 0.6;
    double m_magicSimplifyEpsilon = 0.5;
    int m_shapePointSize = 10;
    double m_shapeLineWidth = 4.0;
    double m_hitEpsilon = 10.0;
    bool m_crosshairVisible = true;
    double m_crosshairWidth = 2.0;
    QColor m_crosshairColor = QColor(QStringLiteral("#00FF00"));
    ShapeColorMode m_shapeColorMode = ShapeColorMode::Auto;
    QColor m_defaultShapeColor = QColor(0, 255, 0);
    int m_shiftAutoShapeColor = 0;
    QHash<QString, int> m_labelColorIds;
    QColor m_lineColor = QColor(0, 255, 0, 128);
    QColor m_fillColor = QColor(220, 220, 220, 150);
    QColor m_vertexFillColor = QColor(0, 255, 0, 255);
    QColor m_selectLineColor = QColor(255, 255, 255, 255);
    QColor m_selectFillColor = QColor(0, 255, 0, 155);
    QColor m_hoverVertexFillColor = QColor(255, 255, 255, 255);
    int m_hoverVertex = -1;
    int m_hoverShape = -1;
    double m_crosshairOpacity = 0.5;
    bool m_doubleClickClose = true;
    bool m_doubleClickEditLabel = true;
    bool m_wheelRectangleEditing = true;
    double m_wheelAdjustStep = 2.0;
    double m_wheelScaleStep = 0.05;
    double m_brushPointDistance = 25.0;
    int m_brushMaxUndoSteps = 30;
    int m_brushMaxUndoMemoryMb = 128;
    double m_rotationLargeIncrement = 1.0;
    double m_rotationSmallIncrement = 0.1;
    QPointF m_cuboidDepthVector = QPointF(24.0, -24.0);
    double m_cuboidMinDepth = 5.0;
    bool m_brushEditing = false;
    bool m_brushStrokeActive = false;
    int m_brushTarget = -1;
    int m_brushRadius = 12;
    int m_brushOperation = 0;
    int m_brushStrokeOperation = 0;
    QImage m_brushStrokeBaseline;
    QImage m_brushStrokeCoverage;
    QString m_brushLabel;
    int m_brushClassId = -1;
    QPointF m_brushLast;
    QImage m_brushMask;
    QVector<QImage> m_brushHistory;
    QVector<QImage> m_brushRedo;
    QPointF m_start;
    QPointF m_last;
    QPointF m_hover;
    QPointF m_pan;
    QPointF m_panStart;
    double m_rotationStart = 0.0;
    QVector<QPointF> m_rotationPoints;
    double m_zoom = 1.0;
    bool m_fillDrawing = true;
    double m_compareSplit = 0.5;
    double m_scale = 1.0;
    QPointF m_origin;
    bool m_showShapes = true;
    bool m_showMasks = true;
    bool m_showLabels = true;
    bool m_showScores = true;
    bool m_showGroups = false;
    bool m_showTexts = true;
    bool m_showAttributes = true;
    bool m_showDegrees = false;
    bool m_showLinking = true;
    int m_shapeOpacity = 100;
    int m_maskOpacity = 80;
    int m_brightness = 50;
    int m_contrast = 50;
    QImage m_displayImage;

    void rebuildDisplayImage();
};
