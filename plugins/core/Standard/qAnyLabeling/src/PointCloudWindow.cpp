// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "PointCloudWindow.h"

#include <QAbstractItemView>
#include <QActionGroup>
#include <QApplication>
#include <QCache>
#include <QCheckBox>
#include <QCloseEvent>
#include <QColorDialog>
#include <QComboBox>
#include <QCryptographicHash>
#include <QDateTime>
#include <QDialog>
#include <QDialogButtonBox>
#include <QDir>
#include <QDoubleSpinBox>
#include <QEventLoop>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QFutureWatcher>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QHash>
#include <QImageReader>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonParseError>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QMatrix4x4>
#include <QMenuBar>
#include <QMessageBox>
#include <QMouseEvent>
#include <QOpenGLBuffer>
#include <QOpenGLFramebufferObject>
#include <QOpenGLFunctions_3_3_Core>
#include <QOpenGLShaderProgram>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLWidget>
#include <QPainter>
#include <QPen>
#include <QPolygonF>
#include <QProgressDialog>
#include <QPushButton>
#include <QRegularExpression>
#include <QSaveFile>
#include <QScopedValueRollback>
#include <QScrollArea>
#include <QSet>
#include <QSignalBlocker>
#include <QSpinBox>
#include <QSplitter>
#include <QStandardItemModel>
#include <QStatusBar>
#include <QSurfaceFormat>
#include <QThread>
#include <QTimer>
#include <QToolBar>
#include <QToolButton>
#include <QVBoxLayout>
#include <QVariantAnimation>
#include <QVector2D>
#include <QWheelEvent>
#include <QtConcurrent>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <exception>
#include <functional>
#include <limits>
#include <memory>

#include "PointCloudCameraPanel.h"
#include "PointCloudColors.h"
#include "PointCloudCuboidView.h"
#include "PointCloudDatasetExporter.h"
#include "PointCloudViewTransform.h"

namespace {
QPointF mousePosition(const QMouseEvent* event) {
#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)
    return event->position();
#else
    return event->localPos();
#endif
}

QColor idColor(quint32 id) {
    if (!id) return QColor(145, 151, 158);
    return QColor::fromHsv(int((id * 137) % 360), 170, 230);
}

bool windowFailure(QString* error, const QString& message) {
    if (error) *error = message;
    return false;
}

QVector<PointCloudDatasetImportClass> datasetClasses(
        const QMap<quint16, QString>& names,
        const QMap<quint16, QColor>& colors) {
    QVector<PointCloudDatasetImportClass> result;
    for (auto it = names.cbegin(); it != names.cend(); ++it)
        result.append({int(it.key()), it.value(),
                       colors.value(it.key(), idColor(it.key()))});
    return result;
}

bool readClassSection(const QJsonValue& value,
                      bool detection,
                      QMap<quint16, QString>* names,
                      QMap<quint16, QColor>* colors,
                      QString* error) {
    if (!value.isArray())
        return windowFailure(
                error,
                QStringLiteral("Task configuration requires a classes array."));
    names->clear();
    colors->clear();
    QSet<QString> usedNames;
    for (const auto& entry : value.toArray()) {
        const auto object = entry.toObject();
        const double id = object.value(QStringLiteral("id")).toDouble(-1);
        const QString name =
                object.value(QStringLiteral("name")).toString().trimmed();
        const QString color = object.value(QStringLiteral("color")).toString();
        if (id < (detection ? 1 : 0) || id > 65535 || std::floor(id) != id ||
            name.isEmpty() || names->contains(quint16(id)) ||
            (detection && usedNames.contains(name)) ||
            !QRegularExpression(QStringLiteral("^#[0-9a-fA-F]{6}$"))
                     .match(color)
                     .hasMatch())
            return windowFailure(
                    error,
                    QStringLiteral("Invalid or duplicate task class entry."));
        names->insert(quint16(id), name);
        colors->insert(quint16(id), QColor(color));
        usedNames.insert(name);
    }
    return detection || names->contains(0) ||
           windowFailure(error,
                         QStringLiteral("Segmentation classes require ID 0."));
}

QJsonObject classSection(const QMap<quint16, QString>& names,
                         const QMap<quint16, QColor>& colors) {
    QJsonArray entries;
    for (auto it = names.cbegin(); it != names.cend(); ++it)
        entries.append(QJsonObject{
                {QStringLiteral("id"), int(it.key())},
                {QStringLiteral("name"), it.value()},
                {QStringLiteral("color"),
                 colors.value(it.key(), idColor(it.key())).name().toUpper()}});
    return QJsonObject{{QStringLiteral("classes"), entries}};
}
}  // namespace

class PointCloudView final : public QOpenGLWidget,
                             protected QOpenGLFunctions_3_3_Core {
public:
    enum Tool { Browse, Brush, Polygon };
    using ColorMode = PointCloudColorMode;

    explicit PointCloudView(QWidget* parent = nullptr) : QOpenGLWidget(parent) {
        // The live upstream window uses CuboidViewport even for segmentation.
        m_camera.perspective = true;
        m_camera.minimumScale = 0.3 * std::tan(25 * std::acos(-1.0) / 180);
        m_camera.maximumScale = 100 * std::tan(25 * std::acos(-1.0) / 180);
        QSurfaceFormat surface = format();
        surface.setVersion(3, 3);
        surface.setProfile(QSurfaceFormat::CoreProfile);
        surface.setDepthBufferSize(24);
        surface.setSamples(4);
        setFormat(surface);
        setFocusPolicy(Qt::StrongFocus);
        m_focusAnimation.setObjectName(
                QStringLiteral("CuboidPerspectiveFocusAnimation"));
        m_focusAnimation.setEasingCurve(QEasingCurve::Linear);
        m_focusAnimation.setStartValue(0.0);
        m_focusAnimation.setEndValue(1.0);
        connect(&m_focusAnimation, &QVariantAnimation::valueChanged, this,
                [this](const QVariant& value) {
                    double progress = value.toDouble();
                    if (progress < 1)
                        progress = -std::expm1(-focusDecay() *
                                               m_focusAnimation.currentTime() /
                                               1000.0);
                    for (int axis = 0; axis < 3; ++axis)
                        m_camera.center[axis] = m_focusStart.center[axis] +
                                                (m_focusTarget.center[axis] -
                                                 m_focusStart.center[axis]) *
                                                        progress;
                    m_camera.setScale(
                            m_focusStart.scale +
                            (m_focusTarget.scale - m_focusStart.scale) *
                                    progress);
                    update();
                });
    }
    std::function<void(const QVector<int>&)> onSelection;
    std::function<void(int)> onCuboidSelected;
    std::function<void(const PointCloudCuboid&)> onCuboidEdited;
    std::function<void(const CuboidVector&, const CuboidVector&)>
            onCuboidCreated;
    std::function<void()> onCreationCancelled;
    std::function<void()> onCreationPreviewChanged;
    const PointCloudCuboid* creationPreview() const {
        return m_hasCreationPreview ? &m_creationPreview : nullptr;
    }
    void setEditPreview(const PointCloudCuboid* box) {
        m_hasEditPreview = box != nullptr;
        if (box) m_editPreview = *box;
        update();
    }
    void setCreating(bool value) {
        m_focusAnimation.stop();
        m_creating = value;
        const bool hadPreview = m_hasCreationPreview;
        m_hasCreationPreview = false;
        setMouseTracking(value);
        setCursor(value ? Qt::CrossCursor : Qt::ArrowCursor);
        update();
        if (hadPreview && onCreationPreviewChanged) onCreationPreviewChanged();
    }
    std::function<void(int)> onBrushSizeChanged;
    std::function<void(const QString&)> onSelectionError;
    bool detection = false;
    int selectedCuboid = 0;
    QSet<int> hiddenCuboids;
    QMap<quint16, QString> classNames;
    void setFrame(const PointCloudFrame* frame) {
        m_focusAnimation.stop();
        m_frame = frame;
        rebuildVisibleIndices();
        rebuildIntensityRange();
        m_hasEditPreview = false;
        setCreating(false);
        m_camera.yaw = -45;
        m_camera.pitch = 35;
        fit();
        updateColors();
    }
    void setTool(Tool tool) {
        m_tool = tool;
        m_polygon.clear();
        update();
    }
    void setThrough(bool value) { m_through = value; }
    void setPointSize(int size) {
        m_pointSize = size;
        update();
    }
    QColor pointColorAt(int index) const {
        if (!m_frame || index < 0 || index >= m_frame->points.size())
            return QColor(QStringLiteral("#809bb0"));
        const QVector4D& point = m_frame->points[index];
        const quint32 label =
                index < m_frame->labels.size() ? m_frame->labels[index] : 0;
        const QColor* rgb =
                index < m_frame->rgb.size() ? &m_frame->rgb[index] : nullptr;
        const QVector3D color = pointCloudDisplayColor(
                label, point.w(), rgb, m_colorMode, m_pointSemanticColors,
                m_intensityLow, m_intensityHigh);
        return QColor::fromRgbF(color.x(), color.y(), color.z());
    }
    void setBrushSize(int size) { m_brushSize = size; }
    void setColorMode(ColorMode mode) {
        m_colorMode = mode;
        updateColors();
    }
    void setVisibility(const QSet<quint16>& hiddenSemantics,
                       const QSet<quint16>& hiddenInstances) {
        m_hiddenSemantics = hiddenSemantics;
        m_hiddenInstances = hiddenInstances;
        rebuildVisibleIndices();
        m_upload = true;
        update();
    }
    const QVector<int>& visibleIndices() const { return m_visibleIndices; }
    void setSemanticColors(const QHash<quint16, QColor>& colors) {
        m_semanticColors = colors;
        m_upload = true;
        update();
    }
    void setPointSemanticColors(const QHash<quint16, QColor>& colors) {
        m_pointSemanticColors = colors;
        updateColors();
    }
    void setView(int view) {
        m_focusAnimation.stop();
        m_camera.setView(view);
        update();
    }
    void setPerspective(bool enabled) {
        m_camera.perspective = enabled;
        fit();
    }
    void focusCuboid(const PointCloudCuboid& box) {
        m_focusAnimation.stop();
        m_focusStart = m_camera;
        m_camera.focus(box, size());
        m_focusTarget = m_camera;
        double delta =
                std::max(std::abs(m_focusTarget.scale - m_focusStart.scale) /
                                 std::tan(25 * std::acos(-1.0) / 180),
                         1e-5);
        for (int axis = 0; axis < 3; ++axis)
            delta = std::max(delta, std::abs(m_focusTarget.center[axis] -
                                             m_focusStart.center[axis]));
        m_focusAnimation.setDuration(std::max(
                1,
                int(std::ceil(std::log(delta / 1e-5) / focusDecay() * 1000))));
        m_camera = m_focusStart;
        m_focusAnimation.start();
        update();
    }
    void fit() {
        m_focusAnimation.stop();
        if (m_frame) m_camera.fit(m_frame->points, size());
        update();
    }
    void updateColors() {
        m_upload = true;
        update();
    }
    void finishPolygon() {
        if (m_polygon.size() >= 3)
            select(
                    [this](const QPointF& p) {
                        return m_polygon.containsPoint(p, Qt::OddEvenFill);
                    },
                    false, m_polygon.boundingRect());
        m_polygon.clear();
        update();
    }
    void cancelPolygon() {
        m_polygon.clear();
        update();
    }

protected:
    void initializeGL() override {
        initializeOpenGLFunctions();
        glClearColor(0, 0, 0, 1);
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_PROGRAM_POINT_SIZE);
        float range[2] = {1, 1};
        glGetFloatv(GL_ALIASED_POINT_SIZE_RANGE, range);
        m_maxPointSize = std::max(1.0f, range[1]);
        m_vao.create();
        m_vao.bind();
        m_vbo.create();
        m_vbo.bind();
        m_shader.create();
        m_shader.addShaderFromSourceCode(
                QOpenGLShader::Vertex,
                "#version 330 core\nlayout(location=0) in vec3 position;\n"
                "layout(location=1) in vec3 color;\nuniform mat4 matrix;\n"
                "uniform vec2 viewport; uniform float size, pointScale, "
                "maxPointSize;\n"
                "out vec3 shade; flat out uint owner;\n"
                "void main(){vec4 clip=matrix*vec4(position,1);"
                "float pixels=size; if(pointScale>0) "
                "pixels=floor(clamp(pointScale/"
                "max(clip.w,.1),1,maxPointSize)+.5);"
                "if(any(greaterThan(abs(clip.xyz),vec3(clip.w))) || "
                "clip.x>=clip.w || clip.y<=-clip.w)"
                "gl_Position=vec4(2,2,2,1); else {"
                "vec2 screen=(clip.xy/clip.w+1)*viewport*.5; "
                "screen.y=viewport.y-screen.y;"
                "screen=floor(screen)+vec2(mod(pixels,2)*.5); "
                "screen.y=viewport.y-screen.y;"
                "clip.xy=(screen/viewport*2-1)*clip.w; gl_Position=clip;}"
                "gl_PointSize=pixels;shade=color;owner=uint(gl_VertexID)+1u;}");
        m_shader.addShaderFromSourceCode(
                QOpenGLShader::Fragment,
                "#version 330 core\nin vec3 shade; flat in uint owner; uniform "
                "bool picking; out vec4 pixel;\n"
                "void "
                "main(){pixel=picking?vec4(float(owner&255u),float((owner>>8)&"
                "255u),"
                "float((owner>>16)&255u),float((owner>>24)&255u))/"
                "255:vec4(shade,1);}");
        m_shader.link();
        m_shader.bind();
        m_shader.enableAttributeArray(0);
        m_shader.enableAttributeArray(1);
        m_shader.setAttributeBuffer(0, GL_FLOAT, 0, 3, 6 * sizeof(float));
        m_shader.setAttributeBuffer(1, GL_FLOAT, 3 * sizeof(float), 3,
                                    6 * sizeof(float));
        m_shader.release();
        m_vbo.release();
        m_vao.release();
        m_meshVao.create();
        m_meshVao.bind();
        m_meshVbo.create();
        m_meshVbo.bind();
        m_meshShader.addShaderFromSourceCode(
                QOpenGLShader::Vertex,
                "#version 330 core\nlayout(location=0) in vec3 position;"
                "layout(location=1) in vec4 color;uniform mat4 matrix;out vec4 "
                "shade;"
                "void "
                "main(){gl_Position=matrix*vec4(position,1);shade=color;}");
        m_meshShader.addShaderFromSourceCode(
                QOpenGLShader::Fragment,
                "#version 330 core\nin vec4 shade;out vec4 pixel;void "
                "main(){pixel=shade;}");
        m_meshShader.link();
        m_meshShader.bind();
        m_meshShader.enableAttributeArray(0);
        m_meshShader.enableAttributeArray(1);
        m_meshShader.setAttributeBuffer(0, GL_FLOAT, 0, 3, 7 * sizeof(float));
        m_meshShader.setAttributeBuffer(1, GL_FLOAT, 3 * sizeof(float), 4,
                                        7 * sizeof(float));
        m_meshShader.release();
        m_meshVbo.release();
        m_meshVao.release();
    }
    void paintGL() override {
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LEQUAL);
        glDepthMask(GL_TRUE);
        glDisable(GL_BLEND);
        glDisable(GL_DITHER);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        if (!m_frame || m_frame->points.isEmpty()) return;
        if (m_upload) upload();
        drawPoints(false);
        if (detection) drawCuboids();
        if (property("captureGeometryOnly").toBool()) return;
        // QPainter overlays are in screen coordinates, independent of the
        // point-cloud depth buffer populated above.
        glDisable(GL_DEPTH_TEST);
        if (detection) {
            QPainter painter(this);
            painter.setRenderHint(QPainter::Antialiasing);
            const auto transform = matrix();
            auto boxes = m_frame->cuboids;
            if (m_hasEditPreview)
                for (auto& box : boxes)
                    if (box.id == m_editPreview.id) box = m_editPreview;
            if (m_hasCreationPreview) boxes.append(m_creationPreview);
            for (const auto& box : boxes) {
                if (hiddenCuboids.contains(box.id) ||
                    m_hiddenSemantics.contains(quint16(box.classId)))
                    continue;
                const auto corners = box.corners();
                QPointF screen[8];
                bool visible = false;
                double left = std::numeric_limits<double>::infinity(),
                       top = left;
                for (int i = 0; i < 8; ++i) {
                    const auto& p = corners[i];
                    const auto clip =
                            transform *
                            QVector4D(float(p[0]), float(p[1]), float(p[2]), 1);
                    if (clip.w() <= 0) continue;
                    screen[i] = {(clip.x() / clip.w() + 1) * width() / 2,
                                 (1 - clip.y() / clip.w()) * height() / 2};
                    left = std::min(left, screen[i].x());
                    top = std::min(top, screen[i].y());
                    visible = true;
                }
                if (!visible) continue;
                const QColor color = m_semanticColors.value(
                        quint16(box.classId), QColor("#64b5f6"));
                painter.setPen(
                        QPen(color, box.id == selectedCuboid ? 2.5 : 1.3));
                if (box.id != 0)
                    painter.drawText(
                            QPointF(left, top - 7),
                            QStringLiteral("#%1 %2%3")
                                    .arg(box.id)
                                    .arg(classNames.value(
                                                 quint16(box.classId),
                                                 QString::number(box.classId)),
                                         box.locked ? tr(" [locked]")
                                                    : QString()));
            }
            painter.setPen(QColor("#cbd2df"));
            painter.drawText(QPointF(12, 22), tr("3D"));
        }
        if (!m_polygon.isEmpty()) {
            QPainter painter(this);
            painter.setPen(QPen(QColor(72, 160, 239), 2));
            painter.drawPolyline(m_polygon);
        }
        glEnable(GL_DEPTH_TEST);
    }
    void mousePressEvent(QMouseEvent* event) override {
        m_focusAnimation.stop();
        m_last = mousePosition(event);
        if (detection && m_creating) {
            updateCreationPreview(m_last);
            return;
        }
        m_navigationStart = m_last;
        m_navigationDragged = false;
        m_selectClick = detection && event->button() == Qt::LeftButton &&
                        !event->modifiers();
        if (m_tool == Polygon && event->button() == Qt::LeftButton) {
            m_polygon.append(m_last);
            update();
            return;
        }
        if (m_tool == Brush && event->button() == Qt::LeftButton) {
            m_brushing = true;
            m_hasBrushPosition = false;
            collectBrush(m_last);
            return;
        }
        m_panning = event->button() == Qt::RightButton ||
                    event->button() == Qt::MiddleButton ||
                    (event->button() == Qt::LeftButton &&
                     event->modifiers().testFlag(Qt::ControlModifier));
        m_orbiting = event->button() == Qt::LeftButton && !m_panning;
    }
    void mouseMoveEvent(QMouseEvent* event) override {
        const QPointF current = mousePosition(event);
        if (detection && m_creating) {
            updateCreationPreview(current);
            return;
        }
        if (detection && (m_orbiting || m_panning) && !m_navigationDragged) {
            if (QLineF(current, m_navigationStart).length() <
                QApplication::startDragDistance())
                return;
            m_navigationDragged = true;
        }
        const QPointF delta = current - m_last;
        m_last = current;
        if (m_brushing)
            collectBrush(current);
        else if (m_orbiting) {
            m_camera.orbit(delta.x(), delta.y());
            update();
        } else if (m_panning) {
            m_camera.pan(delta.x(), delta.y(), height());
            if (m_tool == Polygon)
                for (auto& point : m_polygon) point += delta;
            update();
        }
    }
    void leaveEvent(QEvent* event) override {
        QOpenGLWidget::leaveEvent(event);
        if (!m_hasCreationPreview) return;
        m_hasCreationPreview = false;
        update();
        if (onCreationPreviewChanged) onCreationPreviewChanged();
    }
    void mouseReleaseEvent(QMouseEvent* event) override {
        if (m_selectClick && !m_navigationDragged && onCuboidSelected)
            onCuboidSelected(hitCuboid(mousePosition(event)));
        m_selectClick = false;
        if (event->button() == Qt::LeftButton && m_brushing) {
            m_brushing = false;
            m_hasBrushPosition = false;
            QVector<int> indices;
            indices.reserve(m_stroke.size());
            for (int index : m_stroke) indices.append(index);
            m_stroke.clear();
            if (onSelection && !indices.isEmpty()) onSelection(indices);
        }
        m_panning = false;
        m_orbiting = false;
    }
    void mouseDoubleClickEvent(QMouseEvent* event) override {
        if (detection && event->button() == Qt::LeftButton &&
            !event->modifiers()) {
            if (m_creating) {
                updateCreationPreview(mousePosition(event));
                if (m_hasCreationPreview && onCuboidCreated)
                    onCuboidCreated(m_creationPreview.center,
                                    m_creationPreview.size);
            } else {
                const int id = hitCuboid(mousePosition(event));
                if (onCuboidSelected) onCuboidSelected(id);
                if (id && m_frame) {
                    for (const auto& box : m_frame->cuboids)
                        if (box.id == id) {
                            focusCuboid(box);
                            break;
                        }
                } else
                    fit();
            }
            m_panning = m_orbiting = m_selectClick = false;
            event->accept();
            return;
        }
        QOpenGLWidget::mouseDoubleClickEvent(event);
    }
    void wheelEvent(QWheelEvent* event) override {
        m_focusAnimation.stop();
        if (m_tool == Brush &&
            event->modifiers().testFlag(Qt::ControlModifier)) {
            m_brushSize = std::clamp(
                    m_brushSize + event->angleDelta().y() / 120 * 2, 2, 100);
            if (onBrushSizeChanged) onBrushSizeChanged(m_brushSize);
        } else {
            const double previous = m_camera.scale;
            m_camera.zoom(event->angleDelta().y());
            if (!m_polygon.isEmpty()) {
                const QPointF center(width() / 2.0, height() / 2.0);
                for (auto& point : m_polygon)
                    point = center +
                            (point - center) * (previous / m_camera.scale);
            }
        }
        event->accept();
        update();
        if (detection && m_creating) {
#if QT_VERSION >= QT_VERSION_CHECK(5, 14, 0)
            updateCreationPreview(event->position());
#else
            updateCreationPreview(event->posF());
#endif
        }
    }
    void keyPressEvent(QKeyEvent* event) override {
        if (detection && (m_tool == Browse || m_creating) && m_frame &&
            event->modifiers() == Qt::NoModifier) {
            int x = 0, y = 0;
            switch (event->key()) {
                case Qt::Key_Left:
                    x = -1;
                    break;
                case Qt::Key_Right:
                    x = 1;
                    break;
                case Qt::Key_Up:
                    y = 1;
                    break;
                case Qt::Key_Down:
                    y = -1;
                    break;
                default:
                    break;
            }
            if (x || y) {
                for (const auto& box : m_frame->cuboids)
                    if (box.id == selectedCuboid && !box.locked &&
                        onCuboidEdited) {
                        const auto changed = m_camera.nudgedCuboid(box, x, y);
                        cancelPolygon();
                        onCuboidEdited(changed);
                        event->accept();
                        return;
                    }
            }
        }
        const bool cameraLetter = event->modifiers() == Qt::AltModifier;
        const bool cameraArrow = event->modifiers() == Qt::ShiftModifier;
        QString command;
        if (cameraLetter &&
            (event->key() == Qt::Key_U || event->key() == Qt::Key_I ||
             event->key() == Qt::Key_O || event->key() == Qt::Key_J ||
             event->key() == Qt::Key_K || event->key() == Qt::Key_L))
            command = QChar(event->key());
        if (cameraArrow) {
            if (event->key() == Qt::Key_Left) command = "Left";
            if (event->key() == Qt::Key_Right) command = "Right";
            if (event->key() == Qt::Key_Up) command = "Up";
            if (event->key() == Qt::Key_Down) command = "Down";
        }
        if (!command.isEmpty() && m_frame && !m_frame->points.isEmpty() &&
            m_polygon.isEmpty() && !m_brushing) {
            m_focusAnimation.stop();
            m_camera.moveCamera(command);
            update();
            event->accept();
            return;
        }
        if (event->key() == Qt::Key_Return || event->key() == Qt::Key_Enter)
            finishPolygon();
        else if (event->key() == Qt::Key_Escape) {
            cancelPolygon();
            if (m_creating) {
                setCreating(false);
                if (onCreationCancelled) onCreationCancelled();
            }
        } else
            QOpenGLWidget::keyPressEvent(event);
    }

private:
    void rebuildIntensityRange() {
        m_intensityLow = 0;
        m_intensityHigh = 1;
        if (!m_frame || m_frame->points.isEmpty()) return;
        double low = std::numeric_limits<double>::infinity();
        double high = -low;
        for (const auto& point : m_frame->points) {
            if (!std::isfinite(point.w())) continue;
            low = std::min(low, double(point.w()));
            high = std::max(high, double(point.w()));
        }
        if (std::isfinite(low)) {
            m_intensityLow = low;
            m_intensityHigh = high;
        }
    }
    void rebuildVisibleIndices() {
        m_visibleIndices.clear();
        if (!m_frame) return;
        m_visibleIndices.reserve(m_frame->points.size());
        for (int i = 0; i < m_frame->points.size(); ++i) {
            const quint32 label =
                    i < m_frame->labels.size() ? m_frame->labels[i] : 0;
            if (m_hiddenSemantics.contains(quint16(label)) ||
                (quint16(label >> 16) &&
                 m_hiddenInstances.contains(quint16(label >> 16))))
                continue;
            m_visibleIndices.append(i);
        }
    }
    void drawCuboids() {
        QVector<PointCloudCuboid> visible;
        for (const auto& member : m_frame->cuboids) {
            const auto& box = m_hasEditPreview && member.id == m_editPreview.id
                                      ? m_editPreview
                                      : member;
            if (!hiddenCuboids.contains(box.id) &&
                !m_hiddenSemantics.contains(quint16(box.classId)))
                visible.append(box);
        }
        if (visible != m_meshBoxes || m_meshSelected != selectedCuboid ||
            m_meshColors != m_semanticColors ||
            m_meshPreviewActive != m_hasCreationPreview ||
            (m_hasCreationPreview && m_meshPreview != m_creationPreview)) {
            PointCloudCuboidMesh mesh;
            QString error;
            if (!pointCloudCuboidMesh(
                        visible, selectedCuboid, m_semanticColors,
                        m_hasCreationPreview ? &m_creationPreview : nullptr,
                        false, &mesh, &error)) {
                if (onSelectionError) onSelectionError(error);
                return;
            }
            m_meshVbo.bind();
            m_meshVbo.allocate(mesh.vertices.constData(),
                               int(mesh.vertices.size() * sizeof(float)));
            m_meshVbo.release();
            m_meshFaces = mesh.faceVertices;
            m_meshCount = mesh.vertices.size() / 7;
            m_meshBoxes = visible;
            m_meshSelected = selectedCuboid;
            m_meshColors = m_semanticColors;
            m_meshPreviewActive = m_hasCreationPreview;
            m_meshPreview = m_creationPreview;
        }
        if (!m_meshCount) return;
        m_meshShader.bind();
        m_meshShader.setUniformValue("matrix", matrix());
        m_meshVao.bind();
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LEQUAL);
        glDepthMask(GL_TRUE);
        glEnable(GL_BLEND);
        glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE,
                            GL_ONE_MINUS_SRC_ALPHA);
        glEnable(GL_CULL_FACE);
        glCullFace(GL_BACK);
        glEnable(GL_POLYGON_OFFSET_FILL);
        glPolygonOffset(1, 1);
        glDrawArrays(GL_TRIANGLES, 0, m_meshFaces);
        glDisable(GL_POLYGON_OFFSET_FILL);
        glDisable(GL_CULL_FACE);
        glLineWidth(1);
        glDrawArrays(GL_LINES, m_meshFaces, m_meshCount - m_meshFaces);
        glDisable(GL_BLEND);
        m_meshVao.release();
        m_meshShader.release();
    }
    int hitCuboid(const QPointF& at) const {
        if (!m_frame) return 0;
        QVector<PointCloudCuboid> visible;
        for (const auto& box : m_frame->cuboids) {
            if (hiddenCuboids.contains(box.id) ||
                m_hiddenSemantics.contains(quint16(box.classId)))
                continue;
            visible.append(box);
        }
        return m_camera.hit(at, size(), visible);
    }
    void updateCreationPreview(const QPointF& at) {
        if (!m_creating || !m_frame) return;
        if (!QRectF(rect()).contains(at)) {
            m_hasCreationPreview = false;
            update();
            if (onCreationPreviewChanged) onCreationPreviewChanged();
            return;
        }
        QVector<int> visible;
        visible.reserve(m_frame->points.size());
        for (int i = 0; i < m_frame->points.size(); ++i) {
            const auto label = m_frame->labels[i];
            if (!m_hiddenSemantics.contains(quint16(label)) &&
                (!quint16(label >> 16) ||
                 !m_hiddenInstances.contains(quint16(label >> 16))))
                visible.append(i);
        }
        PointCloudCuboid box;
        m_hasCreationPreview = false;
        if (m_camera.creationPreview(at, size(), m_frame->points, visible,
                                     &box)) {
            box.id = 0;
            m_creationPreview = box;
            m_hasCreationPreview = true;
        }
        update();
        if (onCreationPreviewChanged) onCreationPreviewChanged();
    }
    QMatrix4x4 matrix() const { return m_camera.matrix(size()); }
    void upload() {
        m_surfaceOwners = {};
        m_projectionValid = false;
        QVector<float> packed;
        packed.reserve(m_frame->points.size() * 6);
        for (const int i : m_visibleIndices) {
            const QVector4D& point = m_frame->points[i];
            const quint32 label =
                    i < m_frame->labels.size() ? m_frame->labels[i] : 0;
            const auto color = pointCloudDisplayColor(
                    label, point.w(),
                    i < m_frame->rgb.size() ? &m_frame->rgb[i] : nullptr,
                    m_colorMode, m_pointSemanticColors, m_intensityLow,
                    m_intensityHigh);
            packed << point.x() << point.y() << point.z() << color.x()
                   << color.y() << color.z();
        }
        m_vbo.bind();
        m_vbo.allocate(packed.constData(), int(packed.size() * sizeof(float)));
        m_vbo.release();
        m_upload = false;
    }
    QSize physicalSize() const {
        return {std::max(1, qRound(width() * devicePixelRatioF())),
                std::max(1, qRound(height() * devicePixelRatioF()))};
    }
    float pointScale() const {
        return m_camera.perspective ? .05f * (m_pointSize / 2.0f) *
                                              physicalSize().height() / 2.0f
                                    : 0;
    }
    int pixelSize(float depth) const {
        if (!m_camera.perspective)
            return std::max(1, qRound(m_pointSize * devicePixelRatioF()));
        return int(std::floor(std::clamp(pointScale() / std::max(depth, .1f),
                                         1.0f, m_maxPointSize) +
                              .5f));
    }
    void drawPoints(bool picking) {
        m_shader.bind();
        m_shader.setUniformValue("matrix", matrix());
        m_shader.setUniformValue(
                "viewport",
                QVector2D(physicalSize().width(), physicalSize().height()));
        m_shader.setUniformValue(
                "size",
                float(std::max(1, qRound(m_pointSize * devicePixelRatioF()))));
        m_shader.setUniformValue("pointScale", pointScale());
        m_shader.setUniformValue("maxPointSize", m_maxPointSize);
        m_shader.setUniformValue("picking", picking);
        m_vao.bind();
        glDrawArrays(GL_POINTS, 0, m_visibleIndices.size());
        m_vao.release();
        m_shader.release();
    }
    QImage captureSurface() {
        if (!isValid()) return {};
        if (!m_upload && !m_surfaceOwners.isNull() &&
            m_surfaceOwners.size() == physicalSize() &&
            m_surfaceMatrix == matrix() && m_surfacePointSize == m_pointSize)
            return m_surfaceOwners;
        makeCurrent();
        if (m_upload) upload();
        QImage owners;
        {
            QOpenGLFramebufferObjectFormat format;
            format.setAttachment(
                    QOpenGLFramebufferObject::CombinedDepthStencil);
            format.setInternalTextureFormat(GL_RGBA8);
            QOpenGLFramebufferObject framebuffer(physicalSize(), format);
            if (framebuffer.isValid() && framebuffer.bind()) {
                const bool blend = glIsEnabled(GL_BLEND),
                           dither = glIsEnabled(GL_DITHER),
                           srgb = glIsEnabled(GL_FRAMEBUFFER_SRGB);
                glDisable(GL_BLEND);
                glDisable(GL_DITHER);
                glDisable(GL_FRAMEBUFFER_SRGB);
                GLboolean depthWrite = GL_TRUE;
                glGetBooleanv(GL_DEPTH_WRITEMASK, &depthWrite);
                glEnable(GL_DEPTH_TEST);
                glDepthFunc(GL_LESS);
                glDepthMask(GL_TRUE);
                glViewport(0, 0, physicalSize().width(),
                           physicalSize().height());
                glClearColor(0, 0, 0, 0);
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
                drawPoints(true);
                owners = QImage(physicalSize(), QImage::Format_RGBA8888);
                if (!owners.isNull()) {
                    glReadPixels(0, 0, owners.width(), owners.height(), GL_RGBA,
                                 GL_UNSIGNED_BYTE, owners.bits());
                    if (glGetError() != GL_NO_ERROR)
                        owners = {};
                    else
                        owners = owners.mirrored(false, true);
                }
                glDepthMask(depthWrite);
                if (blend) glEnable(GL_BLEND);
                if (dither) glEnable(GL_DITHER);
                if (srgb) glEnable(GL_FRAMEBUFFER_SRGB);
                glClearColor(0, 0, 0, 1);
            }
            glBindFramebuffer(GL_FRAMEBUFFER, defaultFramebufferObject());
        }
        doneCurrent();
        setProperty("surfacePickingAvailable", !owners.isNull());
        m_surfaceOwners = owners;
        m_surfaceMatrix = matrix();
        m_surfacePointSize = m_pointSize;
        return owners;
    }
    void collectBrush(const QPointF& at) {
        const QPointF start =
                m_brushing && m_hasBrushPosition ? m_brushPosition : at;
        m_brushPosition = at;
        m_hasBrushPosition = true;
        select(
                [this, start, at](const QPointF& p) {
                    const QPointF axis = at - start;
                    const double squared = QPointF::dotProduct(axis, axis);
                    const double t =
                            squared > 0 ? std::clamp(QPointF::dotProduct(
                                                             p - start, axis) /
                                                             squared,
                                                     0.0, 1.0)
                                        : 0;
                    return QLineF(p, start + axis * t).length() <= m_brushSize;
                },
                true,
                QRectF(start, at).normalized().adjusted(
                        -m_brushSize, -m_brushSize, m_brushSize, m_brushSize));
    }
    void select(const std::function<bool(const QPointF&)>& contains,
                bool accumulate = false,
                const QRectF& bounds = {}) {
        if (!m_frame || !onSelection) return;
        if (!isValid()) {
            if (onSelectionError)
                onSelectionError(PointCloudWindow::tr(
                        "The OpenGL viewport is not ready for selection."));
            return;
        }
        QImage owners;
        if (!m_through) {
            owners = captureSurface();
            if (owners.isNull()) {
                if (onSelectionError)
                    onSelectionError(PointCloudWindow::tr(
                            "The OpenGL viewport is not ready for selection."));
                return;
            }
        } else if (m_upload) {
            makeCurrent();
            upload();
            doneCurrent();
        }
        const QMatrix4x4 transform = matrix();
        const QSize size = physicalSize();
        const double ratio = devicePixelRatioF();
        const QRect region =
                QRectF(bounds.topLeft() * ratio, bounds.bottomRight() * ratio)
                        .toAlignedRect()
                        .intersected(QRect(QPoint(0, 0), size));
        if (region.isEmpty()) return;
        if (!m_projectionValid || m_projectionMatrix != transform ||
            m_projectionSize != size || m_projectionPointSize != m_pointSize) {
            m_projected.fill({}, m_visibleIndices.size());
            m_tileColumns = (size.width() + 15) / 16;
            const int tileRows = (size.height() + 15) / 16;
            m_selectionTiles.clear();
            m_selectionTiles.resize(m_tileColumns * tileRows);
            m_selectionMaxPixels = 1;
            // O(n + viewport tiles), O(n + viewport tiles) storage. A point
            // belongs to its center tile once; queries expand by the largest
            // footprint so even a large perspective point cannot be missed.
            for (int local = 0; local < m_visibleIndices.size(); ++local) {
                const auto clip =
                        transform *
                        QVector4D(m_frame->points[m_visibleIndices[local]]
                                          .toVector3D(),
                                  1);
                if (clip.w() <= 0) continue;
                const float depth = clip.z() / clip.w();
                const double x = (clip.x() / clip.w() + 1) * size.width() / 2.0,
                             y = (1 - clip.y() / clip.w()) * size.height() /
                                 2.0;
                if (!std::isfinite(x) || !std::isfinite(y) || depth < -1 ||
                    depth > 1 || x < 0 || y < 0 || x >= size.width() ||
                    y >= size.height())
                    continue;
                const int pixels = pixelSize(clip.w());
                m_projected[local] = {QPoint(int(std::floor(x)) - pixels / 2,
                                             int(std::floor(y)) - pixels / 2),
                                      depth, pixels};
                m_selectionTiles[(int(y) / 16) * m_tileColumns + int(x) / 16]
                        .append(local);
                m_selectionMaxPixels = std::max(m_selectionMaxPixels, pixels);
            }
            m_projectionMatrix = transform;
            m_projectionSize = size;
            m_projectionPointSize = m_pointSize;
            m_projectionValid = true;
            setProperty("selectionProjectionBuilds", ++m_projectionBuilds);
        }
        const int margin = (m_selectionMaxPixels + 1) / 2;
        const QRect tileRegion =
                region.adjusted(-margin, -margin, margin, margin)
                        .intersected(QRect(QPoint(0, 0), size));
        QVector<int> candidates;
        for (int y = tileRegion.top() / 16; y <= tileRegion.bottom() / 16; ++y)
            for (int x = tileRegion.left() / 16; x <= tileRegion.right() / 16;
                 ++x)
                candidates += m_selectionTiles[y * m_tileColumns + x];
        // O(k log k), needed only for source point order after gathering tiles.
        std::sort(candidates.begin(), candidates.end());
        setProperty("selectionCandidateCount", candidates.size());
        QVector<int> selected;
        for (int local : candidates) {
            const auto& p = m_projected[local];
            bool hit = false;
            for (int y = std::max(region.top(), p.origin.y());
                 y < std::min(region.bottom() + 1, p.origin.y() + p.size) &&
                 !hit;
                 ++y)
                for (int x = std::max(region.left(), p.origin.x());
                     x < std::min(region.right() + 1, p.origin.x() + p.size);
                     ++x) {
                    if (!contains(QPointF((x + .5) / ratio, (y + .5) / ratio)))
                        continue;
                    if (!m_through) {
                        const auto* rgba = owners.constScanLine(y) + x * 4;
                        const quint32 owner = quint32(rgba[0]) |
                                              (quint32(rgba[1]) << 8) |
                                              (quint32(rgba[2]) << 16) |
                                              (quint32(rgba[3]) << 24);
                        if (!owner || owner > quint32(m_projected.size()))
                            continue;
                        const auto& front = m_projected[int(owner) - 1];
                        // Equal-depth points sharing a footprint all remain
                        // selectable, including ties hidden by GL_LESS.
                        if (owner != quint32(local + 1) &&
                            (front.origin != p.origin ||
                             front.depth != p.depth))
                            continue;
                    }
                    hit = true;
                    break;
                }
            if (hit) selected.append(m_visibleIndices[local]);
        }
        if (accumulate)
            for (int index : selected) m_stroke.insert(index);
        else if (!selected.isEmpty())
            onSelection(selected);
    }

    const PointCloudFrame* m_frame = nullptr;
    QOpenGLVertexArrayObject m_vao;
    QOpenGLBuffer m_vbo{QOpenGLBuffer::VertexBuffer};
    QOpenGLShaderProgram m_shader;
    QOpenGLVertexArrayObject m_meshVao;
    QOpenGLBuffer m_meshVbo{QOpenGLBuffer::VertexBuffer};
    QOpenGLShaderProgram m_meshShader;
    QVector<PointCloudCuboid> m_meshBoxes;
    QHash<quint16, QColor> m_meshColors;
    PointCloudCuboid m_meshPreview;
    int m_meshSelected = -1, m_meshFaces = 0, m_meshCount = 0;
    bool m_meshPreviewActive = false;
    PointCloudViewTransform m_camera;
    QVariantAnimation m_focusAnimation{this};
    PointCloudViewTransform m_focusStart, m_focusTarget;
    static double focusDecay() { return -60 * std::log(0.95); }
    int m_pointSize = 2;
    int m_brushSize = 14;
    Tool m_tool = Browse;
    ColorMode m_colorMode = ColorMode::Semantic;
    bool m_upload = true;
    bool m_through = false;
    bool m_brushing = false;
    bool m_hasBrushPosition = false;
    QPointF m_brushPosition;
    float m_maxPointSize = 1;
    struct Projected {
        QPoint origin;
        float depth = 0;
        int size = 0;
    };
    QVector<Projected> m_projected;
    QVector<QVector<int>> m_selectionTiles;
    QMatrix4x4 m_projectionMatrix;
    QSize m_projectionSize;
    int m_projectionPointSize = 0, m_selectionMaxPixels = 1, m_tileColumns = 0;
    int m_projectionBuilds = 0;
    bool m_projectionValid = false;
    QImage m_surfaceOwners;
    QMatrix4x4 m_surfaceMatrix;
    int m_surfacePointSize = 0;
    bool m_creating = false;
    bool m_hasCreationPreview = false;
    PointCloudCuboid m_creationPreview;
    bool m_hasEditPreview = false;
    PointCloudCuboid m_editPreview;
    QPointF m_navigationStart;
    bool m_navigationDragged = false;
    bool m_selectClick = false;
    bool m_panning = false;
    bool m_orbiting = false;
    QPointF m_last;
    QPolygonF m_polygon;
    QSet<int> m_stroke;
    QVector<int> m_visibleIndices;
    QSet<quint16> m_hiddenSemantics;
    double m_intensityLow = 0;
    double m_intensityHigh = 1;
    QSet<quint16> m_hiddenInstances;
    QHash<quint16, QColor> m_semanticColors;
    QHash<quint16, QColor> m_pointSemanticColors;
};

PointCloudWindow::PointCloudWindow(QWidget* parent) : QMainWindow(parent) {
    setObjectName(QStringLiteral("qAnyLabelingPointCloud"));
    setWindowTitle(tr("Point cloud annotation"));
    resize(1280, 800);
    auto* toolbar = addToolBar(tr("Point cloud files"));
    toolbar->addAction(tr("Open file"), this, [this]() {
        const QString path = QFileDialog::getOpenFileName(
                this, tr("Open point cloud"), QString(),
                tr("Point clouds (*.bin *.ply)"));
        if (!path.isEmpty()) openFile(path);
    });
    toolbar->addAction(tr("Open dir"), this, &PointCloudWindow::openDirectory);
    auto* configureCamera = toolbar->addAction(
            tr("Configure camera"), this, &PointCloudWindow::configureCamera);
    configureCamera->setObjectName(QStringLiteral("PointCloudCameraConfigure"));
    toolbar->addAction(tr("Output dir"), this, [this]() {
        const QString path = QFileDialog::getExistingDirectory(
                this, tr("Label output directory"));
        if (!path.isEmpty()) m_outputDirectory = path;
    });
    toolbar->addAction(tr("Load classes"), this,
                       &PointCloudWindow::loadClasses);
    toolbar->addAction(tr("Save classes"), this,
                       &PointCloudWindow::saveClasses);
    toolbar->addAction(tr("New class"), this, &PointCloudWindow::addClass);
    toolbar->addAction(tr("Edit class"), this, &PointCloudWindow::editClass);
    toolbar->addAction(tr("Remove class"), this,
                       &PointCloudWindow::removeClass);
    auto* datasetMenu = menuBar()->addMenu(tr("Dataset"));
    auto* importAction = datasetMenu->addAction(
            tr("Import 3D cuboids"), this, [this] { datasetDialog(true); });
    importAction->setObjectName(QStringLiteral("PointCloudDatasetImport"));
    auto* exportAction = datasetMenu->addAction(
            tr("Export 3D cuboids"), this, [this] { datasetDialog(false); });
    exportAction->setObjectName(QStringLiteral("PointCloudDatasetExport"));
    auto* saveAction =
            toolbar->addAction(tr("Save"), this, &PointCloudWindow::save);
    saveAction->setShortcut(QKeySequence::Save);
    toolbar->addAction(tr("Save as"), this, &PointCloudWindow::saveAs);
    auto* undoAction = toolbar->addAction(tr("Undo"), this, [this]() {
        if (m_document.undo()) refresh();
    });
    undoAction->setShortcut(QKeySequence::Undo);
    auto* redoAction = toolbar->addAction(tr("Redo"), this, [this]() {
        if (m_document.redo()) refresh();
    });
    redoAction->setShortcut(QKeySequence::Redo);

    auto* splitter = new QSplitter(this);
    m_fileList = new QListWidget(splitter);
    m_fileList->setObjectName(QStringLiteral("PointCloudFrames"));
    splitter->addWidget(m_fileList);
    auto* center = new QWidget(splitter);
    auto* layout = new QVBoxLayout(center);
    layout->setContentsMargins(0, 0, 0, 0);
    auto* tools = new QToolBar(center);
    layout->addWidget(tools);
    m_task = new QComboBox(tools);
    m_task->setObjectName(QStringLiteral("PointCloudTask"));
    m_task->addItems({tr("Segmentation"), tr("3D object detection")});
    tools->addWidget(m_task);
    m_view = new PointCloudView(center);
    layout->addWidget(m_view, 1);
    splitter->addWidget(center);
    auto* right = new QWidget(splitter);
    auto* rightLayout = new QVBoxLayout(right);
    m_classHeading = new QLabel(tr("Semantic class"), right);
    m_classHeading->setObjectName(QStringLiteral("PointCloudClassHeading"));
    rightLayout->addWidget(m_classHeading);
    m_class = new QComboBox(right);
    m_class->setObjectName(QStringLiteral("PointCloudClass"));
    m_class->setEditable(true);
    refreshClassCombo();
    rightLayout->addWidget(m_class);
    auto* operationHeading = new QLabel(tr("Operation"), right);
    rightLayout->addWidget(operationHeading);
    m_operation = new QComboBox(right);
    m_operation->setObjectName(QStringLiteral("PointCloudOperation"));
    m_operation->addItem(tr("Assign semantic"),
                         int(PointCloudDocument::AssignSemantic));
    m_operation->addItem(tr("Create instance"),
                         int(PointCloudDocument::CreateInstance));
    m_operation->addItem(tr("Add to instance"),
                         int(PointCloudDocument::AddToInstance));
    m_operation->addItem(tr("Remove from instance"),
                         int(PointCloudDocument::RemoveFromInstance));
    m_operation->addItem(tr("Create cuboid from selection"),
                         PointCloudDocument::CreateCuboid);
    m_operation->addItem(tr("Split instance"),
                         int(PointCloudDocument::SplitInstance));
    rightLayout->addWidget(m_operation);
    auto* instanceHeading = new QLabel(tr("Instance ID"), right);
    rightLayout->addWidget(instanceHeading);
    m_instance = new QSpinBox(right);
    m_instance->setRange(0, 65535);
    rightLayout->addWidget(m_instance);
    rightLayout->addWidget(new QLabel(tr("Class visibility"), right));
    m_classList = new QListWidget(right);
    m_classList->setSelectionMode(QAbstractItemView::SingleSelection);
    rightLayout->addWidget(m_classList, 1);
    auto* instanceVisibilityHeading =
            new QLabel(tr("Instance visibility"), right);
    rightLayout->addWidget(instanceVisibilityHeading);
    m_instanceList = new QListWidget(right);
    m_instanceList->setSelectionMode(QAbstractItemView::NoSelection);
    rightLayout->addWidget(m_instanceList, 1);
    m_segmentationControls = {operationHeading,          m_operation,
                              instanceHeading,           m_instance,
                              instanceVisibilityHeading, m_instanceList};
    rightLayout->addStretch();
    splitter->addWidget(right);
    setupCuboids(center, layout, rightLayout);
    splitter->setStretchFactor(1, 1);
    setCentralWidget(splitter);
    m_cameraPanel = new PointCloudCameraPanel(m_view);
    m_cameraPanel->attachToViewport(m_view);
    m_cameraPanel->hide();
    m_view->onCreationPreviewChanged = [this] {
        m_cameraPanel->setCreationPreview(m_view->creationPreview());
    };
    m_cameraToggle = toolbar->addAction(tr("Show camera images"));
    m_cameraToggle->setObjectName(QStringLiteral("PointCloudCameraToggle"));
    m_cameraToggle->setCheckable(true);
    m_cameraToggle->setEnabled(false);
    connect(m_cameraToggle, &QAction::toggled, this, [this](bool visible) {
        m_cameraPanel->setVisible(visible);
        if (visible) m_cameraPanel->raise();
        m_cameraToggle->setText(visible ? tr("Hide camera images")
                                        : tr("Show camera images"));
        m_cameraToggle->setToolTip(m_cameraToggle->text());
    });
    m_status = new QLabel(this);
    statusBar()->addWidget(m_status);
    connect(m_fileList, &QListWidget::currentRowChanged, this,
            &PointCloudWindow::selectFrame);
    m_view->onSelectionError = [this](const QString& error) {
        m_status->setText(error);
    };
    m_view->onSelection = [this](const QVector<int>& selected) {
        applySelection(selected);
    };
    connect(m_classList, &QListWidget::itemChanged, this,
            [this](QListWidgetItem*) { updateVisibility(); });
    connect(m_instanceList, &QListWidget::itemChanged, this,
            [this](QListWidgetItem*) { updateVisibility(); });
    connect(m_classList, &QListWidget::currentRowChanged, this,
            [this](int row) {
                if (row < 0 || row >= m_classList->count()) return;
                const quint16 semantic = quint16(
                        m_classList->item(row)->data(Qt::UserRole).toInt());
                const QSignalBlocker blocker(m_class);
                const int comboRow = m_class->findData(semantic);
                if (comboRow >= 0) m_class->setCurrentIndex(comboRow);
            });
    connect(m_classList, &QListWidget::itemDoubleClicked, this,
            [this](QListWidgetItem*) { editClass(); });

    auto* group = new QActionGroup(this);
    for (const auto& item : {std::pair<const char*, PointCloudView::Tool>{
                                     "Browse", PointCloudView::Browse},
                             {"Brush", PointCloudView::Brush},
                             {"Polygon", PointCloudView::Polygon}}) {
        QAction* action = tools->addAction(tr(item.first));
        action->setObjectName(QStringLiteral("PointCloudTool") +
                              QString::fromLatin1(item.first));
        action->setCheckable(true);
        group->addAction(action);
        if (item.second != PointCloudView::Browse)
            m_segmentationActions.append(action);
        if (item.second == PointCloudView::Browse) action->setChecked(true);
        connect(action, &QAction::triggered, this,
                [this, item]() { m_view->setTool(item.second); });
    }
    m_segmentationActions.append(tools->addAction(
            tr("Finish polygon"), m_view, &PointCloudView::finishPolygon));
    m_segmentationActions.append(tools->addAction(
            tr("Cancel polygon"), m_view, &PointCloudView::cancelPolygon));
    tools->addAction(tr("Fit"), m_view, &PointCloudView::fit);
    auto* perspective = tools->addAction(tr("Perspective"));
    perspective->setObjectName(QStringLiteral("PointCloudPerspective"));
    perspective->setCheckable(true);
    perspective->setChecked(true);
    connect(perspective, &QAction::toggled, m_view,
            &PointCloudView::setPerspective);
    for (const auto& item :
         {std::pair<const char*, int>{"Top", 0}, {"Front", 1}, {"Side", 2}})
        tools->addAction(tr(item.first), m_view,
                         [this, item]() { m_view->setView(item.second); })
                ->setObjectName(QStringLiteral("PointCloudView") +
                                QString::fromLatin1(item.first));
    QAction* through = tools->addAction(tr("Through selection"));
    m_segmentationActions.append(through);
    through->setObjectName(QStringLiteral("PointCloudThrough"));
    through->setCheckable(true);
    connect(through, &QAction::toggled, m_view, &PointCloudView::setThrough);
    auto* render = m_colorMode = new QComboBox(tools);
    render->setObjectName(QStringLiteral("PointCloudColorMode"));
    render->addItems(
            {tr("Semantic"), tr("Intensity"), tr("RGB"), tr("Instance")});
    connect(render, QOverload<int>::of(&QComboBox::currentIndexChanged), m_view,
            [this](int index) {
                m_view->setColorMode(PointCloudView::ColorMode(index));
                for (auto* view : m_orthographic) view->update();
                updateCamera();
            });
    tools->addWidget(render);
    auto* pointSize = new QSpinBox(tools);
    pointSize->setObjectName(QStringLiteral("PointCloudPointSize"));
    pointSize->setRange(1, 10);
    pointSize->setValue(2);
    pointSize->setToolTip(tr("Point size (px)"));
    connect(pointSize, QOverload<int>::of(&QSpinBox::valueChanged), this,
            &PointCloudWindow::setPointCloudPointSize);
    tools->addWidget(pointSize);
    m_brushSizeControl = new QSpinBox(tools);
    m_brushSizeControl->setObjectName(QStringLiteral("PointCloudBrushSize"));
    m_brushSizeControl->setRange(2, 100);
    m_brushSizeControl->setValue(14);
    m_brushSizeControl->setToolTip(tr("Brush size (px)"));
    connect(m_brushSizeControl, QOverload<int>::of(&QSpinBox::valueChanged),
            m_view, &PointCloudView::setBrushSize);
    m_view->onBrushSizeChanged = [this](int size) {
        m_brushSizeControl->setValue(size);
    };
    m_segmentationActions.append(tools->addWidget(m_brushSizeControl));
}

void PointCloudWindow::configureCamera() {
    QDialog dialog(this);
    dialog.setObjectName(QStringLiteral("PointCloudCameraConfigurationDialog"));
    dialog.setWindowTitle(tr("Configure camera images"));
    dialog.setMinimumWidth(560);
    auto* form = new QVBoxLayout(&dialog);
    auto* scroll = new QScrollArea(&dialog);
    scroll->setWidgetResizable(true);
    scroll->setMinimumHeight(220);
    auto* sourcesWidget = new QWidget(scroll);
    auto* sourceRows = new QVBoxLayout(sourcesWidget);
    sourceRows->setAlignment(Qt::AlignTop);
    scroll->setWidget(sourcesWidget);
    form->addWidget(scroll);
    auto* preview = new QLabel(&dialog);
    preview->setObjectName(QStringLiteral("PointCloudCameraPreview"));
    preview->setAccessibleName(tr("Camera preview"));
    preview->setAlignment(Qt::AlignCenter);
    preview->setMinimumHeight(180);
    preview->setStyleSheet(QStringLiteral("background: #101114;"));
    auto* previewControls = new QHBoxLayout;
    auto* previousPreview = new QToolButton(&dialog);
    previousPreview->setObjectName(
            QStringLiteral("PointCloudCameraPreviewPrevious"));
    previousPreview->setArrowType(Qt::LeftArrow);
    previousPreview->setAccessibleName(tr("Previous camera"));
    auto* previewCamera = new QComboBox(&dialog);
    previewCamera->setObjectName(
            QStringLiteral("PointCloudCameraPreviewSelector"));
    auto* nextPreview = new QToolButton(&dialog);
    nextPreview->setObjectName(QStringLiteral("PointCloudCameraPreviewNext"));
    nextPreview->setArrowType(Qt::RightArrow);
    nextPreview->setAccessibleName(tr("Next camera"));
    previewControls->addWidget(previousPreview);
    previewControls->addWidget(previewCamera, 1);
    previewControls->addWidget(nextPreview);
    // Decode one thumbnail per source, without requiring calibration or a
    // matching point-cloud frame. The LRU cache retains at most eight images,
    // each decoded at no more than 960x540 (about 2 MiB in RGBA).
    QCache<QString, QImage> previewCache(8);
    QVector<QPair<QString, QImage>> previews;
    auto* previewTimer = new QTimer(&dialog);
    previewTimer->setSingleShot(true);
    previewTimer->setInterval(150);
    const auto showPreview = [&] {
        const int index = previewCamera->currentIndex();
        const bool valid = index >= 0 && index < previews.size();
        preview->setProperty("loaded", valid);
        preview->setProperty("directory",
                             valid ? previews[index].first : QString());
        if (valid)
            preview->setPixmap(QPixmap::fromImage(previews[index].second)
                                       .scaled(520, 180, Qt::KeepAspectRatio,
                                               Qt::SmoothTransformation));
        else
            preview->clear();
        previousPreview->setEnabled(previews.size() > 1);
        nextPreview->setEnabled(previews.size() > 1);
        previewCamera->setEnabled(valid);
    };
    const auto refreshPreviews = [&] {
        const QString current = previewCamera->currentData().toString();
        const QSignalBlocker blocked(previewCamera);
        previews.clear();
        previewCamera->clear();
        for (auto* entry : sourcesWidget->findChildren<QWidget*>(
                     QStringLiteral("PointCloudCameraSourceFields"),
                     Qt::FindDirectChildrenOnly)) {
            if (entry->property("removed").toBool()) continue;
            const QString directory =
                    entry->findChild<QLineEdit*>(
                                 QStringLiteral("PointCloudCameraDirectory"))
                            ->text()
                            .trimmed();
            if (directory.isEmpty()) continue;
            QImage image;
            if (const auto* cached = previewCache.object(directory))
                image = *cached;
            else {
                QHash<QString, QString> files;
                QString error;
                if (!listPointCloudCameraImages(directory, &files, &error))
                    continue;
                auto paths = files.values();
                paths.sort(Qt::CaseSensitive);
                for (const auto& path : paths) {
                    QImageReader reader(path);
                    if (reader.size().isValid())
                        reader.setScaledSize(reader.size().scaled(
                                960, 540, Qt::KeepAspectRatio));
                    image = reader.read();
                    if (!image.isNull()) break;
                }
                if (image.isNull()) continue;
                previewCache.insert(directory, new QImage(image));
            }
            previews.append(qMakePair(directory, image));
            const QString title =
                    QFileInfo(QDir(directory).absolutePath()).fileName();
            previewCamera->addItem(title.isEmpty() ? directory : title,
                                   directory);
            previewCamera->setItemData(previewCamera->count() - 1, directory,
                                       Qt::ToolTipRole);
        }
        const int index = previewCamera->findData(current);
        if (index >= 0) previewCamera->setCurrentIndex(index);
        showPreview();
    };
    connect(previewTimer, &QTimer::timeout, &dialog, refreshPreviews);
    connect(previewCamera, QOverload<int>::of(&QComboBox::currentIndexChanged),
            &dialog, showPreview);
    const auto stepPreview = [&](int delta) {
        if (previews.size() > 1)
            previewCamera->setCurrentIndex(
                    (previewCamera->currentIndex() + delta + previews.size()) %
                    previews.size());
    };
    connect(previousPreview, &QToolButton::clicked, &dialog,
            [&] { stepPreview(-1); });
    connect(nextPreview, &QToolButton::clicked, &dialog,
            [&] { stepPreview(1); });
    auto* add = new QPushButton(tr("Add camera"), &dialog);
    add->setObjectName(QStringLiteral("PointCloudCameraAdd"));
    add->setAutoDefault(false);
    form->addWidget(add);
    const auto addSource = [&](const PointCloudCameraSource* source) {
        auto* entry = new QWidget(sourcesWidget);
        entry->setObjectName(QStringLiteral("PointCloudCameraSourceFields"));
        auto* fields = new QFormLayout(entry);
        auto* remove = new QPushButton(tr("Remove camera"), entry);
        remove->setObjectName(QStringLiteral("PointCloudCameraRemove"));
        remove->setAutoDefault(false);
        fields->addRow(remove);
        const auto addField = [&](const QString& title,
                                  const QString& objectName,
                                  const QString& value, bool calibration) {
            auto* row = new QWidget(entry);
            auto* layout = new QHBoxLayout(row);
            layout->setContentsMargins(0, 0, 0, 0);
            auto* field = new QLineEdit(value, row);
            field->setObjectName(objectName);
            field->setAccessibleName(title);
            auto* browse = new QPushButton(tr("Browse"), row);
            browse->setAutoDefault(false);
            layout->addWidget(field, 1);
            layout->addWidget(browse);
            fields->addRow(title, row);
            if (!calibration)
                connect(field, &QLineEdit::textChanged, previewTimer,
                        QOverload<>::of(&QTimer::start));
            connect(browse, &QPushButton::clicked, &dialog,
                    [&, field, calibration] {
                        const QString selected =
                                calibration
                                        ? QFileDialog::getOpenFileName(
                                                  &dialog,
                                                  tr("Camera calibration"),
                                                  field->text(),
                                                  tr("Calibration (*.json)"))
                                        : QFileDialog::getExistingDirectory(
                                                  &dialog,
                                                  tr("Camera image directory"),
                                                  field->text());
                        if (!selected.isEmpty()) field->setText(selected);
                    });
        };
        addField(tr("Image directory"),
                 QStringLiteral("PointCloudCameraDirectory"),
                 source ? source->directory : QString(), false);
        addField(tr("Calibration JSON (optional)"),
                 QStringLiteral("PointCloudCameraCalibration"),
                 source ? source->calibrationPath : QString(), true);
        connect(remove, &QPushButton::clicked, &dialog,
                [entry, sourceRows, previewTimer] {
                    sourceRows->removeWidget(entry);
                    entry->setProperty("removed", true);
                    entry->hide();
                    entry->deleteLater();
                    previewTimer->start();
                });
        sourceRows->addWidget(entry);
    };
    if (m_cameraPanel->sources().isEmpty())
        addSource(nullptr);
    else
        for (const auto& source : m_cameraPanel->sources()) addSource(&source);
    connect(add, &QPushButton::clicked, &dialog, [&] { addSource(nullptr); });
    form->addWidget(preview);
    form->addLayout(previewControls);
    refreshPreviews();
    auto* overlay =
            new QCheckBox(tr("Show projected points and cuboids"), &dialog);
    overlay->setChecked(!m_cameraPanel || m_cameraPanel->overlayEnabled());
    form->addWidget(overlay);
    auto* validationError = new QLabel(&dialog);
    validationError->setObjectName(
            QStringLiteral("PointCloudCameraConfigurationError"));
    validationError->setWordWrap(true);
    validationError->setTextFormat(Qt::PlainText);
    form->addWidget(validationError);
    auto* buttons = new QDialogButtonBox(
            QDialogButtonBox::Ok | QDialogButtonBox::Cancel, &dialog);
    form->addWidget(buttons);
    QVector<PointCloudCameraSource> configuration;
    bool validated = false;
    connect(buttons, &QDialogButtonBox::accepted, &dialog, [&] {
        QVector<PointCloudCameraSource> sources;
        for (auto* entry : sourcesWidget->findChildren<QWidget*>(
                     QStringLiteral("PointCloudCameraSourceFields"),
                     Qt::FindDirectChildrenOnly)) {
            if (entry->property("removed").toBool()) continue;
            const QString directory =
                    entry->findChild<QLineEdit*>(
                                 QStringLiteral("PointCloudCameraDirectory"))
                            ->text()
                            .trimmed();
            const QString calibration =
                    entry->findChild<QLineEdit*>(
                                 QStringLiteral("PointCloudCameraCalibration"))
                            ->text()
                            .trimmed();
            if (directory.isEmpty()) continue;
            PointCloudCameraSource source;
            QString error;
            if (!loadPointCloudCameraSource(directory, calibration, &source,
                                            &error)) {
                validationError->setText(error);
                scroll->ensureWidgetVisible(entry);
                return;
            }
            sources.append(source);
        }
        configuration = std::move(sources);
        validated = true;
        dialog.accept();
    });
    connect(buttons, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);
    if (dialog.exec() != QDialog::Accepted || !validated) return;
    m_cameraPanel->setSources(configuration);
    m_cameraPanel->setOverlayEnabled(overlay->isChecked());
    m_cameraToggle->setEnabled(!configuration.isEmpty());
    m_cameraToggle->setChecked(!configuration.isEmpty());
    updateCamera();
}

void PointCloudWindow::updateCamera() {
    if (!m_cameraPanel) return;
    m_cameraPanel->setFrame(&m_document.frame());
    QHash<quint16, QColor> pointColors;
    const auto& segmentationColors =
            m_detectionClassesActive ? m_otherClassColors : m_classColors;
    for (auto it = segmentationColors.cbegin(); it != segmentationColors.cend();
         ++it)
        pointColors.insert(it.key(), it.value());
    m_cameraPanel->setPointColorStyle(
            PointCloudColorMode(m_colorMode ? m_colorMode->currentIndex() : 0),
            pointColors);
    m_cameraPanel->setCreationPreview(m_view->creationPreview());
    QHash<int, QColor> colors;
    const auto& detectionColors =
            m_detectionClassesActive ? m_classColors : m_otherClassColors;
    for (auto it = detectionColors.cbegin(); it != detectionColors.cend(); ++it)
        colors.insert(it.key(), it.value());
    m_cameraPanel->setCuboidStyle(colors, m_selectedCuboid,
                                  m_detectionClassesActive);
    QSet<int> hiddenCuboids = m_view->hiddenCuboids;
    const auto& hiddenClasses = m_detectionClassesActive
                                        ? m_hiddenSemantics
                                        : m_otherHiddenSemantics;
    for (const auto& box : m_document.frame().cuboids)
        if (hiddenClasses.contains(quint16(box.classId)))
            hiddenCuboids.insert(box.id);
    m_cameraPanel->setVisibility(m_detectionClassesActive
                                         ? m_otherHiddenSemantics
                                         : m_hiddenSemantics,
                                 m_hiddenInstances, hiddenCuboids);
}

void PointCloudWindow::openFile(const QString& path) {
    if (m_loading) return;
    const QFileInfo info(path);
    QDir directory = info.dir();
    if (directory.dirName() != QStringLiteral("velodyne") &&
        directory.exists(QStringLiteral("velodyne")))
        directory.cd(QStringLiteral("velodyne"));
    const QStringList files = directory.entryList(
            {QStringLiteral("*.bin"), QStringLiteral("*.ply")}, QDir::Files,
            QDir::Name);
    QStringList candidates;
    for (const QString& name : files) {
        candidates.append(directory.filePath(name));
    }
    const QString targetPath = info.canonicalFilePath().isEmpty()
                                       ? info.absoluteFilePath()
                                       : info.canonicalFilePath();
    int index = -1;
    for (int row = 0; row < candidates.size(); ++row) {
        const QFileInfo candidate(candidates[row]);
        const QString candidatePath = candidate.canonicalFilePath().isEmpty()
                                              ? candidate.absoluteFilePath()
                                              : candidate.canonicalFilePath();
        if (candidatePath == targetPath) {
            index = row;
            break;
        }
    }
    if (index >= 0) loadFrame(candidates, index);
}

void PointCloudWindow::openDirectory() {
    const QString path = QFileDialog::getExistingDirectory(
            this, tr("Open point cloud directory"));
    if (path.isEmpty()) return;
    QDir directory(path);
    if (directory.exists(QStringLiteral("velodyne")))
        directory.cd(QStringLiteral("velodyne"));
    const QStringList files = directory.entryList(
            {QStringLiteral("*.bin"), QStringLiteral("*.ply")}, QDir::Files,
            QDir::Name);
    if (files.isEmpty()) {
        QMessageBox::warning(this, windowTitle(),
                             tr("No BIN or PLY files found"));
        return;
    }
    openFile(directory.filePath(files.first()));
}

void PointCloudWindow::selectFrame(int index) {
    if (m_changingFile || index < 0 || index >= m_files.size()) return;
    if (!loadFrame(m_files, index)) {
        const QScopedValueRollback<bool> changing(m_changingFile, true);
        m_fileList->setCurrentRow(m_files.indexOf(m_document.frame().path));
    }
}

namespace {
QString framePrefetchKey(const QString& path, const QString& output) {
    // A cached frame includes annotations: reject results produced before a
    // save/import or an external file change, including optional sidecars.
    QString key =
            QFileInfo(path).absoluteFilePath() + QLatin1Char('|') + output;
    const QFileInfo source(path);
    const QDir base =
            source.dir().dirName() == QStringLiteral("velodyne")
                    ? QDir(source.dir().absolutePath() + QStringLiteral("/.."))
                    : source.dir();
    const QString direct =
            source.dir().filePath(source.completeBaseName() + ".label");
    const QString sibling =
            base.filePath("labels/" + source.completeBaseName() + ".label");
    for (const QString& candidate :
         {path, output, direct, sibling, pointCloudCuboidPath(output),
          pointCloudCuboidPath(direct), pointCloudCuboidPath(sibling)}) {
        const QFileInfo info(candidate);
        key += QLatin1Char('|') + QString::number(info.exists()) +
               QLatin1Char(':') + QString::number(info.size()) +
               QLatin1Char(':') +
               QString::number(info.lastModified().toMSecsSinceEpoch());
    }
    return key;
}
}  // namespace

bool PointCloudWindow::loadFrame(const QStringList& files, int index) {
    if (m_loading || index < 0 || index >= files.size() || !confirmSaved())
        return false;
    const QScopedValueRollback<bool> loading(m_loading, true);
    const QString path = files[index];
    const QByteArray snapshot = datasetSnapshot();
    PointCloudFrame frame;
    QString error;
    const QString output =
            m_outputDirectory.isEmpty()
                    ? QString()
                    : QDir(m_outputDirectory)
                              .filePath(QFileInfo(path).completeBaseName() +
                                        QStringLiteral(".label"));
    const QString cacheKey = framePrefetchKey(path, output);
    auto cached = m_prefetch.find(cacheKey);
    if (cached != m_prefetch.end()) {
        frame = std::move(cached.value());
        m_prefetch.erase(cached);
    } else {
        QProgressDialog progress(tr("Loading and validating point cloud…"),
                                 tr("Cancel"), 0, 0, this);
        progress.setObjectName(QStringLiteral("PointCloudLoadProgress"));
        progress.setWindowModality(Qt::WindowModal);
        progress.setMinimumDuration(500);
        progress.setAutoClose(false);
        m_loadProgress = &progress;
        QVector<QPointer<QWidget>> disabled;
        const auto disable = [&](QWidget* widget) {
            if (widget && widget->isEnabled()) {
                disabled.append(widget);
                widget->setEnabled(false);
            }
        };
        disable(centralWidget());
        disable(menuBar());
        for (auto* toolbar : findChildren<QToolBar*>()) disable(toolbar);
        std::atomic<bool> cancelled{false};
        connect(&progress, &QProgressDialog::canceled, this,
                [&] { cancelled.store(true); });
        QTimer cancellationPoll;
        connect(&cancellationPoll, &QTimer::timeout, this, [&] {
            if (progress.wasCanceled()) cancelled.store(true);
        });
        cancellationPoll.start(10);
        bool success = false;
        // One candidate frame, plus bounded parser staging; no worker touches
        // Qt UI or the current document. wait() joins all accesses before local
        // teardown.
        std::unique_ptr<QThread> reader(QThread::create([&] {
            try {
                success = loadPointCloudFrame(path, &frame, &error, output,
                                              [&] { return cancelled.load(); });
            } catch (const std::exception& exception) {
                error = QString::fromUtf8(exception.what());
            } catch (...) {
                error = tr("Point-cloud loading failed");
            }
        }));
        QEventLoop loop;
        connect(reader.get(), &QThread::finished, &loop, &QEventLoop::quit);
        reader->start();
        progress.setValue(0);
        loop.exec();
        reader->wait();
        cancellationPoll.stop();
        const bool wasCancelled = cancelled.load() || progress.wasCanceled();
        m_loadProgress = nullptr;
        progress.close();
        for (const auto& widget : disabled)
            if (widget) widget->setEnabled(true);
        if (wasCancelled) {
            m_status->setText(tr("Point-cloud loading cancelled"));
            return false;
        }
        if (snapshot != datasetSnapshot()) {
            m_status->setText(
                    tr("The point-cloud workspace changed during loading"));
            return false;
        }
        if (!success) {
            QMessageBox::warning(this, windowTitle(), error);
            return false;
        }
    }
    m_document.setFrame(std::move(frame));
    {
        const QScopedValueRollback<bool> changing(m_changingFile, true);
        m_files = files;
        m_fileList->clear();
        for (const auto& candidate : m_files)
            m_fileList->addItem(QFileInfo(candidate).fileName());
        m_fileList->setCurrentRow(index);
    }
    m_view->setFrame(&m_document.frame());
    refresh();
    updateCamera();
    schedulePrefetch(index);
    return true;
}

PointCloudWindow::~PointCloudWindow() {
    // Join plugin-owned code before the window and plugin may be unloaded.
    m_prefetchCancelled.store(true);
    for (auto& job : m_prefetchJobs) job.waitForFinished();
}

void PointCloudWindow::schedulePrefetch(int index) {
    m_prefetchJobs.erase(
            std::remove_if(m_prefetchJobs.begin(), m_prefetchJobs.end(),
                           [](const auto& job) { return job.isFinished(); }),
            m_prefetchJobs.end());
    for (int offset : {-1, 1, -2, 2}) {
        const int target = index + offset;
        if (target < 0 || target >= m_files.size()) continue;
        const QString path = m_files[target];
        const QString output =
                m_outputDirectory.isEmpty()
                        ? QString()
                        : QDir(m_outputDirectory)
                                  .filePath(QFileInfo(path).completeBaseName() +
                                            ".label");
        const QString key = framePrefetchKey(path, output);
        if (path == m_document.frame().path || m_prefetch.contains(key) ||
            m_prefetching.contains(key) || m_prefetching.size() >= 4)
            continue;
        m_prefetching.insert(key);
        auto* watcher =
                new QFutureWatcher<QPair<PointCloudFrame, QString>>(this);
        connect(watcher,
                &QFutureWatcher<QPair<PointCloudFrame, QString>>::finished,
                this, [this, watcher, key, path, output]() {
                    watcher->deleteLater();
                    m_prefetching.remove(key);
                    const auto result = watcher->result();
                    if (!result.second.isEmpty() ||
                        key != framePrefetchKey(path, output) ||
                        !m_files.contains(path))
                        return;
                    while (m_prefetch.size() >= 4)
                        m_prefetch.erase(m_prefetch.begin());
                    m_prefetch.insert(key, result.first);
                });
        auto future = QtConcurrent::run([path, output,
                                         cancelled = &m_prefetchCancelled]() {
            PointCloudFrame frame;
            QString error;
            try {
                if (!loadPointCloudFrame(
                            path, &frame, &error, output,
                            [cancelled] { return cancelled->load(); }) &&
                    error.isEmpty())
                    error = QStringLiteral("Point-cloud prefetch failed");
            } catch (const std::exception& exception) {
                error = QString::fromUtf8(exception.what());
            } catch (...) {
                error = QStringLiteral("Point-cloud prefetch failed");
            }
            return qMakePair(frame, error);
        });
        m_prefetchJobs.append(future);
        watcher->setFuture(future);
    }
}

void PointCloudWindow::save() {
    if (m_document.frame().path.isEmpty()) return;
    QString error;
    saveAnnotations(m_document.frame().labelPath);
}

void PointCloudWindow::saveAs() {
    if (m_document.frame().path.isEmpty()) return;
    const QString path = QFileDialog::getSaveFileName(
            this, tr("Save point labels"), m_document.frame().labelPath,
            tr("Point labels (*.label)"));
    if (path.isEmpty()) return;
    QString error;
    saveAnnotations(path);
}

bool PointCloudWindow::confirmSaved() {
    if (!m_document.dirty() && !classConfigDirty()) return true;
    const auto choice = QMessageBox::question(
            this, windowTitle(),
            tr("Labels or class definitions have unsaved changes. Save before "
               "leaving?"),
            QMessageBox::Save | QMessageBox::Discard | QMessageBox::Cancel,
            QMessageBox::Save);
    if (choice == QMessageBox::Cancel) return false;
    if (choice == QMessageBox::Discard) return true;
    if (m_document.dirty()) {
        save();
        if (m_document.dirty()) return false;
    }
    if (classConfigDirty()) {
        saveClasses();
        if (classConfigDirty()) return false;
    }
    return true;
}

void PointCloudWindow::applySelection(const QVector<int>& indices) {
    if (m_task->currentIndex() == 1) return;
    bool ok = false;
    int semantic = m_class->currentData().toInt(&ok);
    if (!ok) semantic = m_class->currentText().split(' ').first().toInt(&ok);
    if (!ok || semantic < 0 || semantic > 65535) {
        QMessageBox::warning(this, windowTitle(),
                             tr("Semantic ID must be from 0 to 65535"));
        return;
    }
    QString error;
    const auto operation =
            PointCloudDocument::Operation(m_operation->currentData().toInt());
    // Cuboid creation works on the raw selection bounds: an axis-aligned
    // box around the selected points (upstream cuboid data-model parity;
    // interactive rotation/resizing stays with the 3D viewport).
    if (operation == PointCloudDocument::CreateCuboid) {
        if (indices.isEmpty()) {
            QMessageBox::warning(this, windowTitle(),
                                 tr("Select points to create a cuboid"));
            return;
        }
        if (semantic < 1) {
            QMessageBox::warning(this, windowTitle(),
                                 tr("Cuboids need a class ID of 1 or higher"));
            return;
        }
        QVector3D low = m_document.frame().points[indices.first()].toVector3D();
        QVector3D high = low;
        for (int index : indices) {
            const QVector3D point =
                    m_document.frame().points[index].toVector3D();
            low.setX(std::min(low.x(), point.x()));
            low.setY(std::min(low.y(), point.y()));
            low.setZ(std::min(low.z(), point.z()));
            high.setX(std::max(high.x(), point.x()));
            high.setY(std::max(high.y(), point.y()));
            high.setZ(std::max(high.z(), point.z()));
        }
        PointCloudCuboid box;
        box.id = 1;
        for (const PointCloudCuboid& existing : m_document.frame().cuboids)
            box.id = std::max(box.id, existing.id + 1);
        box.classId = semantic;
        box.center[0] = double((low.x() + high.x()) * 0.5);
        box.center[1] = double((low.y() + high.y()) * 0.5);
        box.center[2] = double((low.z() + high.z()) * 0.5);
        box.size[0] = std::max(0.01, double(high.x() - low.x()));
        box.size[1] = std::max(0.01, double(high.y() - low.y()));
        box.size[2] = std::max(0.01, double(high.z() - low.z()));
        auto cuboids = m_document.frame().cuboids;
        cuboids.append(box);
        if (!m_document.replaceCuboids(cuboids, &error)) {
            QMessageBox::warning(this, windowTitle(), error);
            return;
        }
        refresh();
        updateCamera();
        m_status->setText(tr("Cuboid %1 created from %2 points")
                                  .arg(box.id)
                                  .arg(indices.size()));
        return;
    }
    if (m_document.operate(indices, quint16(semantic),
                           quint16(m_instance->value()), operation, &error))
        refresh();
    else if (!error.isEmpty())
        QMessageBox::warning(this, windowTitle(), error);
}

void PointCloudWindow::refresh() {
    const auto& frame = m_document.frame();
    m_status->setText(tr("%1 points  |  %2")
                              .arg(frame.points.size())
                              .arg((m_document.dirty() || classConfigDirty())
                                           ? tr("Unsaved")
                                           : frame.labelPath));
    setWindowTitle(tr("Point cloud annotation - %1%2")
                           .arg(QFileInfo(frame.path).fileName(),
                                (m_document.dirty() || classConfigDirty())
                                        ? QStringLiteral(" *")
                                        : QString()));
    QSet<quint16> semantics;
    QSet<quint16> instances;
    QHash<quint16, int> semanticCounts;
    if (m_detectionClassesActive)
        for (const auto& cuboid : frame.cuboids) {
            semantics.insert(quint16(cuboid.classId));
            ++semanticCounts[quint16(cuboid.classId)];
        }
    for (quint32 label : frame.labels) {
        const quint16 semantic = quint16(label & 0xffffu);
        if (!m_detectionClassesActive) semantics.insert(semantic);
        if (!m_detectionClassesActive) ++semanticCounts[semantic];
        if (label >> 16) instances.insert(quint16(label >> 16));
    }
    for (auto it = m_classNames.cbegin(); it != m_classNames.cend(); ++it)
        semantics.insert(it.key());
    int selectedListId = -1;
    if (const auto* item = m_classList->currentItem())
        selectedListId = item->data(Qt::UserRole).toInt();
    {
        const QSignalBlocker blocker(m_classList);
        m_classList->clear();
        QList<quint16> ids = semantics.values();
        std::sort(ids.begin(), ids.end());
        for (quint16 id : ids) {
            const QString name = m_classNames.value(id, tr("Class %1").arg(id));
            auto* item = new QListWidgetItem(
                    (m_detectionClassesActive ? tr("%1 (%2) - %3 objects")
                                              : tr("%1 (%2) - %3 points"))
                            .arg(name)
                            .arg(id)
                            .arg(semanticCounts.value(id)),
                    m_classList);
            item->setData(Qt::UserRole, int(id));
            const QColor color = m_classColors.value(id, idColor(id));
            item->setBackground(color);
            item->setForeground(qGray(color.rgb()) < 128 ? Qt::white
                                                         : Qt::black);
            item->setFlags(item->flags() | Qt::ItemIsUserCheckable);
            item->setCheckState(m_hiddenSemantics.contains(id) ? Qt::Unchecked
                                                               : Qt::Checked);
        }
    }
    if (selectedListId >= 0) {
        for (int row = 0; row < m_classList->count(); ++row) {
            if (m_classList->item(row)->data(Qt::UserRole).toInt() ==
                selectedListId) {
                m_classList->setCurrentRow(row);
                break;
            }
        }
    }
    {
        const QSignalBlocker blocker(m_instanceList);
        m_instanceList->clear();
        QList<quint16> ids = instances.values();
        std::sort(ids.begin(), ids.end());
        for (quint16 id : ids) {
            auto* item = new QListWidgetItem(tr("Instance %1").arg(id),
                                             m_instanceList);
            item->setData(Qt::UserRole, int(id));
            item->setFlags(item->flags() | Qt::ItemIsUserCheckable);
            item->setCheckState(m_hiddenInstances.contains(id) ? Qt::Unchecked
                                                               : Qt::Checked);
        }
    }
    updateVisibility();
    bool classIdOk = false;
    const quint16 selectedClass =
            quint16(m_class->currentData().toInt(&classIdOk));
    refreshClassCombo(classIdOk ? selectedClass : 0);
    QHash<quint16, QColor> colors;
    for (auto it = m_classColors.cbegin(); it != m_classColors.cend(); ++it)
        colors.insert(it.key(), it.value());
    m_view->setSemanticColors(colors);
    QHash<quint16, QColor> pointColors;
    const auto& segmentationColors =
            m_detectionClassesActive ? m_otherClassColors : m_classColors;
    for (auto it = segmentationColors.cbegin(); it != segmentationColors.cend();
         ++it)
        pointColors.insert(it.key(), it.value());
    m_view->setPointSemanticColors(pointColors);
    if (m_colorMode) {
        auto* model = qobject_cast<QStandardItemModel*>(m_colorMode->model());
        if (model) model->item(1)->setEnabled(m_document.frame().hasIntensity);
        if (!m_document.frame().hasIntensity &&
            m_colorMode->currentIndex() == 1)
            m_colorMode->setCurrentIndex(m_detectionClassesActive ? 2 : 0);
    }
    m_view->classNames = m_classNames;
    m_view->updateColors();
    refreshCuboids();
}

void PointCloudWindow::updateVisibility() {
    QSet<quint16> hiddenSemantics;
    QSet<quint16> hiddenInstances;
    for (int row = 0; row < m_classList->count(); ++row) {
        auto* item = m_classList->item(row);
        if (item->checkState() == Qt::Unchecked)
            hiddenSemantics.insert(quint16(item->data(Qt::UserRole).toInt()));
    }
    for (int row = 0; row < m_instanceList->count(); ++row) {
        auto* item = m_instanceList->item(row);
        if (item->checkState() == Qt::Unchecked)
            hiddenInstances.insert(quint16(item->data(Qt::UserRole).toInt()));
    }
    m_hiddenSemantics = hiddenSemantics;
    m_hiddenInstances = hiddenInstances;
    m_view->setVisibility(hiddenSemantics, hiddenInstances);
    for (auto* view : m_orthographic) view->update();
    updateCamera();
}

void PointCloudWindow::refreshClassCombo(quint16 selected) {
    const QSignalBlocker blocker(m_class);
    m_class->clear();
    for (auto it = m_classNames.cbegin(); it != m_classNames.cend(); ++it)
        m_class->addItem(tr("%1  (%2)").arg(it.value()).arg(it.key()),
                         it.key());
    if (m_detectionClassesActive)
        for (const auto& cuboid : m_document.frame().cuboids)
            if (m_class->findData(cuboid.classId) < 0)
                m_class->addItem(tr("Class %1").arg(cuboid.classId),
                                 cuboid.classId);
    const int row = m_class->findData(selected);
    if (row >= 0) m_class->setCurrentIndex(row);
}

void PointCloudWindow::loadClasses() {
    const QString path = QFileDialog::getOpenFileName(
            this, tr("Load class definitions"), QString(),
            tr("Class definitions (*.json)"));
    if (path.isEmpty()) return;
    if (classConfigDirty()) {
        const auto choice = QMessageBox::question(
                this, windowTitle(),
                tr("Replace unsaved class definitions? Point labels retain "
                   "their original IDs."),
                QMessageBox::Save | QMessageBox::Discard | QMessageBox::Cancel,
                QMessageBox::Cancel);
        if (choice == QMessageBox::Cancel) return;
        if (choice == QMessageBox::Save) {
            saveClasses();
            if (classConfigDirty()) return;
        }
    }
    QString error;
    if (!loadClassDefinitions(path, &error))
        QMessageBox::warning(this, windowTitle(), error);
}

bool PointCloudWindow::loadClassDefinitions(const QString& path,
                                            QString* error) {
    if (m_loading)
        return windowFailure(error,
                             tr("Wait for point-cloud loading to finish"));
    if (error) error->clear();
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) {
        return windowFailure(error, file.errorString());
    }
    QJsonParseError parseError;
    const QJsonDocument document =
            QJsonDocument::fromJson(file.readAll(), &parseError);
    if (parseError.error != QJsonParseError::NoError || !document.isObject()) {
        return windowFailure(error, tr("Invalid class configuration JSON"));
    }
    const QJsonObject root = document.object();
    if (root.value(QStringLiteral("version")).toDouble(-1) != 1)
        return windowFailure(error,
                             tr("Class configuration requires version 1"));
    QMap<quint16, QString> segmentationNames{{0, QStringLiteral("Unlabeled")}},
            detectionNames;
    QMap<quint16, QColor> segmentationColors{
            {0, QColor(QStringLiteral("#808080"))}},
            detectionColors;
    if (root.contains(QStringLiteral("classes"))) {
        if (!readClassSection(root.value(QStringLiteral("classes")), false,
                              &segmentationNames, &segmentationColors, error))
            return false;
        // Legacy plugin files used a single class table for both tasks.
        detectionNames = segmentationNames;
        detectionColors = segmentationColors;
        detectionNames.remove(0);
        detectionColors.remove(0);
    } else {
        if (!root.contains(QStringLiteral("detection")) &&
            !root.contains(QStringLiteral("segmentation")))
            return windowFailure(error,
                                 tr("Class configuration requires detection or "
                                    "segmentation classes"));
        if (root.contains(QStringLiteral("detection")) &&
            !readClassSection(root.value(QStringLiteral("detection"))
                                      .toObject()
                                      .value(QStringLiteral("classes")),
                              true, &detectionNames, &detectionColors, error))
            return false;
        if (root.contains(QStringLiteral("segmentation")) &&
            !readClassSection(root.value(QStringLiteral("segmentation"))
                                      .toObject()
                                      .value(QStringLiteral("classes")),
                              false, &segmentationNames, &segmentationColors,
                              error))
            return false;
    }
    const int selected = m_class->currentData().toInt();
    m_classNames =
            m_detectionClassesActive ? detectionNames : segmentationNames;
    m_classColors =
            m_detectionClassesActive ? detectionColors : segmentationColors;
    m_otherClassNames =
            m_detectionClassesActive ? segmentationNames : detectionNames;
    m_otherClassColors =
            m_detectionClassesActive ? segmentationColors : detectionColors;
    m_savedClassNames = m_classNames;
    m_savedClassColors = m_classColors;
    m_otherSavedClassNames = m_otherClassNames;
    m_otherSavedClassColors = m_otherClassColors;
    m_classConfigPath = QFileInfo(path).absoluteFilePath();
    refreshClassCombo(quint16(selected));
    refresh();
    return true;
}

bool PointCloudWindow::classConfigDirty() const {
    return m_classNames != m_savedClassNames ||
           m_classColors != m_savedClassColors ||
           m_otherClassNames != m_otherSavedClassNames ||
           m_otherClassColors != m_otherSavedClassColors;
}

bool PointCloudWindow::writeClassDefinitions(const QString& path,
                                             QString* error) const {
    if (error) error->clear();
    const QFileInfo destination(path);
    const QString resolved = destination.canonicalFilePath().isEmpty()
                                     ? destination.absoluteFilePath()
                                     : destination.canonicalFilePath();
    // A class file must not replace another frame's cuboid sidecar, including
    // one reached through a symlink with an unrelated display name.
    if (resolved.endsWith(QStringLiteral(".cuboids.json"),
                          Qt::CaseInsensitive) ||
        QFileInfo(resolved).suffix().compare(QStringLiteral("json"),
                                             Qt::CaseInsensitive) != 0)
        return windowFailure(
                error, tr("Class definitions must use a separate JSON file."));
    if (path.isEmpty() ||
        QFileInfo(path).suffix().compare(QStringLiteral("json"),
                                         Qt::CaseInsensitive) != 0 ||
        QFileInfo(path).absoluteFilePath() ==
                QFileInfo(m_document.frame().path).absoluteFilePath() ||
        QFileInfo(path).absoluteFilePath() ==
                QFileInfo(m_document.frame().labelPath).absoluteFilePath() ||
        QFileInfo(path).absoluteFilePath() ==
                QFileInfo(m_document.frame().cuboidPath).absoluteFilePath() ||
        (!m_document.frame().labelPath.isEmpty() &&
         QFileInfo(path).absoluteFilePath() ==
                 QFileInfo(pointCloudCuboidPath(m_document.frame().labelPath))
                         .absoluteFilePath())) {
        if (error)
            *error = tr("Class definitions must use a separate JSON file.");
        return false;
    }
    const QJsonObject detection =
            m_detectionClassesActive
                    ? classSection(m_classNames, m_classColors)
                    : classSection(m_otherClassNames, m_otherClassColors);
    const QJsonObject segmentation =
            m_detectionClassesActive
                    ? classSection(m_otherClassNames, m_otherClassColors)
                    : classSection(m_classNames, m_classColors);
    QSaveFile output(path);
    const QByteArray data =
            QJsonDocument(
                    QJsonObject{{QStringLiteral("version"), 1},
                                {QStringLiteral("detection"), detection},
                                {QStringLiteral("segmentation"), segmentation}})
                    .toJson(QJsonDocument::Indented);
    if (!output.open(QIODevice::WriteOnly) ||
        output.write(data) != data.size() || !output.commit()) {
        if (error) *error = output.errorString();
        return false;
    }
    return true;
}

void PointCloudWindow::saveClasses() {
    QString path = m_classConfigPath;
    if (path.isEmpty())
        path = QFileDialog::getSaveFileName(this, tr("Save class definitions"),
                                            QString(),
                                            tr("Class definitions (*.json)"));
    if (path.isEmpty()) return;
    if (QFileInfo(path).suffix().compare(QStringLiteral("json"),
                                         Qt::CaseInsensitive) != 0) {
        QMessageBox::warning(this, windowTitle(),
                             tr("Class configuration must use .json"));
        return;
    }
    QString error;
    if (!saveClassDefinitions(path, &error)) {
        QMessageBox::warning(this, windowTitle(), error);
        return;
    }
}

bool PointCloudWindow::saveClassDefinitions(const QString& path,
                                            QString* error) {
    if (!writeClassDefinitions(path, error)) return false;
    m_classConfigPath = QFileInfo(path).absoluteFilePath();
    m_savedClassNames = m_classNames;
    m_savedClassColors = m_classColors;
    m_otherSavedClassNames = m_otherClassNames;
    m_otherSavedClassColors = m_otherClassColors;
    refresh();
    return true;
}

void PointCloudWindow::switchTaskClasses(bool detection) {
    if (detection == m_detectionClassesActive) return;
    m_classNames.swap(m_otherClassNames);
    m_classColors.swap(m_otherClassColors);
    m_savedClassNames.swap(m_otherSavedClassNames);
    m_savedClassColors.swap(m_otherSavedClassColors);
    m_hiddenSemantics.swap(m_otherHiddenSemantics);
    m_detectionClassesActive = detection;
}

void PointCloudWindow::addClass() { editClassDefinition(false); }

void PointCloudWindow::editClass() { editClassDefinition(true); }

void PointCloudWindow::editClassDefinition(bool editing) {
    quint16 selected = 0;
    if (const auto* item = m_classList->currentItem())
        selected = quint16(item->data(Qt::UserRole).toInt());
    else
        selected = quint16(m_class->currentData().toInt());
    if (editing && !m_classNames.contains(selected)) {
        QMessageBox::warning(this, windowTitle(), tr("Select a class to edit"));
        return;
    }

    QDialog dialog(this);
    dialog.setWindowTitle(editing ? tr("Edit class") : tr("New class"));
    dialog.setMinimumWidth(360);
    auto* form = new QFormLayout(&dialog);
    auto* id = new QSpinBox(&dialog);
    id->setRange(m_detectionClassesActive ? 1 : 0, 65535);
    id->setValue(editing ? selected : 1);
    id->setEnabled(!editing);
    auto* name = new QLineEdit(&dialog);
    auto* color = new QLineEdit(&dialog);
    color->setMaxLength(7);
    color->setPlaceholderText(QStringLiteral("#RRGGBB"));
    if (editing) {
        name->setText(
                m_classNames.value(selected, tr("Class %1").arg(selected)));
        color->setText(m_classColors.value(selected, idColor(selected))
                               .name()
                               .toUpper());
    } else {
        QSet<int> used;
        for (auto it = m_classNames.cbegin(); it != m_classNames.cend(); ++it)
            used.insert(it.key());
        int suggested = 1;
        while (suggested <= 65535 && used.contains(suggested)) ++suggested;
        id->setValue(std::min(suggested, 65535));
        name->setText(tr("Class %1").arg(id->value()));
        color->setText(QStringLiteral("#60A5FA"));
    }
    auto* choose = new QPushButton(tr("Choose color"), &dialog);
    connect(choose, &QPushButton::clicked, &dialog, [this, color, &dialog]() {
        const QColor chosen = QColorDialog::getColor(
                QColor(color->text()), &dialog, tr("Choose color"));
        if (chosen.isValid()) color->setText(chosen.name().toUpper());
    });
    auto* colorRow = new QWidget(&dialog);
    auto* colorLayout = new QHBoxLayout(colorRow);
    colorLayout->setContentsMargins(0, 0, 0, 0);
    colorLayout->addWidget(color, 1);
    colorLayout->addWidget(choose);
    form->addRow(tr("Semantic ID"), id);
    form->addRow(tr("Name"), name);
    form->addRow(tr("Color"), colorRow);
    auto* buttons = new QDialogButtonBox(
            QDialogButtonBox::Ok | QDialogButtonBox::Cancel, &dialog);
    form->addRow(buttons);
    connect(buttons, &QDialogButtonBox::accepted, &dialog, &QDialog::accept);
    connect(buttons, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);
    for (;;) {
        if (dialog.exec() != QDialog::Accepted) return;
        const QString className = name->text().trimmed();
        const QString classColor = color->text().trimmed().toUpper();
        const bool validColor =
                QRegularExpression(QStringLiteral("^#[0-9A-F]{6}$"))
                        .match(classColor)
                        .hasMatch();
        const quint16 classId = quint16(id->value());
        if (className.isEmpty()) {
            QMessageBox::warning(&dialog, windowTitle(),
                                 tr("Class name cannot be empty"));
            continue;
        }
        if (!validColor) {
            QMessageBox::warning(&dialog, windowTitle(),
                                 tr("Color must use #RRGGBB"));
            continue;
        }
        if (!editing && m_classNames.contains(classId)) {
            QMessageBox::warning(&dialog, windowTitle(),
                                 tr("Semantic ID already exists"));
            continue;
        }
        bool duplicateName = false;
        for (auto it = m_classNames.cbegin(); it != m_classNames.cend(); ++it)
            duplicateName |= it.value() == className &&
                             (!editing || it.key() != classId);
        if (m_detectionClassesActive && duplicateName) {
            QMessageBox::warning(&dialog, windowTitle(),
                                 tr("Detection class names must be unique"));
            continue;
        }
        m_classNames.insert(classId, className);
        m_classColors.insert(classId, QColor(classColor));
        refreshClassCombo(classId);
        refresh();
        for (int row = 0; row < m_classList->count(); ++row) {
            if (m_classList->item(row)->data(Qt::UserRole).toInt() == classId) {
                m_classList->setCurrentRow(row);
                break;
            }
        }
        return;
    }
}

void PointCloudWindow::removeClass() {
    const auto* item = m_classList->currentItem();
    const quint16 classId = item ? quint16(item->data(Qt::UserRole).toInt())
                                 : quint16(m_class->currentData().toInt());
    if (classId == 0 || !m_classNames.contains(classId)) {
        QMessageBox::warning(this, windowTitle(),
                             tr("Select a removable class"));
        return;
    }
    QVector<int> indices;
    const auto& labels = m_document.frame().labels;
    if (m_detectionClassesActive) {
        QVector<PointCloudCuboid> remaining;
        int removed = 0;
        for (const auto& box : m_document.frame().cuboids) {
            if (box.classId != classId)
                remaining.append(box);
            else {
                if (box.locked) {
                    QMessageBox::warning(this, windowTitle(),
                                         tr("Unlock the objects before "
                                            "removing this class"));
                    return;
                }
                ++removed;
            }
        }
        if (QMessageBox::question(this, windowTitle(),
                                  tr("Delete this class and its %1 objects in "
                                     "the current frame? Point segmentation "
                                     "and other frames are preserved.")
                                          .arg(removed),
                                  QMessageBox::Yes | QMessageBox::No,
                                  QMessageBox::No) != QMessageBox::Yes)
            return;
        m_document.replaceCuboids(remaining);
    } else {
        for (int index = 0; index < labels.size(); ++index)
            if ((labels[index] & 0xffffu) == classId) indices.append(index);
    }
    if (!indices.isEmpty()) {
        const auto choice = QMessageBox::question(
                this, windowTitle(),
                tr("Remove class %1 and clear %2 points in the current frame?")
                        .arg(classId)
                        .arg(indices.size()),
                QMessageBox::Yes | QMessageBox::No, QMessageBox::No);
        if (choice != QMessageBox::Yes) return;
        QString error;
        if (!m_document.operate(indices, 0, 0,
                                PointCloudDocument::AssignSemantic, &error)) {
            QMessageBox::warning(this, windowTitle(), error);
            return;
        }
    }
    m_classNames.remove(classId);
    m_classColors.remove(classId);
    m_hiddenSemantics.remove(classId);
    refresh();
}

void PointCloudWindow::closeEvent(QCloseEvent* event) {
    if (m_loading) {
        if (m_loadProgress) {
            m_loadProgress->cancel();
            emit m_loadProgress->canceled();
        }
        event->ignore();
        return;
    }
    if (confirmSaved())
        event->accept();
    else
        event->ignore();
}

const PointCloudCuboid* PointCloudWindow::selectedCuboid() const {
    for (const auto& box : m_document.frame().cuboids)
        if (box.id == m_selectedCuboid) return &box;
    return nullptr;
}

void PointCloudWindow::commitCuboid(const PointCloudCuboid& box) {
    QString error;
    if (m_document.setCuboid(box, &error))
        m_selectedCuboid = box.id;
    else if (!error.isEmpty())
        QMessageBox::warning(this, windowTitle(), error);
    refresh();
}
void PointCloudWindow::previewCuboid(const PointCloudCuboid* box,
                                     PointCloudCuboidView* source) {
    // Only a small value object changes; all views continue to borrow the
    // immutable point frame. The document/undo stack changes on commit only.
    m_view->setEditPreview(box);
    if (m_cameraPanel) m_cameraPanel->setEditPreview(box);
    for (auto* view : m_orthographic) view->setEditPreview(box, view != source);
}

void PointCloudWindow::setupCuboids(QWidget* center,
                                    QVBoxLayout* layout,
                                    QVBoxLayout* rightLayout) {
    auto* views = new QSplitter(Qt::Horizontal, center);
    views->setObjectName(QStringLiteral("PointCloudOrthographicViews"));
    m_cuboidViews = views;
    for (const auto& spec : {std::pair<int, int>{0, 1}, {0, 2}, {1, 2}}) {
        const QString name = spec.second == 1  ? tr("Top")
                             : spec.first == 0 ? tr("Side")
                                               : tr("Front");
        auto* view =
                new PointCloudCuboidView(spec.first, spec.second, name, views);
        view->setObjectName(QStringLiteral("CuboidView_%1_%2")
                                    .arg(spec.first)
                                    .arg(spec.second));
        view->onExpandRequested = [this, view]() {
            toggleExpandedCuboidView(view);
        };
        view->onEdited = [this](const PointCloudCuboid& box) {
            commitCuboid(box);
        };
        view->onEditPreview = [this, view](const PointCloudCuboid* box) {
            previewCuboid(box, view);
        };
        view->onCancelled = [this]() {
            m_view->setCreating(false);
            for (auto* v : m_orthographic) v->setCreating(false);
        };
        view->onCreated = [this](const CuboidVector& center,
                                 const CuboidVector& size) {
            PointCloudCuboid box;
            box.id = m_document.nextCuboidId();
            box.classId = m_class->currentData().toInt();
            box.center = center;
            box.size = size;
            if (box.classId == 0) {
                statusBar()->showMessage(
                        tr("Select a nonzero class before drawing a cuboid."),
                        4000);
                return;
            }
            m_view->setCreating(false);
            for (auto* v : m_orthographic) v->setCreating(false);
            commitCuboid(box);
        };
        m_orthographic.append(view);
        views->addWidget(view);
    }
    m_view->onCuboidCreated = m_orthographic.first()->onCreated;
    m_view->onCuboidEdited = [this](const PointCloudCuboid& box) {
        commitCuboid(box);
    };
    m_view->onCreationCancelled = [this]() {
        for (auto* view : m_orthographic) view->setCreating(false);
    };
    layout->addWidget(views, 1);
    m_cuboidPanel = new QWidget;
    auto* panel = new QVBoxLayout(m_cuboidPanel);
    panel->setContentsMargins(0, 0, 0, 0);
    panel->addWidget(new QLabel(tr("3D objects")));
    auto* tools = new QToolBar;
    panel->addWidget(tools);
    auto action = [this, tools](const QString& title, const QString& name,
                                const QString& shortcut,
                                std::function<void()> callback) {
        auto* a = tools->addAction(title);
        a->setObjectName(name);
        if (!shortcut.isEmpty()) a->setShortcut(QKeySequence(shortcut));
        connect(a, &QAction::triggered, this, [callback]() { callback(); });
        m_cuboidActions.append(a);
        return a;
    };
    action(tr("Draw cuboid"), "CuboidDraw", "N", [this]() {
        if (m_document.frame().path.isEmpty()) return;
        m_selectedCuboid = 0;
        refreshCuboids();
        m_view->setCreating(true);
        for (auto* view : m_orthographic) view->setCreating(true);
        m_cuboidViews->show();
        statusBar()->showMessage(tr("Double-click a point in the main view, or "
                                    "draw in Top, Side or Front. Esc cancels."),
                                 5000);
    });
    action(tr("Edit object"), "CuboidProperties", "",
           [this]() { editCuboidProperties(); });
    action(tr("Delete"), "CuboidDelete", "Delete", [this]() {
        QString error;
        if (m_document.deleteCuboid(m_selectedCuboid, &error)) {
            m_selectedCuboid = 0;
            refresh();
        } else if (!error.isEmpty())
            statusBar()->showMessage(error, 4000);
    });
    action(tr("Duplicate"), "CuboidDuplicate", "Ctrl+D", [this]() {
        const auto* current = selectedCuboid();
        if (!current) return;
        auto box = *current;
        box.id = m_document.nextCuboidId();
        box.locked = false;
        const auto offset = cuboidTransform(cuboidRotationMatrix(box.rotation),
                                            {{0.5, 0, 0}});
        for (int i = 0; i < 3; ++i) box.center[i] += offset[i];
        commitCuboid(box);
    });
    action(tr("Copy"), "CuboidCopy", "Ctrl+C", [this]() {
        if (const auto* box = selectedCuboid()) {
            m_cuboidClipboard = *box;
            m_hasCuboidClipboard = true;
            refreshCuboids();
        }
    });
    action(tr("Paste"), "CuboidPaste", "Ctrl+V", [this]() {
        if (!m_hasCuboidClipboard || m_document.frame().path.isEmpty()) return;
        auto box = m_cuboidClipboard;
        box.id = m_document.nextCuboidId();
        box.locked = false;
        commitCuboid(box);
    });
    action(tr("Fit to points"), "CuboidFit", "", [this]() {
        const auto* box = selectedCuboid();
        if (!box || box->locked) return;
        PointCloudCuboid fitted;
        QString error;
        if (box->fitted(m_document.frame().points, &fitted, &error))
            commitCuboid(fitted);
        else
            statusBar()->showMessage(error, 4000);
    });
    action(tr("Focus"), "CuboidFocus", "G", [this]() {
        if (const auto* box = selectedCuboid()) m_view->focusCuboid(*box);
        for (auto* view : m_orthographic) view->focusCuboid(true);
    });
    action(tr("Lock / Unlock"), "CuboidLock", "", [this]() {
        if (const auto* current = selectedCuboid()) {
            auto box = *current;
            box.locked = !box.locked;
            commitCuboid(box);
        }
    });
    m_cuboidList = new QListWidget;
    m_cuboidList->setObjectName(QStringLiteral("PointCloudCuboids"));
    m_cuboidList->setContextMenuPolicy(Qt::ActionsContextMenu);
    m_cropDepthAction = new QAction(tr("Crop depth to selected cuboid"), this);
    m_cropDepthAction->setObjectName(QStringLiteral("PointCloudCropDepth"));
    m_cropDepthAction->setCheckable(true);
    m_cropDepthAction->setChecked(true);
    m_cuboidList->addAction(m_cropDepthAction);
    connect(m_cropDepthAction, &QAction::toggled, this,
            &PointCloudWindow::setCuboidDepthCropEnabled);
    panel->addWidget(m_cuboidList, 1);
    connect(m_cuboidList, &QListWidget::currentItemChanged, this,
            [this](QListWidgetItem* item, QListWidgetItem*) {
                m_selectedCuboid = item ? item->data(Qt::UserRole).toInt() : 0;
                refreshCuboids();
            });
    connect(m_cuboidList, &QListWidget::itemDoubleClicked, this,
            [this](QListWidgetItem*) { editCuboidProperties(); });
    connect(m_cuboidList, &QListWidget::itemChanged, this,
            [this](QListWidgetItem* item) {
                const int id = item->data(Qt::UserRole).toInt();
                if (item->checkState() == Qt::Checked)
                    m_view->hiddenCuboids.remove(id);
                else
                    m_view->hiddenCuboids.insert(id);
                m_view->update();
                updateCamera();
            });
    m_view->onCuboidSelected = [this](int id) {
        m_selectedCuboid = id;
        refreshCuboids();
    };
    rightLayout->insertWidget(0, m_cuboidPanel, 1);
    connect(m_task, QOverload<int>::of(&QComboBox::currentIndexChanged), this,
            [this](int index) {
                const bool detection = index == 1;
                if (!detection && m_expandedCuboidView)
                    toggleExpandedCuboidView(m_expandedCuboidView);
                m_taskColorModes[m_detectionClassesActive ? 1 : 0] =
                        m_colorMode->currentIndex();
                m_colorMode->setCurrentIndex(m_taskColorModes[index]);
                switchTaskClasses(detection);
                m_view->detection = detection;
                m_cuboidPanel->setVisible(detection);
                m_cuboidViews->setVisible(detection);
                m_classHeading->setText(detection ? tr("Object class")
                                                  : tr("Semantic class"));
                for (auto* control : m_segmentationControls)
                    control->setVisible(!detection);
                for (auto* action : m_segmentationActions)
                    action->setVisible(!detection);
                if (m_brushSizeControl)
                    m_brushSizeControl->setVisible(!detection);
                for (auto* a : m_cuboidActions)
                    a->setEnabled(detection &&
                                  !m_document.frame().path.isEmpty());
                m_view->setCreating(false);
                for (auto* view : m_orthographic) view->setCreating(false);
                m_view->setTool(PointCloudView::Browse);
                refresh();
                m_view->update();
            });
    m_cuboidViews->hide();
    m_cuboidPanel->hide();
    for (auto* a : m_cuboidActions) a->setEnabled(false);
}

void PointCloudWindow::refreshCuboids() {
    if (!m_cuboidList) return;
    previewCuboid(nullptr);
    const QSignalBlocker blocker(m_cuboidList);
    m_cuboidList->clear();
    if (!selectedCuboid()) m_selectedCuboid = 0;
    for (const auto& box : m_document.frame().cuboids) {
        const QString name = m_classNames.value(
                quint16(box.classId), tr("Class %1").arg(box.classId));
        auto* item = new QListWidgetItem(
                QStringLiteral("#%1  %2%3")
                        .arg(box.id)
                        .arg(name, box.locked ? tr(" (locked)") : QString()),
                m_cuboidList);
        item->setData(Qt::UserRole, box.id);
        item->setFlags(item->flags() | Qt::ItemIsUserCheckable);
        item->setCheckState(m_view->hiddenCuboids.contains(box.id)
                                    ? Qt::Unchecked
                                    : Qt::Checked);
        if (box.id == m_selectedCuboid) m_cuboidList->setCurrentItem(item);
    }
    for (auto* view : m_orthographic) {
        view->setDocument(&m_document, m_selectedCuboid);
        view->setVisiblePointIndices(&m_view->visibleIndices());
        view->setPointColorProvider(
                [this](int index) { return m_view->pointColorAt(index); });
        view->setPointSize(m_pointSize);
        view->setDepthCropEnabled(m_cropDepthAction->isChecked());
    }
    const bool enabled =
            m_task->currentIndex() == 1 && !m_document.frame().path.isEmpty();
    for (auto* a : m_cuboidActions) {
        const QString name = a->objectName();
        a->setEnabled(enabled &&
                      (name == "CuboidDraw" ||
                       (name == "CuboidPaste" ? m_hasCuboidClipboard
                                              : selectedCuboid() != nullptr)));
    }
    m_view->selectedCuboid = m_selectedCuboid;
    m_view->update();
    updateCamera();
}

void PointCloudWindow::setCuboidDepthCropEnabled(bool enabled) {
    for (auto* view : m_orthographic) view->setDepthCropEnabled(enabled);
}

void PointCloudWindow::setPointCloudPointSize(int size) {
    m_pointSize = qBound(1, size, 10);
    m_view->setPointSize(m_pointSize);
    for (auto* view : m_orthographic) view->setPointSize(m_pointSize);
}

void PointCloudWindow::toggleExpandedCuboidView(PointCloudCuboidView* view) {
    auto* splitter = qobject_cast<QSplitter*>(m_cuboidViews);
    if (!splitter || !view) return;
    if (m_expandedCuboidView) {
        auto* previous = m_expandedCuboidView;
        m_expandedCuboidView = nullptr;
        previous->setExpanded(false);
        m_view->show();
        for (auto* member : m_orthographic) member->show();
        if (!m_savedCuboidViewSizes.isEmpty())
            splitter->setSizes(m_savedCuboidViewSizes);
        m_savedCuboidViewSizes.clear();
        if (previous == view) {
            view->setFocus();
            return;
        }
    }
    m_savedCuboidViewSizes = splitter->sizes();
    m_expandedCuboidView = view;
    m_view->hide();
    for (auto* member : m_orthographic) member->setVisible(member == view);
    view->setExpanded(true);
    view->setFocus();
}

void PointCloudWindow::editCuboidProperties() {
    const auto* box = selectedCuboid();
    if (!box) return;
    QDialog dialog(this);
    dialog.setObjectName(QStringLiteral("CuboidPropertiesDialog"));
    dialog.setWindowTitle(tr("Edit object"));
    auto* form = new QFormLayout(&dialog);
    auto* classes = new QComboBox(&dialog);
    for (auto it = m_classNames.cbegin(); it != m_classNames.cend(); ++it)
        if (it.key()) classes->addItem(it.value(), it.key());
    if (classes->findData(box->classId) < 0)
        classes->addItem(tr("Class %1").arg(box->classId), box->classId);
    classes->setCurrentIndex(classes->findData(box->classId));
    classes->setEnabled(!box->locked);
    form->addRow(tr("Object class"), classes);
    connect(classes, QOverload<int>::of(&QComboBox::activated), &dialog,
            [this, classes](int) {
                if (const auto* current = selectedCuboid()) {
                    auto edited = *current;
                    edited.classId = classes->currentData().toInt();
                    commitCuboid(edited);
                }
            });
    QVector<QDoubleSpinBox*> fields;
    const QStringList names{QStringLiteral("center"), QStringLiteral("size"),
                            QStringLiteral("rotation")};
    const QStringList titles{tr("Center"), tr("Size"), tr("Rotation (deg)")};
    for (int row = 0; row < 3; ++row) {
        auto* container = new QWidget(&dialog);
        auto* line = new QHBoxLayout(container);
        line->setContentsMargins(0, 0, 0, 0);
        for (int axis = 0; axis < 3; ++axis) {
            auto* field = new QDoubleSpinBox(container);
            field->setObjectName(
                    QStringLiteral("Cuboid_%1_%2").arg(names[row]).arg(axis));
            field->setDecimals(3);
            field->setRange(row == 1 ? 0.01 : -1e9, 1e9);
            field->setSingleStep(row == 2 ? 1 : 0.1);
            field->setKeyboardTracking(false);
            field->setValue(row == 0   ? box->center[axis]
                            : row == 1 ? box->size[axis]
                                       : box->rotation[axis] * 180 /
                                                 3.141592653589793);
            field->setEnabled(!box->locked);
            fields.append(field);
            line->addWidget(field);
            connect(field, &QDoubleSpinBox::editingFinished, &dialog,
                    [this, field, row, axis]() {
                        if (const auto* current = selectedCuboid()) {
                            if (current->locked) return;
                            auto edited = *current;
                            auto& values = row == 0   ? edited.center
                                           : row == 1 ? edited.size
                                                      : edited.rotation;
                            const double value =
                                    field->value() *
                                    (row == 2 ? 3.141592653589793 / 180 : 1);
                            if (std::abs(values[axis] - value) >
                                0.00051 * (row == 2 ? 3.141592653589793 / 180
                                                    : 1)) {
                                values[axis] = value;
                                commitCuboid(edited);
                            }
                        }
                    });
        }
        form->addRow(titles[row], container);
    }
    auto* occluded = new QCheckBox(tr("Occluded"), &dialog);
    occluded->setChecked(box->occluded);
    occluded->setEnabled(!box->locked);
    auto* locked = new QCheckBox(tr("Locked"), &dialog);
    locked->setChecked(box->locked);
    form->addRow(occluded);
    form->addRow(locked);
    connect(occluded, &QCheckBox::toggled, &dialog, [this](bool value) {
        if (const auto* current = selectedCuboid()) {
            auto edited = *current;
            edited.occluded = value;
            commitCuboid(edited);
        }
    });
    connect(locked, &QCheckBox::toggled, &dialog,
            [this, fields, classes, occluded](bool value) {
                if (const auto* current = selectedCuboid()) {
                    auto edited = *current;
                    edited.locked = value;
                    commitCuboid(edited);
                }
                for (auto* f : fields) f->setEnabled(!value);
                classes->setEnabled(!value);
                occluded->setEnabled(!value);
            });
    auto* buttons = new QDialogButtonBox(QDialogButtonBox::Ok, &dialog);
    connect(buttons, &QDialogButtonBox::accepted, &dialog, &QDialog::accept);
    form->addRow(buttons);
    dialog.exec();
}

bool PointCloudWindow::saveAnnotations(const QString& labelPath) {
    const QString boxesPath = pointCloudCuboidPath(labelPath);
    const auto& frame = m_document.frame();
    const bool saveBoxes = m_document.cuboidsDirty() || frame.cuboidExists ||
                           !frame.cuboids.isEmpty();
    if (saveBoxes && QFileInfo::exists(boxesPath) &&
        (boxesPath != frame.cuboidPath || !frame.cuboidExists) &&
        QMessageBox::question(
                this, windowTitle(),
                tr("Replace the existing cuboid file at %1?").arg(boxesPath),
                QMessageBox::Yes | QMessageBox::No,
                QMessageBox::No) != QMessageBox::Yes)
        return false;
    QString error;
    // Each successful write updates only its own baseline. A later failure
    // keeps the remaining dirty state, so retry/close cannot lose annotations.
    const bool saveLabels =
            m_document.labelsDirty() || frame.labelExists || !saveBoxes;
    if (saveLabels) {
        if (!savePointCloudLabels(frame, &error, labelPath)) {
            QMessageBox::warning(this, windowTitle(), error);
            return false;
        }
        m_document.setLabelPath(labelPath);
        m_document.markSaved();
    }
    if (saveBoxes) {
        if (!savePointCloudCuboids(boxesPath, frame.path, frame.cuboids,
                                   &error)) {
            QMessageBox::warning(this, windowTitle(), error);
            return false;
        }
        m_document.markCuboidsSaved(boxesPath);
    }
    // A detection-only Save As changes the next save destination even when
    // there is no semantic .label file to write.
    if (!saveLabels)
        m_document.setLabelPath(labelPath, QFileInfo::exists(labelPath));
    refresh();
    return true;
}

bool PointCloudWindow::datasetFrames(
        QVector<PointCloudFrame>* frames,
        QString* error,
        const std::function<bool(int, int)>& callback) const {
    if (error) error->clear();
    if (m_files.isEmpty())
        return windowFailure(error, tr("Open a point-cloud sequence first"));
    frames->clear();
    for (int index = 0; index < m_files.size(); ++index) {
        if (callback && !callback(index, m_files.size() * 2))
            return windowFailure(
                    error, tr("Point-cloud dataset operation cancelled."));
        PointCloudFrame frame;
        if (m_files.at(index) == m_document.frame().path)
            frame = m_document.frame();
        else {
            const QString labelPath =
                    m_outputDirectory.isEmpty()
                            ? QString()
                            : QDir(m_outputDirectory)
                                      .filePath(QFileInfo(m_files.at(index))
                                                        .completeBaseName() +
                                                QStringLiteral(".label"));
            if (!loadPointCloudFrame(m_files.at(index), &frame, error,
                                     labelPath))
                return false;
        }
        if (!m_outputDirectory.isEmpty())
            frame.labelPath =
                    QDir(m_outputDirectory)
                            .filePath(QFileInfo(frame.path).completeBaseName() +
                                      QStringLiteral(".label"));
        frames->append(frame);
    }
    return true;
}

QByteArray PointCloudWindow::datasetSnapshot() const {
    QJsonArray boxes;
    for (const auto& box : m_document.frame().cuboids)
        boxes.append(box.toJson());
    const auto detection =
            m_detectionClassesActive
                    ? classSection(m_classNames, m_classColors)
                    : classSection(m_otherClassNames, m_otherClassColors);
    const auto segmentation =
            m_detectionClassesActive
                    ? classSection(m_otherClassNames, m_otherClassColors)
                    : classSection(m_classNames, m_classColors);
    const QJsonObject state{
            {QStringLiteral("path"), m_document.frame().path},
            {QStringLiteral("label_path"), m_document.frame().labelPath},
            {QStringLiteral("files"), QJsonArray::fromStringList(m_files)},
            {QStringLiteral("output"), m_outputDirectory},
            {QStringLiteral("cuboids"), boxes},
            {QStringLiteral("detection"), detection},
            {QStringLiteral("segmentation"), segmentation}};
    return QCryptographicHash::hash(
            QJsonDocument(state).toJson(QJsonDocument::Compact),
            QCryptographicHash::Sha256);
}

bool PointCloudWindow::exportDataset(
        const QString& path,
        const QString& format,
        bool includePointCloud,
        const std::function<bool(int, int)>& callback,
        QString* error) const {
    PointCloudDatasetExportOptions options;
    options.outputPath = path;
    options.format = format;
    options.includePointCloud = includePointCloud;
    if (!datasetFrames(&options.frames, error, callback)) return false;
    const auto& names =
            m_detectionClassesActive ? m_classNames : m_otherClassNames;
    const auto& colors =
            m_detectionClassesActive ? m_classColors : m_otherClassColors;
    for (auto it = names.cbegin(); it != names.cend(); ++it)
        options.classNames.insert(it.key(), it.value());
    for (auto it = colors.cbegin(); it != colors.cend(); ++it)
        options.classColors.insert(it.key(), it.value());
    return exportPointCloudDataset(
            options,
            [&](int current, int total) {
                return !callback || callback(options.frames.size() + current,
                                             options.frames.size() + total);
            },
            error);
}

bool PointCloudWindow::prepareDatasetImport(
        const QString& path,
        const QString& format,
        const QString& configPath,
        PointCloudDatasetImportPlan* plan,
        const std::function<bool(int, int)>& callback,
        QString* error) const {
    if (!plan || configPath.isEmpty())
        return windowFailure(error,
                             tr("Select a class configuration for the import"));
    const QByteArray snapshot = datasetSnapshot();
    QVector<PointCloudFrame> frames;
    if (!datasetFrames(&frames, error, callback)) return false;
    QVector<PointCloudDatasetImportTarget> targets;
    for (const auto& frame : frames)
        targets.append({frame.path, frame.labelPath,
                        pointCloudCuboidPath(frame.labelPath), frame.cuboids,
                        true});
    const auto detection =
            m_detectionClassesActive
                    ? datasetClasses(m_classNames, m_classColors)
                    : datasetClasses(m_otherClassNames, m_otherClassColors);
    const auto segmentation =
            m_detectionClassesActive
                    ? datasetClasses(m_otherClassNames, m_otherClassColors)
                    : datasetClasses(m_classNames, m_classColors);
    if (!preparePointCloudDatasetImport(
                path, format, targets, detection, configPath, plan,
                [&](int current, int total) {
                    return !callback || callback(frames.size() + current,
                                                 frames.size() + total);
                },
                error, &segmentation))
        return false;
    if (snapshot != datasetSnapshot())
        return windowFailure(
                error,
                tr("The point-cloud workspace changed during import preview"));
    plan->consumerState = snapshot;
    return true;
}

bool PointCloudWindow::applyDatasetImport(
        const PointCloudDatasetImportPlan& plan,
        const std::function<bool(int, int)>& callback,
        QString* error) {
    if (m_loading)
        return windowFailure(error,
                             tr("Wait for point-cloud loading to finish"));
    if (plan.consumerState.isEmpty() || plan.consumerState != datasetSnapshot())
        return windowFailure(
                error,
                tr("The point-cloud workspace changed after import preview"));
    PointCloudDocument validation;
    for (const auto& target : plan.targets)
        if (!m_files.contains(target.path) ||
            !validation.replaceCuboids(target.cuboids, error))
            return windowFailure(
                    error,
                    tr("Import plan does not match this point-cloud sequence"));
    bool workspaceChanged = false;
    const auto guardedProgress = [&](int current, int total) {
        if (plan.consumerState != datasetSnapshot()) workspaceChanged = true;
        if (workspaceChanged) return false;
        const bool proceed = !callback || callback(current, total);
        if (plan.consumerState != datasetSnapshot()) workspaceChanged = true;
        return proceed && !workspaceChanged;
    };
    if (!applyPointCloudDatasetImport(plan, guardedProgress, error)) {
        if (workspaceChanged)
            return windowFailure(
                    error,
                    tr("The point-cloud workspace changed during import"));
        return false;
    }
    for (const auto& target : plan.targets) {
        if (target.path != m_document.frame().path) continue;
        m_document.replaceCuboids(target.cuboids);
        m_document.markCuboidsSaved(target.cuboidPath);
        m_document.setLabelPath(target.labelPath,
                                QFileInfo::exists(target.labelPath));
    }
    auto& names = m_detectionClassesActive ? m_classNames : m_otherClassNames;
    auto& colors =
            m_detectionClassesActive ? m_classColors : m_otherClassColors;
    names.clear();
    colors.clear();
    for (const auto& entry : plan.classes) {
        names.insert(quint16(entry.id), entry.name);
        colors.insert(quint16(entry.id), entry.color);
    }
    for (const auto& change : plan.changes) {
        const auto object = QJsonDocument::fromJson(change.after).object();
        if (object.contains(QStringLiteral("detection")) &&
            object.contains(QStringLiteral("segmentation")))
            m_classConfigPath = change.path;
    }
    m_savedClassNames = m_classNames;
    m_savedClassColors = m_classColors;
    m_otherSavedClassNames = m_otherClassNames;
    m_otherSavedClassColors = m_otherClassColors;
    refresh();
    updateCamera();
    return true;
}

void PointCloudWindow::datasetDialog(bool importing) {
    if (m_files.isEmpty()) {
        QMessageBox::warning(this, windowTitle(),
                             tr("Open a point-cloud sequence first"));
        return;
    }
    QDialog dialog(this);
    dialog.setObjectName(QStringLiteral("PointCloudDatasetDialog"));
    dialog.setWindowTitle(importing ? tr("Import 3D cuboids")
                                    : tr("Export 3D cuboids"));
    dialog.setMinimumWidth(560);
    auto* form = new QFormLayout(&dialog);
    auto* format = new QComboBox(&dialog);
    format->setObjectName(QStringLiteral("PointCloudDatasetFormat"));
    format->addItem(QStringLiteral("Datumaro 3D"), QStringLiteral("datumaro"));
    format->addItem(QStringLiteral("KITTI Raw"), QStringLiteral("kitti_raw"));
    format->addItem(QStringLiteral("Sly Point Cloud"),
                    QStringLiteral("sly_pointcloud"));
    form->addRow(tr("Format"), format);
    auto* path = new QLineEdit(&dialog);
    path->setObjectName(QStringLiteral("PointCloudDatasetPath"));
    auto* browse = new QPushButton(tr("Browse"), &dialog);
    auto* pathRow = new QWidget(&dialog);
    auto* pathLayout = new QHBoxLayout(pathRow);
    pathLayout->setContentsMargins(0, 0, 0, 0);
    pathLayout->addWidget(path, 1);
    pathLayout->addWidget(browse);
    form->addRow(tr("ZIP archive"), pathRow);
    connect(browse, &QPushButton::clicked, &dialog, [&, importing] {
        const QString selected =
                importing ? QFileDialog::getOpenFileName(
                                    &dialog, tr("Import point-cloud dataset"),
                                    path->text(), tr("ZIP archive (*.zip)"))
                          : QFileDialog::getSaveFileName(
                                    &dialog, tr("Export point-cloud dataset"),
                                    path->text(), tr("ZIP archive (*.zip)"));
        if (!selected.isEmpty()) path->setText(selected);
    });
    auto* payload = new QCheckBox(tr("Include point-cloud files"), &dialog);
    payload->setObjectName(QStringLiteral("PointCloudDatasetPayload"));
    payload->setChecked(true);
    if (!importing) form->addRow(payload);
    auto* config = new QLineEdit(
            m_classConfigPath.isEmpty()
                    ? QFileInfo(m_document.frame().path)
                              .dir()
                              .filePath(QStringLiteral("classes.json"))
                    : m_classConfigPath,
            &dialog);
    config->setObjectName(QStringLiteral("PointCloudDatasetClasses"));
    if (importing) form->addRow(tr("Class configuration"), config);
    auto* scope = new QLabel(
            importing ? tr("Replace cuboids in matched frames. Point "
                           "segmentation and unmatched frames are preserved.")
                      : tr("Export 3D cuboids from all frames. Point "
                           "segmentation is not included."),
            &dialog);
    scope->setWordWrap(true);
    form->addRow(scope);
    auto* buttons = new QDialogButtonBox(
            QDialogButtonBox::Ok | QDialogButtonBox::Cancel, &dialog);
    form->addRow(buttons);
    connect(buttons, &QDialogButtonBox::accepted, &dialog, &QDialog::accept);
    connect(buttons, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);
    if (dialog.exec() != QDialog::Accepted || path->text().trimmed().isEmpty())
        return;
    const QString archivePath = path->text().trimmed();
    if (!importing && QFileInfo::exists(archivePath) &&
        QMessageBox::question(
                this, windowTitle(),
                tr("Replace the existing archive at %1?").arg(archivePath),
                QMessageBox::Yes | QMessageBox::No,
                QMessageBox::No) != QMessageBox::Yes)
        return;
    QString error;
    PointCloudDatasetImportPlan plan;
    bool ok = false;
    {
        QProgressDialog progress(importing
                                         ? tr("Reading point-cloud annotations")
                                         : tr("Exporting point-cloud dataset"),
                                 tr("Cancel"), 0, 100, this);
        progress.setAutoReset(false);
        progress.setAutoClose(false);
        progress.setWindowModality(Qt::WindowModal);
        progress.setMinimumDuration(0);
        const auto update = [&](int current, int total) {
            progress.setRange(0, total);
            progress.setValue(current);
            QApplication::processEvents();
            return !progress.wasCanceled();
        };
        ok = importing
                     ? prepareDatasetImport(
                               archivePath, format->currentData().toString(),
                               config->text().trimmed(), &plan, update, &error)
                     : exportDataset(archivePath,
                                     format->currentData().toString(),
                                     payload->isChecked(), update, &error);
    }
    if (ok && importing) {
        int objects = 0;
        for (const auto& target : plan.targets)
            objects += target.cuboids.size();
        if (QMessageBox::question(
                    this, windowTitle(),
                    tr("Import %1 cuboids into %2 matched frames, replacing %3 "
                       "objects and adding %4 classes?")
                            .arg(objects)
                            .arg(plan.targets.size())
                            .arg(plan.previousObjects)
                            .arg(plan.newClasses),
                    QMessageBox::Yes | QMessageBox::No,
                    QMessageBox::No) != QMessageBox::Yes)
            return;
        QProgressDialog progress(tr("Applying point-cloud annotations"),
                                 tr("Cancel"), 0, plan.changes.size(), this);
        progress.setAutoReset(false);
        progress.setAutoClose(false);
        progress.setWindowModality(Qt::WindowModal);
        progress.setMinimumDuration(0);
        ok = applyDatasetImport(
                plan,
                [&](int current, int total) {
                    progress.setRange(0, total);
                    progress.setValue(current);
                    QApplication::processEvents();
                    return !progress.wasCanceled();
                },
                &error);
    }
    if (!ok &&
        !error.contains(QStringLiteral("cancelled"), Qt::CaseInsensitive))
        QMessageBox::warning(this, windowTitle(), error);
    if (ok)
        statusBar()->showMessage(
                importing ? tr("Point-cloud annotations imported")
                          : tr("Point-cloud dataset exported"),
                5000);
}
