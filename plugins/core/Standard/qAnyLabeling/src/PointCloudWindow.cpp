// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "PointCloudWindow.h"

#include <QAbstractItemView>
#include <QActionGroup>
#include <QCloseEvent>
#include <QColorDialog>
#include <QComboBox>
#include <QDialog>
#include <QDialogButtonBox>
#include <QDir>
#include <QDockWidget>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QFutureWatcher>
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
#include <QOpenGLFunctions_3_3_Core>
#include <QOpenGLShaderProgram>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLWidget>
#include <QPainter>
#include <QPen>
#include <QPolygonF>
#include <QPushButton>
#include <QRegularExpression>
#include <QSaveFile>
#include <QSet>
#include <QSignalBlocker>
#include <QSpinBox>
#include <QSplitter>
#include <QStatusBar>
#include <QSurfaceFormat>
#include <QToolBar>
#include <QVBoxLayout>
#include <QWheelEvent>
#include <QtConcurrent>
#include <algorithm>
#include <cmath>
#include <functional>

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
}  // namespace

class PointCloudView final : public QOpenGLWidget,
                             protected QOpenGLFunctions_3_3_Core {
public:
    enum Tool { Browse, Brush, Polygon };
    enum ColorMode { Semantic, Intensity, Rgb, Instance };

    explicit PointCloudView(QWidget* parent = nullptr) : QOpenGLWidget(parent) {
        QSurfaceFormat surface = format();
        surface.setVersion(3, 3);
        surface.setProfile(QSurfaceFormat::CoreProfile);
        surface.setDepthBufferSize(24);
        setFormat(surface);
        setFocusPolicy(Qt::StrongFocus);
    }
    std::function<void(const QVector<int>&)> onSelection;
    void setFrame(const PointCloudFrame* frame) {
        m_frame = frame;
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
    void setBrushSize(int size) { m_brushSize = size; }
    void setColorMode(ColorMode mode) {
        m_colorMode = mode;
        updateColors();
    }
    void setVisibility(const QSet<quint16>& hiddenSemantics,
                       const QSet<quint16>& hiddenInstances) {
        m_hiddenSemantics = hiddenSemantics;
        m_hiddenInstances = hiddenInstances;
        m_upload = true;
        update();
    }
    void setSemanticColors(const QHash<quint16, QColor>& colors) {
        m_semanticColors = colors;
        m_upload = true;
        update();
    }
    void setView(int view) {
        m_yaw = view == 2 ? 90.0f : 0.0f;
        m_pitch = view == 0 ? 89.0f : 0.0f;
        update();
    }
    void fit() {
        m_center = QVector3D();
        m_distance = 10.0f;
        if (m_frame && !m_frame->points.isEmpty()) {
            QVector3D low = m_frame->points[0].toVector3D();
            QVector3D high = low;
            for (const QVector4D& point : m_frame->points) {
                const QVector3D p = point.toVector3D();
                low.setX(std::min(low.x(), p.x()));
                low.setY(std::min(low.y(), p.y()));
                low.setZ(std::min(low.z(), p.z()));
                high.setX(std::max(high.x(), p.x()));
                high.setY(std::max(high.y(), p.y()));
                high.setZ(std::max(high.z(), p.z()));
            }
            m_center = (low + high) * 0.5f;
            m_distance = std::max(0.5f, (high - low).length() * 1.5f);
        }
        update();
    }
    void updateColors() {
        m_upload = true;
        update();
    }
    void finishPolygon() {
        if (m_polygon.size() >= 3)
            select([this](const QPointF& p) {
                return m_polygon.containsPoint(p, Qt::OddEvenFill);
            });
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
        glClearColor(0.09f, 0.11f, 0.13f, 1.0f);
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_PROGRAM_POINT_SIZE);
        m_vao.create();
        m_vao.bind();
        m_vbo.create();
        m_vbo.bind();
        m_shader.create();
        m_shader.addShaderFromSourceCode(
                QOpenGLShader::Vertex,
                "#version 330 core\nlayout(location=0) in vec3 position;\n"
                "layout(location=1) in vec3 color;\nuniform mat4 "
                "matrix;\nuniform float size;\n"
                "out vec3 shade;\nvoid "
                "main(){gl_Position=matrix*vec4(position,1);"
                "gl_PointSize=size;shade=color;}");
        m_shader.addShaderFromSourceCode(
                QOpenGLShader::Fragment,
                "#version 330 core\nin vec3 shade;\nout vec4 pixel;\n"
                "void main(){pixel=vec4(shade,1);}");
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
    }
    void paintGL() override {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        if (!m_frame || m_frame->points.isEmpty()) return;
        if (m_upload) upload();
        m_shader.bind();
        m_shader.setUniformValue("matrix", matrix());
        m_shader.setUniformValue("size",
                                 float(m_pointSize * devicePixelRatioF()));
        m_vao.bind();
        glDrawArrays(GL_POINTS, 0, m_visibleIndices.size());
        m_vao.release();
        m_shader.release();
        if (!m_polygon.isEmpty()) {
            QPainter painter(this);
            painter.setPen(QPen(QColor(72, 160, 239), 2));
            painter.drawPolyline(m_polygon);
        }
    }
    void mousePressEvent(QMouseEvent* event) override {
        m_last = mousePosition(event);
        if (m_tool == Polygon && event->button() == Qt::LeftButton) {
            m_polygon.append(m_last);
            update();
            return;
        }
        if (m_tool == Brush && event->button() == Qt::LeftButton) {
            m_brushing = true;
            collectBrush(m_last);
            return;
        }
        m_panning = event->button() == Qt::LeftButton;
        m_orbiting = event->button() == Qt::RightButton;
    }
    void mouseMoveEvent(QMouseEvent* event) override {
        const QPointF current = mousePosition(event);
        const QPointF delta = current - m_last;
        m_last = current;
        if (m_brushing)
            collectBrush(current);
        else if (m_orbiting) {
            m_yaw += float(delta.x()) * 0.4f;
            m_pitch = std::clamp(m_pitch + float(delta.y()) * 0.4f, -89.0f,
                                 89.0f);
            update();
        } else if (m_panning) {
            const float scale = m_distance / float(std::max(1, height()));
            m_center += QVector3D(-float(delta.x()) * scale,
                                  float(delta.y()) * scale, 0);
            update();
        }
    }
    void mouseReleaseEvent(QMouseEvent* event) override {
        if (event->button() == Qt::LeftButton && m_brushing) {
            m_brushing = false;
            QVector<int> indices;
            indices.reserve(m_stroke.size());
            for (int index : m_stroke) indices.append(index);
            m_stroke.clear();
            if (onSelection && !indices.isEmpty()) onSelection(indices);
        }
        m_panning = false;
        m_orbiting = false;
    }
    void wheelEvent(QWheelEvent* event) override {
        m_distance = std::clamp(
                m_distance * std::pow(0.9f, event->angleDelta().y() / 120.0f),
                0.01f, 1000000.0f);
        update();
    }
    void keyPressEvent(QKeyEvent* event) override {
        if (event->key() == Qt::Key_Return || event->key() == Qt::Key_Enter)
            finishPolygon();
        else if (event->key() == Qt::Key_Escape)
            cancelPolygon();
        else
            QOpenGLWidget::keyPressEvent(event);
    }

private:
    QMatrix4x4 matrix() const {
        QMatrix4x4 projection;
        projection.perspective(
                50.0f, float(std::max(1, width())) / std::max(1, height()),
                std::max(0.001f, m_distance / 1000), m_distance * 100);
        QMatrix4x4 view;
        const float yaw = m_yaw * 3.14159265f / 180;
        const float pitch = m_pitch * 3.14159265f / 180;
        const QVector3D eye =
                m_center +
                m_distance * QVector3D(std::cos(pitch) * std::sin(yaw),
                                       std::sin(pitch),
                                       std::cos(pitch) * std::cos(yaw));
        view.lookAt(eye, m_center, QVector3D(0, 1, 0));
        return projection * view;
    }
    void upload() {
        QVector<float> packed;
        m_visibleIndices.clear();
        packed.reserve(m_frame->points.size() * 6);
        for (int i = 0; i < m_frame->points.size(); ++i) {
            const QVector4D& point = m_frame->points[i];
            QColor color;
            const quint32 label = m_frame->labels[i];
            if (m_hiddenSemantics.contains(quint16(label & 0xffffu)) ||
                (quint16(label >> 16) &&
                 m_hiddenInstances.contains(quint16(label >> 16))))
                continue;
            m_visibleIndices.append(i);
            if (m_colorMode == Semantic)
                color = m_semanticColors.value(quint16(label & 0xffff),
                                               idColor(label & 0xffff));
            else if (m_colorMode == Instance)
                color = idColor(label >> 16);
            else if (m_colorMode == Rgb && i < m_frame->rgb.size())
                color = m_frame->rgb[i];
            else if (m_colorMode == Intensity) {
                const int gray =
                        std::isfinite(point.w())
                                ? int(std::clamp(point.w(), 0.0f, 1.0f) * 255)
                                : 128;
                color = QColor(gray, gray, gray);
            } else
                color = idColor(label & 0xffff);
            packed << point.x() << point.y() << point.z() << float(color.redF())
                   << float(color.greenF()) << float(color.blueF());
        }
        m_vbo.bind();
        m_vbo.allocate(packed.constData(), int(packed.size() * sizeof(float)));
        m_vbo.release();
        m_upload = false;
    }
    void collectBrush(const QPointF& at) {
        select(
                [this, at](const QPointF& p) {
                    return QLineF(p, at).length() <= m_brushSize;
                },
                true);
    }
    void select(const std::function<bool(const QPointF&)>& contains,
                bool accumulate = false) {
        if (!m_frame || !onSelection) return;
        const QMatrix4x4 transform = matrix();
        // Upstream selection.py parity: every point owns a point_size²
        // footprint on screen, and the non-through filter keeps a point iff
        // any of its footprint pixels lists it at the exact closest depth
        // (float equality — same expression, same value — so equal-depth
        // ties stay selectable together).
        const int size = std::max(1, m_pointSize);
        QVector<int> selected;
        QHash<quint64, float> closest;
        for (int visible = 0; visible < m_visibleIndices.size(); ++visible) {
            const int i = m_visibleIndices[visible];
            const QVector4D clip =
                    transform * QVector4D(m_frame->points[i].toVector3D(), 1);
            if (clip.w() <= 0) continue;
            const float z = clip.z() / clip.w();
            if (z < -1 || z > 1) continue;
            const QPointF screen((clip.x() / clip.w() + 1) * width() / 2,
                                 (1 - clip.y() / clip.w()) * height() / 2);
            if (!contains(screen)) continue;
            selected.append(i);
            const int originX = int(screen.x());
            const int originY = int(screen.y());
            for (int dy = 0; dy < size; ++dy) {
                for (int dx = 0; dx < size; ++dx) {
                    const int px = originX + dx;
                    const int py = originY + dy;
                    if (px < 0 || px >= width() || py < 0 || py >= height())
                        continue;
                    const quint64 key = quint64(py) * quint64(width()) + px;
                    if (!closest.contains(key) || z < closest.value(key))
                        closest.insert(key, z);
                }
            }
        }
        if (!m_through) {
            selected.erase(
                    std::remove_if(
                            selected.begin(), selected.end(),
                            [this, &transform, &closest, size](int i) {
                                const QVector4D clip =
                                        transform *
                                        QVector4D(
                                                m_frame->points[i].toVector3D(),
                                                1);
                                if (clip.w() <= 0) return true;
                                const QPointF screen(
                                        (clip.x() / clip.w() + 1) * width() / 2,
                                        (1 - clip.y() / clip.w()) * height() /
                                                2);
                                const float z = clip.z() / clip.w();
                                const int originX = int(screen.x());
                                const int originY = int(screen.y());
                                for (int dy = 0; dy < size; ++dy) {
                                    for (int dx = 0; dx < size; ++dx) {
                                        const int px = originX + dx;
                                        const int py = originY + dy;
                                        if (px < 0 || px >= width() || py < 0 ||
                                            py >= height())
                                            continue;
                                        const quint64 key =
                                                quint64(py) * quint64(width()) +
                                                px;
                                        if (closest.contains(key) &&
                                            closest.value(key) == z)
                                            return false;
                                    }
                                }
                                return true;
                            }),
                    selected.end());
        }
        if (accumulate) {
            for (int index : selected) m_stroke.insert(index);
        } else if (!selected.isEmpty())
            onSelection(selected);
    }

    const PointCloudFrame* m_frame = nullptr;
    QOpenGLVertexArrayObject m_vao;
    QOpenGLBuffer m_vbo{QOpenGLBuffer::VertexBuffer};
    QOpenGLShaderProgram m_shader;
    QVector3D m_center;
    float m_distance = 10;
    float m_yaw = 35;
    float m_pitch = 20;
    int m_pointSize = 2;
    int m_brushSize = 14;
    Tool m_tool = Browse;
    ColorMode m_colorMode = Semantic;
    bool m_upload = true;
    bool m_through = false;
    bool m_brushing = false;
    bool m_panning = false;
    bool m_orbiting = false;
    QPointF m_last;
    QPolygonF m_polygon;
    QSet<int> m_stroke;
    QVector<int> m_visibleIndices;
    QSet<quint16> m_hiddenSemantics;
    QSet<quint16> m_hiddenInstances;
    QHash<quint16, QColor> m_semanticColors;
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
    m_view = new PointCloudView(center);
    layout->addWidget(m_view, 1);
    splitter->addWidget(center);
    auto* right = new QWidget(splitter);
    auto* rightLayout = new QVBoxLayout(right);
    rightLayout->addWidget(new QLabel(tr("Semantic class"), right));
    m_class = new QComboBox(right);
    m_class->setEditable(true);
    refreshClassCombo();
    rightLayout->addWidget(m_class);
    rightLayout->addWidget(new QLabel(tr("Operation"), right));
    m_operation = new QComboBox(right);
    m_operation->addItem(tr("Assign semantic"),
                         int(PointCloudDocument::AssignSemantic));
    m_operation->addItem(tr("Create instance"),
                         int(PointCloudDocument::CreateInstance));
    m_operation->addItem(tr("Add to instance"),
                         int(PointCloudDocument::AddToInstance));
    m_operation->addItem(tr("Remove from instance"),
                         int(PointCloudDocument::RemoveFromInstance));
    m_operation->addItem(tr("Split instance"),
                         int(PointCloudDocument::SplitInstance));
    m_operation->addItem(tr("Create cuboid from selection"),
                         int(PointCloudDocument::CreateCuboid));
    rightLayout->addWidget(m_operation);
    rightLayout->addWidget(new QLabel(tr("Instance ID"), right));
    m_instance = new QSpinBox(right);
    m_instance->setRange(0, 65535);
    rightLayout->addWidget(m_instance);
    rightLayout->addWidget(new QLabel(tr("Class visibility"), right));
    m_classList = new QListWidget(right);
    m_classList->setSelectionMode(QAbstractItemView::SingleSelection);
    rightLayout->addWidget(m_classList, 1);
    rightLayout->addWidget(new QLabel(tr("Instance visibility"), right));
    m_instanceList = new QListWidget(right);
    m_instanceList->setSelectionMode(QAbstractItemView::NoSelection);
    rightLayout->addWidget(m_instanceList, 1);
    rightLayout->addStretch();
    splitter->addWidget(right);
    splitter->setStretchFactor(1, 1);
    setCentralWidget(splitter);
    m_status = new QLabel(this);
    statusBar()->addWidget(m_status);

    // Camera dock: calibrated image sources with the current frame's point
    // cloud projected on top (upstream pointcloud camera sources parity).
    auto* cameraDock = new QDockWidget(tr("Camera"), this);
    cameraDock->setObjectName(QStringLiteral("PointCloudCameraDock"));
    auto* cameraBody = new QWidget(cameraDock);
    auto* cameraLayout = new QVBoxLayout(cameraBody);
    cameraLayout->setContentsMargins(4, 4, 4, 4);
    auto* addCameraButton = new QPushButton(tr("Add camera..."), cameraBody);
    connect(addCameraButton, &QPushButton::clicked, this,
            &PointCloudWindow::addCamera);
    cameraLayout->addWidget(addCameraButton);
    m_cameraList = new QListWidget(cameraBody);
    m_cameraList->setObjectName(QStringLiteral("PointCloudCameras"));
    connect(m_cameraList, &QListWidget::currentRowChanged, this,
            &PointCloudWindow::showCamera);
    cameraLayout->addWidget(m_cameraList, 1);
    m_cameraImage = new QLabel(cameraBody);
    m_cameraImage->setObjectName(QStringLiteral("PointCloudCameraImage"));
    m_cameraImage->setAlignment(Qt::AlignCenter);
    m_cameraImage->setMinimumSize(320, 240);
    cameraLayout->addWidget(m_cameraImage, 3);
    addDockWidget(Qt::RightDockWidgetArea, cameraDock);
    cameraDock->hide();
    toolbar->addAction(cameraDock->toggleViewAction());
    restoreCameras();
    connect(m_fileList, &QListWidget::currentRowChanged, this,
            &PointCloudWindow::selectFrame);
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
        action->setCheckable(true);
        group->addAction(action);
        if (item.second == PointCloudView::Browse) action->setChecked(true);
        connect(action, &QAction::triggered, this,
                [this, item]() { m_view->setTool(item.second); });
    }
    tools->addAction(tr("Finish polygon"), m_view,
                     &PointCloudView::finishPolygon);
    tools->addAction(tr("Cancel polygon"), m_view,
                     &PointCloudView::cancelPolygon);
    tools->addAction(tr("Fit"), m_view, &PointCloudView::fit);
    for (const auto& item :
         {std::pair<const char*, int>{"Top", 0}, {"Front", 1}, {"Side", 2}})
        tools->addAction(tr(item.first), m_view,
                         [this, item]() { m_view->setView(item.second); });
    QAction* through = tools->addAction(tr("Through selection"));
    through->setCheckable(true);
    connect(through, &QAction::toggled, m_view, &PointCloudView::setThrough);
    auto* render = new QComboBox(tools);
    render->addItems(
            {tr("Semantic"), tr("Intensity"), tr("RGB"), tr("Instance")});
    connect(render, QOverload<int>::of(&QComboBox::currentIndexChanged), m_view,
            [this](int index) {
                m_view->setColorMode(PointCloudView::ColorMode(index));
            });
    tools->addWidget(render);
    auto* pointSize = new QSpinBox(tools);
    pointSize->setRange(1, 10);
    pointSize->setValue(2);
    pointSize->setToolTip(tr("Point size (px)"));
    connect(pointSize, QOverload<int>::of(&QSpinBox::valueChanged), m_view,
            &PointCloudView::setPointSize);
    tools->addWidget(pointSize);
    auto* brushSize = new QSpinBox(tools);
    brushSize->setRange(1, 100);
    brushSize->setValue(14);
    brushSize->setToolTip(tr("Brush size (px)"));
    connect(brushSize, QOverload<int>::of(&QSpinBox::valueChanged), m_view,
            &PointCloudView::setBrushSize);
    tools->addWidget(brushSize);
}

void PointCloudWindow::openFile(const QString& path) {
    if (!confirmSaved()) return;
    const QFileInfo info(path);
    QDir directory = info.dir();
    if (directory.dirName() != QStringLiteral("velodyne") &&
        directory.exists(QStringLiteral("velodyne")))
        directory.cd(QStringLiteral("velodyne"));
    const QStringList files = directory.entryList(
            {QStringLiteral("*.bin"), QStringLiteral("*.ply")}, QDir::Files,
            QDir::Name);
    m_changingFile = true;
    m_files.clear();
    m_fileList->clear();
    for (const QString& name : files) {
        m_files.append(directory.filePath(name));
        m_fileList->addItem(name);
    }
    m_changingFile = false;
    const QString targetPath = info.canonicalFilePath().isEmpty()
                                       ? info.absoluteFilePath()
                                       : info.canonicalFilePath();
    int index = -1;
    for (int row = 0; row < m_files.size(); ++row) {
        const QFileInfo candidate(m_files[row]);
        const QString candidatePath = candidate.canonicalFilePath().isEmpty()
                                              ? candidate.absoluteFilePath()
                                              : candidate.canonicalFilePath();
        if (candidatePath == targetPath) {
            index = row;
            break;
        }
    }
    if (index >= 0) m_fileList->setCurrentRow(index);
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
    if (!m_document.frame().path.isEmpty() && !confirmSaved()) {
        m_changingFile = true;
        m_fileList->setCurrentRow(m_files.indexOf(m_document.frame().path));
        m_changingFile = false;
        return;
    }
    PointCloudFrame frame;
    QString error;
    const QString output =
            m_outputDirectory.isEmpty()
                    ? QString()
                    : QDir(m_outputDirectory)
                              .filePath(QFileInfo(m_files[index])
                                                .completeBaseName() +
                                        QStringLiteral(".label"));
    // Streaming prefetch hit: switching to a neighboring frame skips the
    // file decode entirely.
    const auto cached = m_prefetch.constFind(m_files[index]);
    if (cached != m_prefetch.constEnd()) {
        frame = cached.value();
        m_prefetch.erase(cached);
        if (!output.isEmpty()) frame.labelPath = output;
    } else if (!loadPointCloudFrame(m_files[index], &frame, &error, output)) {
        QMessageBox::warning(this, windowTitle(), error);
        return;
    }
    m_document.setFrame(std::move(frame));
    m_view->setFrame(&m_document.frame());
    refresh();
    refreshCameras();
    schedulePrefetch(index);
}

void PointCloudWindow::schedulePrefetch(int index) {
    for (int offset : {-1, 1, -2, 2}) {
        const int target = index + offset;
        if (target < 0 || target >= m_files.size()) continue;
        const QString path = m_files[target];
        if (path == m_document.frame().path || m_prefetch.contains(path) ||
            m_prefetching.contains(path))
            continue;
        const QString output =
                m_outputDirectory.isEmpty()
                        ? QString()
                        : QDir(m_outputDirectory)
                                  .filePath(QFileInfo(path).completeBaseName() +
                                            QStringLiteral(".label"));
        m_prefetching.insert(path);
        auto* watcher =
                new QFutureWatcher<QPair<PointCloudFrame, QString>>(this);
        connect(watcher,
                &QFutureWatcher<QPair<PointCloudFrame, QString>>::finished,
                this, [this, watcher, path]() {
                    watcher->deleteLater();
                    m_prefetching.remove(path);
                    // Error strings double as the failure flag; only
                    // successfully decoded frames enter the cache.
                    if (!watcher->result().second.isEmpty()) return;
                    // Bounded cache: evict an arbitrary older entry past
                    // four decoded frames (~ a few hundred MB worst case).
                    while (m_prefetch.size() >= 4)
                        m_prefetch.erase(m_prefetch.begin());
                    m_prefetch.insert(path, watcher->result().first);
                });
        watcher->setFuture(QtConcurrent::run([path, output]() {
            PointCloudFrame frame;
            QString error;
            loadPointCloudFrame(path, &frame, &error, output);
            return qMakePair(frame, error);
        }));
    }
}

void PointCloudWindow::save() {
    if (m_document.frame().path.isEmpty()) return;
    QString error;
    if (!savePointCloudLabels(m_document.frame(), &error))
        QMessageBox::warning(this, windowTitle(), error);
    else {
        m_document.markSaved();
        refresh();
    }
}

void PointCloudWindow::saveAs() {
    if (m_document.frame().path.isEmpty()) return;
    const QString path = QFileDialog::getSaveFileName(
            this, tr("Save point labels"), m_document.frame().labelPath,
            tr("Point labels (*.label)"));
    if (path.isEmpty()) return;
    QString error;
    if (!savePointCloudLabels(m_document.frame(), &error, path))
        QMessageBox::warning(this, windowTitle(), error);
    else {
        m_document.setLabelPath(path);
        m_document.markSaved();
        refresh();
    }
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
        m_document.setCuboids(cuboids);
        refresh();
        refreshCameras();
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
    for (quint32 label : frame.labels) {
        const quint16 semantic = quint16(label & 0xffffu);
        semantics.insert(semantic);
        ++semanticCounts[semantic];
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
            auto* item =
                    new QListWidgetItem(tr("%1 (%2) - %3 points")
                                                .arg(name)
                                                .arg(id)
                                                .arg(semanticCounts.value(id)),
                                        m_classList);
            item->setData(Qt::UserRole, int(id));
            item->setBackground(m_classColors.value(id, idColor(id)));
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
    m_view->updateColors();
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
}

void PointCloudWindow::refreshClassCombo(quint16 selected) {
    const QSignalBlocker blocker(m_class);
    m_class->clear();
    for (auto it = m_classNames.cbegin(); it != m_classNames.cend(); ++it)
        m_class->addItem(tr("%1  (%2)").arg(it.value()).arg(it.key()),
                         it.key());
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
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) {
        QMessageBox::warning(this, windowTitle(), file.errorString());
        return;
    }
    QJsonParseError parseError;
    const QJsonDocument document =
            QJsonDocument::fromJson(file.readAll(), &parseError);
    if (parseError.error != QJsonParseError::NoError || !document.isObject()) {
        QMessageBox::warning(this, windowTitle(),
                             tr("Invalid class configuration JSON"));
        return;
    }
    const QJsonObject root = document.object();
    if (root.value(QStringLiteral("version")).toInt(-1) != 1 ||
        !root.value(QStringLiteral("classes")).isArray()) {
        QMessageBox::warning(this, windowTitle(),
                             tr("Class configuration requires version 1"));
        return;
    }
    QMap<quint16, QString> names;
    QMap<quint16, QColor> colors;
    for (const QJsonValue& value :
         root.value(QStringLiteral("classes")).toArray()) {
        const QJsonObject item = value.toObject();
        bool idOk = false;
        const int id = item.value(QStringLiteral("id")).toInt(-1);
        const QString name =
                item.value(QStringLiteral("name")).toString().trimmed();
        const QColor color(item.value(QStringLiteral("color")).toString());
        idOk = id >= 0 && id <= 65535;
        if (!idOk || name.isEmpty() || !color.isValid() ||
            names.contains(quint16(id))) {
            QMessageBox::warning(this, windowTitle(),
                                 tr("Invalid or duplicate class entry"));
            return;
        }
        names.insert(quint16(id), name);
        colors.insert(quint16(id), color);
    }
    if (!names.contains(0)) {
        QMessageBox::warning(this, windowTitle(),
                             tr("Class configuration needs ID 0"));
        return;
    }
    const int selected = m_class->currentData().toInt();
    m_classNames = std::move(names);
    m_classColors = std::move(colors);
    m_savedClassNames = m_classNames;
    m_savedClassColors = m_classColors;
    m_classConfigPath = QFileInfo(path).absoluteFilePath();
    refreshClassCombo(quint16(selected));
    refresh();
}

bool PointCloudWindow::classConfigDirty() const {
    return m_classNames != m_savedClassNames ||
           m_classColors != m_savedClassColors;
}

bool PointCloudWindow::writeClassDefinitions(const QString& path,
                                             QString* error) const {
    if (error) error->clear();
    if (path.isEmpty() ||
        QFileInfo(path).absoluteFilePath() ==
                QFileInfo(m_document.frame().path).absoluteFilePath() ||
        QFileInfo(path).absoluteFilePath() ==
                QFileInfo(m_document.frame().labelPath).absoluteFilePath()) {
        if (error)
            *error = tr("Class definitions must use a separate JSON file.");
        return false;
    }
    QJsonArray entries;
    for (auto it = m_classNames.cbegin(); it != m_classNames.cend(); ++it) {
        const QColor color = m_classColors.value(it.key(), idColor(it.key()));
        entries.append(
                QJsonObject{{QStringLiteral("id"), int(it.key())},
                            {QStringLiteral("name"), it.value()},
                            {QStringLiteral("color"), color.name().toUpper()}});
    }
    QSaveFile output(path);
    const QByteArray data =
            QJsonDocument(QJsonObject{{QStringLiteral("version"), 1},
                                      {QStringLiteral("classes"), entries}})
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
    if (!writeClassDefinitions(path, &error)) {
        QMessageBox::warning(this, windowTitle(), error);
        return;
    }
    m_classConfigPath = QFileInfo(path).absoluteFilePath();
    m_savedClassNames = m_classNames;
    m_savedClassColors = m_classColors;
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
    id->setRange(0, 65535);
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
    for (int index = 0; index < labels.size(); ++index)
        if ((labels[index] & 0xffffu) == classId) indices.append(index);
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
    if (confirmSaved())
        event->accept();
    else
        event->ignore();
}

void PointCloudWindow::addCamera() {
    QDialog dialog(this);
    dialog.setObjectName(QStringLiteral("PointCloudCameraDialog"));
    dialog.setWindowTitle(tr("Add camera"));
    auto* form = new QFormLayout(&dialog);
    auto* directory = new QLineEdit(&dialog);
    auto* browseDirectory = new QPushButton(tr("Browse..."), &dialog);
    connect(browseDirectory, &QPushButton::clicked, directory,
            [this, directory]() {
                const QString path = QFileDialog::getExistingDirectory(
                        this, tr("Camera image directory"));
                if (!path.isEmpty()) directory->setText(path);
            });
    auto* directoryRow = new QHBoxLayout();
    directoryRow->addWidget(directory, 1);
    directoryRow->addWidget(browseDirectory);
    form->addRow(tr("Image directory"), directoryRow);
    auto* calibrationPath = new QLineEdit(&dialog);
    auto* browseCalibration = new QPushButton(tr("Browse..."), &dialog);
    connect(browseCalibration, &QPushButton::clicked, calibrationPath,
            [this, calibrationPath]() {
                const QString path = QFileDialog::getOpenFileName(
                        this, tr("Calibration sidecar"), QString(),
                        tr("Calibration JSON (*.json)"));
                if (!path.isEmpty()) calibrationPath->setText(path);
            });
    auto* calibrationRow = new QHBoxLayout();
    calibrationRow->addWidget(calibrationPath, 1);
    calibrationRow->addWidget(browseCalibration);
    form->addRow(tr("Calibration sidecar"), calibrationRow);
    auto* buttons = new QDialogButtonBox(
            QDialogButtonBox::Ok | QDialogButtonBox::Cancel, &dialog);
    connect(buttons, &QDialogButtonBox::accepted, &dialog, &QDialog::accept);
    connect(buttons, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);
    form->addRow(buttons);
    if (dialog.exec() != QDialog::Accepted) return;

    PointCloudCalibration calibration;
    QString error;
    if (!loadPointCloudCalibration(calibrationPath->text().trimmed(),
                                   &calibration, &error)) {
        QMessageBox::warning(this, windowTitle(), error);
        return;
    }
    CameraSource source;
    source.directory = QDir(directory->text().trimmed()).absolutePath();
    source.calibrationPath = calibrationPath->text().trimmed();
    source.calibration = calibration;
    source.name = QFileInfo(source.calibrationPath).completeBaseName();
    const QStringList suffixes = {"jpg", "jpeg", "png", "bmp"};
    for (const QFileInfo& entry :
         QDir(source.directory)
                 .entryInfoList(suffixes, QDir::Files, QDir::Name))
        source.images.append(entry.absoluteFilePath());
    if (source.images.isEmpty()) {
        QMessageBox::warning(this, windowTitle(),
                             tr("The camera directory holds no images."));
        return;
    }
    m_cameras.append(source);
    persistCameras();
    m_cameraList->clear();
    for (const CameraSource& camera : m_cameras)
        m_cameraList->addItem(camera.name);
    m_cameraList->setCurrentRow(m_cameras.size() - 1);
}

QString PointCloudWindow::findCameraImage(const CameraSource& source,
                                          const QString& framePath) const {
    // Upstream camera_name parity: the camera image shares the frame's
    // complete base name.
    const QString stem = QFileInfo(framePath).completeBaseName();
    for (const QString& image : source.images)
        if (QFileInfo(image).completeBaseName() == stem) return image;
    return QString();
}

void PointCloudWindow::showCamera(int index) {
    if (index < 0 || index >= m_cameras.size()) {
        m_cameraImage->clear();
        return;
    }
    refreshCameras();
}

void PointCloudWindow::refreshCameras() {
    const int row = m_cameraList ? m_cameraList->currentRow() : -1;
    if (row < 0 || row >= m_cameras.size() ||
        m_document.frame().path.isEmpty()) {
        m_cameraImage->clear();
        return;
    }
    const CameraSource& source = m_cameras.at(row);
    const QString image = findCameraImage(source, m_document.frame().path);
    if (image.isEmpty()) {
        m_cameraImage->setText(
                tr("No image for frame %1")
                        .arg(QFileInfo(m_document.frame().path).fileName()));
        return;
    }
    QImageReader reader(image);
    reader.setAutoTransform(true);
    const QImage frame = reader.read();
    if (frame.isNull()) {
        m_cameraImage->setText(reader.errorString());
        return;
    }
    // The point colors follow the semantic class palette used by the GL
    // view; the overlay draws far-to-near so near points win. The frame
    // packs XYZI as QVector4D; the projection kernel takes XYZ only.
    QVector<QVector3D> points;
    points.reserve(m_document.frame().points.size());
    for (const QVector4D& point : m_document.frame().points)
        points.append(point.toVector3D());
    QHash<quint16, QColor> colors;
    for (auto it = m_classColors.constBegin(); it != m_classColors.constEnd();
         ++it)
        colors.insert(it.key(), it.value());
    const QImage overlay =
            renderPointCloudCameraOverlay(frame, source.calibration, points,
                                          m_document.frame().labels, colors, 2);
    // Cuboid wireframes project on top of the point overlay (upstream
    // project_cuboid consumers draw them as class-colored polylines).
    QImage composed = overlay.convertToFormat(QImage::Format_ARGB32);
    QPainter painter(&composed);
    painter.setRenderHint(QPainter::Antialiasing, true);
    for (const PointCloudCuboid& box : m_document.frame().cuboids) {
        const QColor color =
                colors.value(quint16(box.classId), QColor(255, 200, 0));
        QPen pen(color, 2);
        painter.setPen(pen);
        QVector<QPair<QVector2D, QVector2D>> segments;
        projectPointCloudCuboid(box.corners(), source.calibration, &segments);
        for (const auto& segment : segments) {
            painter.drawLine(QPointF(segment.first.x(), segment.first.y()),
                             QPointF(segment.second.x(), segment.second.y()));
        }
    }
    painter.end();
    m_cameraImage->setPixmap(QPixmap::fromImage(composed).scaled(
            m_cameraImage->size(), Qt::KeepAspectRatio,
            Qt::SmoothTransformation));
}

void PointCloudWindow::persistCameras() const {
    QJsonArray entries;
    for (const CameraSource& source : m_cameras) {
        QJsonObject entry;
        entry.insert(QStringLiteral("name"), source.name);
        entry.insert(QStringLiteral("directory"), source.directory);
        entry.insert(QStringLiteral("calibration"), source.calibrationPath);
        entries.append(entry);
    }
    QSettings settings;
    settings.setValue(QStringLiteral("qAnyLabeling/pointCloudCameras"),
                      QString::fromUtf8(QJsonDocument(entries).toJson(
                              QJsonDocument::Compact)));
}

void PointCloudWindow::restoreCameras() {
    QSettings settings;
    const QString raw =
            settings.value(QStringLiteral("qAnyLabeling/pointCloudCameras"))
                    .toString();
    if (raw.isEmpty()) return;
    const QJsonDocument document = QJsonDocument::fromJson(raw.toUtf8());
    if (!document.isArray()) return;
    for (const QJsonValue& value : document.array()) {
        const QJsonObject entry = value.toObject();
        CameraSource source;
        source.name = entry.value(QStringLiteral("name")).toString();
        source.directory = entry.value(QStringLiteral("directory")).toString();
        source.calibrationPath =
                entry.value(QStringLiteral("calibration")).toString();
        QString error;
        if (source.directory.isEmpty() || source.calibrationPath.isEmpty() ||
            !loadPointCloudCalibration(source.calibrationPath,
                                       &source.calibration, &error))
            continue;
        const QStringList suffixes = {"jpg", "jpeg", "png", "bmp"};
        for (const QFileInfo& fileInfo :
             QDir(source.directory)
                     .entryInfoList(suffixes, QDir::Files, QDir::Name))
            source.images.append(fileInfo.absoluteFilePath());
        if (source.images.isEmpty()) continue;
        m_cameras.append(source);
        m_cameraList->addItem(source.name);
    }
    if (!m_cameras.isEmpty()) m_cameraList->setCurrentRow(0);
}
