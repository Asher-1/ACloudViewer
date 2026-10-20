// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QColor>
#include <QFuture>
#include <QMainWindow>
#include <QMap>
#include <QPointer>
#include <QSet>
#include <atomic>

#include "PointCloudCamera.h"
#include "PointCloudDatasetImporter.h"
#include "PointCloudDocument.h"

class QListWidget;
class QComboBox;
class QSpinBox;
class QLabel;
class QListWidgetItem;
class PointCloudView;
class PointCloudCuboidView;
class PointCloudCameraPanel;
class QVBoxLayout;
class QProgressDialog;

class PointCloudWindow final : public QMainWindow {
    Q_OBJECT

public:
    explicit PointCloudWindow(QWidget* parent = nullptr);
    ~PointCloudWindow() override;
    void openFile(const QString& path);
    bool loadClassDefinitions(const QString& path, QString* error = nullptr);
    bool saveClassDefinitions(const QString& path, QString* error = nullptr);
    bool exportDataset(const QString& path,
                       const QString& format,
                       bool includePointCloud = true,
                       const std::function<bool(int, int)>& progress = {},
                       QString* error = nullptr) const;
    bool prepareDatasetImport(
            const QString& path,
            const QString& format,
            const QString& configPath,
            PointCloudDatasetImportPlan* plan,
            const std::function<bool(int, int)>& progress = {},
            QString* error = nullptr) const;
    bool applyDatasetImport(const PointCloudDatasetImportPlan& plan,
                            const std::function<bool(int, int)>& progress = {},
                            QString* error = nullptr);

protected:
    void closeEvent(QCloseEvent* event) override;

private:
    void openDirectory();
    void selectFrame(int index);
    void schedulePrefetch(int index);
    bool loadFrame(const QStringList& files, int index);
    void save();
    void saveAs();
    bool confirmSaved();
    void applySelection(const QVector<int>& indices);
    void refresh();
    void updateVisibility();
    void loadClasses();
    void saveClasses();
    void addClass();
    void editClass();
    void removeClass();
    void editClassDefinition(bool editing);
    void refreshClassCombo(quint16 selected = 0);
    bool classConfigDirty() const;
    bool writeClassDefinitions(const QString& path, QString* error) const;
    void setupCuboids(QWidget* center,
                      QVBoxLayout* layout,
                      QVBoxLayout* rightLayout);
    void configureCamera();
    void updateCamera();
    void refreshCuboids();
    void setCuboidDepthCropEnabled(bool enabled);
    void setPointCloudPointSize(int size);
    void toggleExpandedCuboidView(PointCloudCuboidView* view);
    void commitCuboid(const PointCloudCuboid& box);
    void previewCuboid(const PointCloudCuboid* box,
                       PointCloudCuboidView* source = nullptr);
    void editCuboidProperties();
    const PointCloudCuboid* selectedCuboid() const;
    bool saveAnnotations(const QString& labelPath);
    void datasetDialog(bool importing);
    void switchTaskClasses(bool detection);
    bool datasetFrames(QVector<PointCloudFrame>* frames,
                       QString* error,
                       const std::function<bool(int, int)>& progress) const;
    QByteArray datasetSnapshot() const;

    QHash<QString, PointCloudFrame> m_prefetch;
    QSet<QString> m_prefetching;
    QVector<QFuture<QPair<PointCloudFrame, QString>>> m_prefetchJobs;
    std::atomic<bool> m_prefetchCancelled{false};
    PointCloudDocument m_document;
    QStringList m_files;
    QListWidget* m_fileList = nullptr;
    QListWidget* m_classList = nullptr;
    QListWidget* m_instanceList = nullptr;
    PointCloudView* m_view = nullptr;
    QComboBox* m_class = nullptr;
    QComboBox* m_operation = nullptr;
    QSpinBox* m_instance = nullptr;
    QLabel* m_status = nullptr;
    QString m_outputDirectory;
    bool m_changingFile = false;
    bool m_loading = false;
    QProgressDialog* m_loadProgress = nullptr;
    QSet<quint16> m_hiddenSemantics;
    QSet<quint16> m_hiddenInstances;
    QMap<quint16, QString> m_classNames{{0, QStringLiteral("Unlabeled")}};
    QMap<quint16, QColor> m_classColors{{0, QColor(QStringLiteral("#808080"))}};
    QMap<quint16, QString> m_savedClassNames{{0, QStringLiteral("Unlabeled")}};
    QMap<quint16, QColor> m_savedClassColors{
            {0, QColor(QStringLiteral("#808080"))}};
    QString m_classConfigPath;
    bool m_detectionClassesActive = false;
    QMap<quint16, QString> m_otherClassNames;
    QMap<quint16, QColor> m_otherClassColors;
    QMap<quint16, QString> m_otherSavedClassNames;
    QMap<quint16, QColor> m_otherSavedClassColors;
    QSet<quint16> m_otherHiddenSemantics;
    QComboBox* m_task = nullptr;
    QComboBox* m_colorMode = nullptr;
    int m_taskColorModes[2] = {0, 2};
    int m_pointSize = 2;
    QListWidget* m_cuboidList = nullptr;
    QWidget* m_cuboidPanel = nullptr;
    QWidget* m_cuboidViews = nullptr;
    QVector<PointCloudCuboidView*> m_orthographic;
    QList<int> m_savedCuboidViewSizes;
    PointCloudCuboidView* m_expandedCuboidView = nullptr;
    QAction* m_cropDepthAction = nullptr;
    QVector<QAction*> m_cuboidActions;
    QVector<QAction*> m_segmentationActions;
    QVector<QWidget*> m_segmentationControls;
    QLabel* m_classHeading = nullptr;
    QSpinBox* m_brushSizeControl = nullptr;
    int m_selectedCuboid = 0;
    PointCloudCuboid m_cuboidClipboard;
    bool m_hasCuboidClipboard = false;
    PointCloudCameraPanel* m_cameraPanel = nullptr;
    QAction* m_cameraToggle = nullptr;
};
