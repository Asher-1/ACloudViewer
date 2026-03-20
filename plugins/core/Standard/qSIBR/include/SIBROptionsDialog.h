// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QDialog>

class QLineEdit;
class QSpinBox;
class QComboBox;
class QCheckBox;

struct SIBRViewerOptions {
    int width = 1280;
    int height = 720;
    bool forceAspectRatio = false;
    bool loadImages = false;
    bool noInterop = false;
    int device = 0;
    QString datasetPath;
    QString modelPath;
    QString iteration;
    QString remoteIP = "127.0.0.1";
    int remotePort = 6009;
};

class SIBROptionsDialog : public QDialog {
    Q_OBJECT

public:
    enum ViewerType {
        ULR,
        ULRv2,
        TexturedMesh,
        PointBased,
        GaussianSplatting,
        RemoteGaussian
    };

    SIBROptionsDialog(ViewerType type, QWidget* parent = nullptr);

    void setInitialPaths(const QString& datasetPath, const QString& modelPath);

    void setSelectedEntityInfo(const QString& entityName,
                               const QString& entityType);

    SIBRViewerOptions getOptions() const;

private slots:
    void browseDatasetPath();
    void browseModelPath();

private:
    void setupUI();
    void setupGaussianFields();
    void setupRemoteFields();

    ViewerType m_type;
    class QLabel* m_selectionLabel = nullptr;
    QLineEdit* m_datasetPath = nullptr;
    QLineEdit* m_modelPath = nullptr;
    QLineEdit* m_iteration = nullptr;
    QSpinBox* m_width = nullptr;
    QSpinBox* m_height = nullptr;
    QCheckBox* m_forceAspectRatio = nullptr;
    QCheckBox* m_loadImages = nullptr;
    QCheckBox* m_noInterop = nullptr;
    QSpinBox* m_device = nullptr;
    QLineEdit* m_remoteIP = nullptr;
    QSpinBox* m_remotePort = nullptr;
};
