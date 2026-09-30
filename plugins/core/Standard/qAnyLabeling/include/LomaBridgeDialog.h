// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QDialog>
#include <QImage>
#include <QPointF>
#include <QStringList>
#include <QVector>

class QCheckBox;
class QComboBox;
class QLabel;
class QListWidget;
class QPushButton;
class QTableWidget;
class QThread;
class ecvModelDownloader;

struct LomaCorrespondence {
    QPointF image0;
    QPointF image1;
    float score = 0.0f;
};

Q_DECLARE_METATYPE(LomaCorrespondence)
Q_DECLARE_METATYPE(QVector<LomaCorrespondence>)

class LomaBridgeDialog final : public QDialog {
    Q_OBJECT
public:
    explicit LomaBridgeDialog(QWidget* parent = nullptr);
    ~LomaBridgeDialog() override;

    void setImage0(const QString& path);
    void setImage1(const QString& path);
    QString image0Path() const { return m_image0Path; }
    QString image1Path() const { return m_image1Path; }
    QVector<LomaCorrespondence> acceptedMatches() const;

signals:
    void matchingFinished(bool ok, int count);

private slots:
    void browseImage0();
    void browseImage1();
    void runMatching();
    void onFinished(bool ok,
                    const QString& error,
                    const QVector<LomaCorrespondence>& matches);

private:
    void refreshModelLists();
    void updatePreview();
    void setBusy(bool busy);
    void downloadNextModel();

    QString m_image0Path;
    QString m_image1Path;
    QImage m_image0;
    QImage m_image1;
    QComboBox* m_detector = nullptr;
    QComboBox* m_descriptor = nullptr;
    QComboBox* m_matcher = nullptr;
    QComboBox* m_device = nullptr;
    QLabel* m_preview0 = nullptr;
    QLabel* m_preview1 = nullptr;
    QLabel* m_status = nullptr;
    QTableWidget* m_matches = nullptr;
    QPushButton* m_run = nullptr;
    QPushButton* m_accept = nullptr;
    QThread* m_thread = nullptr;
    ecvModelDownloader* m_downloader = nullptr;
    QStringList m_pendingModels;
    QVector<LomaCorrespondence> m_result;
};
