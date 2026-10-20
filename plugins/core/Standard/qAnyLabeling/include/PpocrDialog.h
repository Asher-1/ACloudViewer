// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once
#include <QDialog>
#include <QJsonObject>
#include <QPointer>
#include <QStringList>

class QComboBox;
class QLabel;
class QLineEdit;
class QListWidget;
class QPlainTextEdit;
class QSpinBox;
class QTextBrowser;
class PpocrPreview;
class PpocrServiceClient;

class PpocrDialog final : public QDialog {
    Q_OBJECT
public:
    explicit PpocrDialog(QWidget* parent = nullptr,
                         const QString& workDirectory = {});
    ~PpocrDialog() override;
    QStringList importFiles(const QStringList& paths);

protected:
    void dragEnterEvent(QDragEnterEvent* event) override;
    void dropEvent(QDropEvent* event) override;
    void closeEvent(QCloseEvent* event) override;

private:
    QString jsonPath(const QString& name) const;
    QJsonObject load(const QString& path) const;
    bool saveCurrent();
    void refresh(const QString& selected = {});
    void selectRecord();
    void render();
    void renderPage();
    void parseSelection();
    void startParsing(const QStringList& names);
    void confirmImport(const QStringList& paths);
    void nextQueued();
    void editBlock();
    void commitBlockEdit(const QString& key, const QString& content);
    void settings();
    void probeService();
    void saveSettings();
    void exportResult(bool json);
    QString m_root, m_name, m_active;
    QStringList m_queue;
    QJsonObject m_document, m_state, m_settings;
    QListWidget *m_records, *m_blocks;
    QLineEdit* m_search;
    QComboBox *m_tab, *m_statusFilter, *m_typeFilter, *m_sort, *m_model;
    QLabel* m_status;
    QSpinBox* m_page;
    PpocrPreview* m_preview;
    QPlainTextEdit* m_json;
    QTextBrowser* m_markdown;
    PpocrServiceClient* m_client;
    QPointer<QWidget> m_editor;
    quint64 m_renderGeneration = 0;
};
