// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QDialog>

#include "RemoteServerClient.h"

class QComboBox;
class QLabel;
class QLineEdit;
class QPushButton;
class QSpinBox;

/** Connection settings for the remote inference service: server URL, API
 *  key, timeout and the model list fetched live from /v1/models. Upstream
 *  parity: the remote_server_select_combobox widget plus
 *  remote_server_settings persistence. */
class RemoteServerDialog final : public QDialog {
    Q_OBJECT
public:
    explicit RemoteServerDialog(QWidget* parent = nullptr);

    void setSettings(const RemoteServerClient::Settings& settings);
    RemoteServerClient::Settings settings() const;
    QString selectedModelId() const;
    void setSelectedModelId(const QString& modelId);

private:
    void fetchModels();

    RemoteServerClient m_client;
    QLineEdit* m_url = nullptr;
    QLineEdit* m_apiKey = nullptr;
    QSpinBox* m_timeout = nullptr;
    QPushButton* m_fetchButton = nullptr;
    QComboBox* m_models = nullptr;
    QLabel* m_status = nullptr;
};
