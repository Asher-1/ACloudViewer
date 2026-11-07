// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "RemoteServerDialog.h"

#include <aicore/inference_log.h>

#include <QComboBox>
#include <QDialogButtonBox>
#include <QFormLayout>
#include <QHBoxLayout>
#include <QJsonArray>
#include <QJsonObject>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QSpinBox>
#include <QVBoxLayout>

RemoteServerDialog::RemoteServerDialog(QWidget* parent) : QDialog(parent) {
    setObjectName(QStringLiteral("RemoteServerDialog"));
    setWindowTitle(tr("Remote inference service"));
    setModal(true);
    resize(520, 240);
    auto* layout = new QVBoxLayout(this);
    auto* form = new QFormLayout;
    m_url = new QLineEdit(this);
    m_url->setObjectName(QStringLiteral("RemoteServerUrl"));
    m_url->setPlaceholderText(QStringLiteral("http://127.0.0.1:8000"));
    m_apiKey = new QLineEdit(this);
    m_apiKey->setObjectName(QStringLiteral("RemoteServerApiKey"));
    m_apiKey->setEchoMode(QLineEdit::Password);
    m_timeout = new QSpinBox(this);
    m_timeout->setObjectName(QStringLiteral("RemoteServerTimeout"));
    m_timeout->setRange(5, 3600);
    m_timeout->setSuffix(tr(" s"));
    m_timeout->setValue(30);
    form->addRow(tr("Server URL"), m_url);
    form->addRow(tr("API key"), m_apiKey);
    form->addRow(tr("Timeout"), m_timeout);
    layout->addLayout(form);

    auto* modelRow = new QHBoxLayout;
    m_fetchButton = new QPushButton(tr("Fetch models"), this);
    m_fetchButton->setObjectName(QStringLiteral("RemoteServerFetchModels"));
    m_models = new QComboBox(this);
    m_models->setObjectName(QStringLiteral("RemoteServerModels"));
    modelRow->addWidget(m_fetchButton, 0);
    modelRow->addWidget(m_models, 1);
    layout->addLayout(modelRow);
    m_status = new QLabel(this);
    m_status->setWordWrap(true);
    layout->addWidget(m_status);

    auto* buttons = new QDialogButtonBox(
            QDialogButtonBox::Ok | QDialogButtonBox::Cancel, this);
    layout->addWidget(buttons);
    connect(buttons, &QDialogButtonBox::accepted, this, &QDialog::accept);
    connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);
    connect(m_fetchButton, &QPushButton::clicked, this,
            &RemoteServerDialog::fetchModels);
    connect(&m_client, &RemoteServerClient::modelsFetched, this,
            [this](const QJsonObject& data, const QString& error) {
                m_fetchButton->setEnabled(true);
                m_models->clear();
                if (!error.isEmpty()) {
                    m_status->setText(error);
                    return;
                }
                for (auto it = data.constBegin(); it != data.constEnd(); ++it)
                    m_models->addItem(it.key());
                m_status->setText(
                        tr("%1 model(s) available").arg(m_models->count()));
            });
}

void RemoteServerDialog::setSettings(
        const RemoteServerClient::Settings& settings) {
    m_url->setText(settings.serverUrl);
    m_apiKey->setText(settings.apiKey);
    m_timeout->setValue(settings.timeoutSec);
}

RemoteServerClient::Settings RemoteServerDialog::settings() const {
    RemoteServerClient::Settings result;
    result.serverUrl = m_url->text().trimmed();
    result.apiKey = m_apiKey->text();
    result.timeoutSec = m_timeout->value();
    return result;
}

QString RemoteServerDialog::selectedModelId() const {
    return m_models->currentText();
}

void RemoteServerDialog::setSelectedModelId(const QString& modelId) {
    if (modelId.isEmpty()) return;
    const int index = m_models->findText(modelId);
    if (index >= 0) m_models->setCurrentIndex(index);
}

void RemoteServerDialog::fetchModels() {
    m_client.applySettings(settings());
    m_fetchButton->setEnabled(false);
    m_status->setText(tr("Fetching model list..."));
    m_client.fetchModels();
}
