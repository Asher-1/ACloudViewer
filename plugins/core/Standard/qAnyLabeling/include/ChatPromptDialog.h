// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "ChatServiceClient.h"

class QWidget;

// Shared remote-service configuration for VQA and arbitrary-category VLM
// classification. Local inference still goes through public AICore APIs.
struct AnnotationChatRequest {
    ChatServiceClient::Settings settings;
    QString prompt;
    QString systemPrompt;
    QJsonArray messages(const QString& imagePath, QString* error) const;
};

bool configureAnnotationChatRequest(QWidget* parent,
                                    const QString& title,
                                    const QString& prompt,
                                    AnnotationChatRequest* request);
