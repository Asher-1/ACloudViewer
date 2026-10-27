// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// Local chat provider ("aicore://llm") over the public aicore/llm_capi.h
// task. Models, URLs and digests come from the AICore llm catalog; this file
// only adapts OpenAI-format chat messages to typed llm requests.
#include <QByteArray>
#include <QJsonArray>
#include <QString>
#include <QUrl>
#include <QVector>
#include <functional>

class QObject;

namespace LocalChatEngine {

/** True for the local provider address (scheme "aicore"). */
bool isLocal(const QUrl& url);

/** Catalog chat models as OpenAI /models entries: {id (catalog filename),
 *  name, vision, downloaded, owned_by}. Projectors are not listed. */
QJsonArray models();

/** A catalog file the model needs that is missing or unverified. */
struct Asset {
    QString path;
    QString url;
    QByteArray sha256;
    qint64 size = 0;
};
/** Model file plus its projector, filtered to the ones still to download.
 *  Returns false (with error) when the id is not a catalog chat model. */
bool missingAssets(const QString& model, QVector<Asset>* missing,
                   QString* error);

struct Callbacks {
    std::function<void(const QString& delta)> delta;  // may be empty
    std::function<void(const QString& text, const QString& error)> done;
};
/** Queues one generation. Requests run FIFO on one worker thread that
 *  owns the llm context (created, reused for the same model and freed
 *  there), so concurrent clients share one loaded model. Callbacks run on
 *  the receiver's thread; none run after cancel(id) returns. */
quint64 submit(const QString& model,
               const QJsonArray& messages,
               double temperature,
               int maxTokens,
               QObject* receiver,
               Callbacks callbacks);
/** Drops a queued request or stops the running one; thread-safe. */
void cancel(quint64 id);

}  // namespace LocalChatEngine
