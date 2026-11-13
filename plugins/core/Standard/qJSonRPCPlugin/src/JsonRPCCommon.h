// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// JSON helpers shared by the JsonRPCMethods*.cpp handlers.

#include <QJsonObject>
#include <QJsonValue>
#include <QString>

class ccHObject;

namespace jsonrpc {

/** Id, name, type, visibility, bbox and cloud/mesh sizes of \p obj;
 *  children too when \p recursive. Empty object for nullptr. */
QJsonObject entityToJson(ccHObject* obj, bool recursive = false);

/** Error `data` objects: D("key", value, ...) with up to five pairs. */
QJsonObject D();
QJsonObject D(const QString& k, const QJsonValue& v);
QJsonObject D(const QString& k1,
              const QJsonValue& v1,
              const QString& k2,
              const QJsonValue& v2);
QJsonObject D(const QString& k1,
              const QJsonValue& v1,
              const QString& k2,
              const QJsonValue& v2,
              const QString& k3,
              const QJsonValue& v3);
QJsonObject D(const QString& k1,
              const QJsonValue& v1,
              const QString& k2,
              const QJsonValue& v2,
              const QString& k3,
              const QJsonValue& v3,
              const QString& k4,
              const QJsonValue& v4);
QJsonObject D(const QString& k1,
              const QJsonValue& v1,
              const QString& k2,
              const QJsonValue& v2,
              const QString& k3,
              const QJsonValue& v3,
              const QString& k4,
              const QJsonValue& v4,
              const QString& k5,
              const QJsonValue& v5);

}  // namespace jsonrpc
