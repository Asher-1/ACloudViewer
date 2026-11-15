// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "JsonRPCCommon.h"

#include <ecvHObjectCaster.h>
#include <ecvMesh.h>
#include <ecvPointCloud.h>

#include <QJsonArray>

namespace jsonrpc {

QJsonObject entityToJson(ccHObject* obj, bool recursive) {
    QJsonObject o;
    if (!obj) return o;
    o["id"] = static_cast<qint64>(obj->getUniqueID());
    o["name"] = obj->getName();
    o["type"] = obj->getClassID() == CV_TYPES::POINT_CLOUD ? "POINT_CLOUD"
                : obj->getClassID() == CV_TYPES::MESH      ? "MESH"
                : obj->getClassID() == CV_TYPES::POLY_LINE ? "POLYLINE"
                : obj->getClassID() == CV_TYPES::HIERARCHY_OBJECT ? "GROUP"
                                                                  : "OTHER";
    o["visible"] = obj->isEnabled();
    o["children_count"] = static_cast<int>(obj->getChildrenNumber());

    ccBBox bb = obj->getOwnBB();
    if (bb.isValid()) {
        QJsonObject bbox;
        bbox["min_x"] = bb.minCorner().x;
        bbox["min_y"] = bb.minCorner().y;
        bbox["min_z"] = bb.minCorner().z;
        bbox["max_x"] = bb.maxCorner().x;
        bbox["max_y"] = bb.maxCorner().y;
        bbox["max_z"] = bb.maxCorner().z;
        o["bbox"] = bbox;
    }

    auto* cloud = ccHObjectCaster::ToPointCloud(obj);
    if (cloud) {
        o["point_count"] = static_cast<qint64>(cloud->size());
        o["has_normals"] = cloud->hasNormals();
        o["has_colors"] = cloud->hasColors();
        o["scalar_field_count"] =
                static_cast<int>(cloud->getNumberOfScalarFields());
    }

    auto* mesh = ccHObjectCaster::ToMesh(obj);
    if (mesh) {
        o["triangle_count"] = static_cast<qint64>(mesh->size());
        auto* verts = mesh->getAssociatedCloud();
        if (verts) o["vertex_count"] = static_cast<qint64>(verts->size());
    }

    if (recursive && obj->getChildrenNumber() > 0) {
        QJsonArray children;
        for (unsigned i = 0; i < obj->getChildrenNumber(); ++i) {
            children.append(entityToJson(obj->getChild(i), true));
        }
        o["children"] = children;
    }
    return o;
}

QJsonObject D() { return QJsonObject(); }
QJsonObject D(const QString& k, const QJsonValue& v) {
    QJsonObject o;
    o[k] = v;
    return o;
}
QJsonObject D(const QString& k1,
              const QJsonValue& v1,
              const QString& k2,
              const QJsonValue& v2) {
    QJsonObject o;
    o[k1] = v1;
    o[k2] = v2;
    return o;
}
QJsonObject D(const QString& k1,
              const QJsonValue& v1,
              const QString& k2,
              const QJsonValue& v2,
              const QString& k3,
              const QJsonValue& v3) {
    QJsonObject o;
    o[k1] = v1;
    o[k2] = v2;
    o[k3] = v3;
    return o;
}
QJsonObject D(const QString& k1,
              const QJsonValue& v1,
              const QString& k2,
              const QJsonValue& v2,
              const QString& k3,
              const QJsonValue& v3,
              const QString& k4,
              const QJsonValue& v4) {
    QJsonObject o;
    o[k1] = v1;
    o[k2] = v2;
    o[k3] = v3;
    o[k4] = v4;
    return o;
}
QJsonObject D(const QString& k1,
              const QJsonValue& v1,
              const QString& k2,
              const QJsonValue& v2,
              const QString& k3,
              const QJsonValue& v3,
              const QString& k4,
              const QJsonValue& v4,
              const QString& k5,
              const QJsonValue& v5) {
    QJsonObject o;
    o[k1] = v1;
    o[k2] = v2;
    o[k3] = v3;
    o[k4] = v4;
    o[k5] = v5;
    return o;
}

}  // namespace jsonrpc
