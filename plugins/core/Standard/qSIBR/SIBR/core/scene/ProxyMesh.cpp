// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ProxyMesh.hpp"

namespace sibr {

void ProxyMesh::loadFromData(const IParseData::Ptr& data) {
    _proxy.reset(new Mesh());
    // GD HACK
    if (boost::filesystem::extension(data->meshPath()) == ".bin") {
        if (!_proxy->loadSfM(data->meshPath(), data->basePathName())) {
            SIBR_WRG << "proxy model not found at " << data->meshPath()
                     << std::endl;
        }
    } else if (!_proxy->load(data->meshPath(), data->basePathName()) &&
               !_proxy->load(removeExtension(data->meshPath()) + ".ply") &&
               !_proxy->load(removeExtension(data->meshPath()) + ".obj")) {
        if (!_proxy->loadSfM(data->meshPath(), data->basePathName())) {
            SIBR_WRG << "proxy model not found at " << data->meshPath()
                     << std::endl;
        }
    }
    if (!_proxy->hasNormals()) {
        _proxy->generateNormals();
    }
}

void ProxyMesh::replaceProxy(Mesh::Ptr newProxy) {
    _proxy.reset(new Mesh());
    _proxy->vertices(newProxy->vertices());
    _proxy->normals(newProxy->normals());
    _proxy->colors(newProxy->colors());
    _proxy->triangles(newProxy->triangles());
    _proxy->texCoords(newProxy->texCoords());

    // Used by inputImageRT init() and debug rendering
    if (!_proxy->hasNormals()) {
        _proxy->generateNormals();
    }
}

void ProxyMesh::replaceProxyPtr(Mesh::Ptr newProxy) { _proxy = newProxy; }

}  // namespace sibr
