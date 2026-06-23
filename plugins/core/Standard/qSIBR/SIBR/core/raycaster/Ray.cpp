// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "Ray.hpp"

#include <algorithm>

namespace sibr {
/*static*/ SIBR_RAYCASTER_EXPORT const float RayHit::InfinityDist =
        std::numeric_limits<float>::infinity();

Ray::Ray(const sibr::Vector3f& orig, const sibr::Vector3f& dir)
    : _orig(orig), _dir(dir) {
    if (_dir[0] != 0.f || _dir[1] != 0.f || _dir[2] != 0.f) _dir.normalize();
}

RayHit::RayHit(const Ray& r,
               float dist,
               const BCCoord& coord,
               const sibr::Vector3f& normal,
               const Primitive& prim)
    : _ray(r), _dist(dist), _coord(coord), _normal(normal), _prim(prim) {
    _dist = std::max(dist, 0.f);

    // normalize '_normal'
    float len = length(_normal);
    if (len > 1e-10) _normal = _normal / len;
}

sibr::Vector3f RayHit::interpolateUV(void) const {
    float ucoord = barycentricCoord().u;
    float vcoord = barycentricCoord().v;
    return sibr::Vector3f(std::max((1.f - ucoord - vcoord), 0.f), ucoord,
                          vcoord);
}

}  // namespace sibr
