// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// Python conversions for the COLMAP hash container aliases
// (util/hash_containers.h), following upstream pycolmap/pybind11_extension.h
// and pycolmap/scene/types.h. pybind11/stl.h only converts std::unordered_*,
// while the aliases are boost::unordered containers, so without these casters
// any binding that passes one by value would compile and then fail at runtime
// with an unregistered-type error. None of these containers is bound with
// PYBIND11_MAKE_OPAQUE / py::bind_map in this module, so the generic casters
// cannot collide with an opaque binding. Every pybind TU that sees the
// aliases must include this header so that all of them agree on the caster.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "util/hash_containers.h"

namespace pybind11 {
namespace detail {

template <typename Key, typename Value, typename Hash, typename Equal>
struct type_caster<colmap::FlatHashMap<Key, Value, Hash, Equal>>
    : map_caster<colmap::FlatHashMap<Key, Value, Hash, Equal>, Key, Value> {};

template <typename Key, typename Hash, typename Equal>
struct type_caster<colmap::FlatHashSet<Key, Hash, Equal>>
    : set_caster<colmap::FlatHashSet<Key, Hash, Equal>, Key> {};

template <typename Key, typename Value, typename Hash, typename Equal>
struct type_caster<colmap::NodeHashMap<Key, Value, Hash, Equal>>
    : map_caster<colmap::NodeHashMap<Key, Value, Hash, Equal>, Key, Value> {};

template <typename Key, typename Hash, typename Equal>
struct type_caster<colmap::NodeHashSet<Key, Hash, Equal>>
    : set_caster<colmap::NodeHashSet<Key, Hash, Equal>, Key> {};

}  // namespace detail
}  // namespace pybind11
