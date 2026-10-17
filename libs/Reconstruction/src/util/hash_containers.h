// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// Hash container aliases for COLMAP (upstream util/hash_containers.h).
//
// Two families of aliases are provided:
//
//   FlatHashMap / FlatHashSet
//       Open-addressing (flat) containers. Fastest and lowest-memory, but they
//       INVALIDATE references, pointers, and iterators to elements on any
//       insertion that triggers a rehash. Use only where no long-lived
//       reference to a stored element is held across a mutation of the same
//       container, and avoid iterator-based erase loops (erase by key instead).
//
//   NodeHashMap / NodeHashSet
//       Node-based containers whose element references/pointers remain valid
//       across insert/erase of *other* elements. Use for element stores that
//       hand out long-lived references.
//
// The default hash is std::hash<K>, so the custom std::hash specializations
// and colmap::PairHash (see util/types.h) are reused unchanged.
// boost::unordered internally re-mixes a non-avalanching hash such as the
// identity std::hash<uint32_t>.
//
// The backend is boost::unordered 1.92 (vendored header-only subset in
// 3rdparty/boost_unordered, put first on ColmapLib's PUBLIC include path), as
// in upstream. It is not selectable: the container's iteration order is
// observable behavior. The SfM gauge-point selection, track completion, and
// filtering iterate these containers, and a std backend was measured to
// diverge the reconstructed model from upstream for identical inputs, while
// upstream compiled with std containers reproduced the fork's divergent model
// (docs in libs/Reconstruction/TEST_ALIGNMENT_TODO.md). The aliases are also
// data members of classes in public headers, so every consumer must see the
// same definition.

#include <boost/unordered/unordered_flat_map.hpp>
#include <boost/unordered/unordered_flat_set.hpp>
#include <boost/unordered/unordered_node_map.hpp>
#include <boost/unordered/unordered_node_set.hpp>
#include <functional>

namespace colmap {

template <class Key,
          class Value,
          class Hash = std::hash<Key>,
          class Eq = std::equal_to<Key>>
using FlatHashMap = boost::unordered_flat_map<Key, Value, Hash, Eq>;

template <class Key, class Hash = std::hash<Key>, class Eq = std::equal_to<Key>>
using FlatHashSet = boost::unordered_flat_set<Key, Hash, Eq>;

template <class Key,
          class Value,
          class Hash = std::hash<Key>,
          class Eq = std::equal_to<Key>>
using NodeHashMap = boost::unordered_node_map<Key, Value, Hash, Eq>;

template <class Key, class Hash = std::hash<Key>, class Eq = std::equal_to<Key>>
using NodeHashSet = boost::unordered_node_set<Key, Hash, Eq>;

}  // namespace colmap
