// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

// Backend-selectable hash container aliases for COLMAP.
//
// This header centralizes the hash map/set implementation used across the
// performance-critical scene and SfM data structures so that the underlying
// container can be swapped at compile time, without touching call sites. The
// backend is selected by one of the following macros, set by CMake via the
// COLMAP_HASH_MAP_BACKEND cache variable:
//
//   COLMAP_HASH_STD    -> std::unordered_map / std::unordered_set (default)
//   COLMAP_HASH_BOOST  -> boost::unordered_flat_* / boost::unordered_node_*
//
// Two families of aliases are provided:
//
//   FlatHashMap / FlatHashSet
//       Open-addressing (flat) containers. Fastest and lowest-memory, but they
//       INVALIDATE references, pointers, and iterators to elements on any
//       insertion that triggers a rehash (and, for boost, on erase). Use only
//       where no long-lived reference to a stored element is held across a
//       mutation of the same container, and avoid iterator-based erase loops
//       (erase by key instead).
//
//   NodeHashMap / NodeHashSet
//       Node-based containers whose element references/pointers remain valid
//       across insert/erase of *other* elements (std::unordered_* semantics).
//       Use as the drop-in replacement for element stores that may hand out
//       long-lived references.
//
// The default hash is std::hash<K> for both backends, so the existing custom
// std::hash specializations and colmap::PairHash (see util/types.h) are reused
// unchanged and the hashing semantics are held constant across backends.
// boost::unordered internally re-mixes a non-avalanching hash such as the
// identity std::hash<uint32_t>.
//
// WARNING: these aliases are data members of classes in public headers, so the
// backend changes their layout. Nothing about the choice reaches the linker, so
// mixing backends corrupts memory instead of failing to link.

#include <functional>

#if defined(COLMAP_HASH_BOOST)
#include <boost/unordered/unordered_flat_map.hpp>
#include <boost/unordered/unordered_flat_set.hpp>
#include <boost/unordered/unordered_node_map.hpp>
#include <boost/unordered/unordered_node_set.hpp>
#else
// Default to the standard library when no backend macro is defined.
#ifndef COLMAP_HASH_STD
#define COLMAP_HASH_STD
#endif
#include <unordered_map>
#include <unordered_set>
#endif

namespace colmap {

#if defined(COLMAP_HASH_BOOST)

// Identifies the layout of every class storing these containers.
inline constexpr const char* kHashMapBackend = "boost";

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

#else  // COLMAP_HASH_STD

inline constexpr const char* kHashMapBackend = "std";

template <class Key,
          class Value,
          class Hash = std::hash<Key>,
          class Eq = std::equal_to<Key>>
using FlatHashMap = std::unordered_map<Key, Value, Hash, Eq>;

template <class Key, class Hash = std::hash<Key>, class Eq = std::equal_to<Key>>
using FlatHashSet = std::unordered_set<Key, Hash, Eq>;

template <class Key,
          class Value,
          class Hash = std::hash<Key>,
          class Eq = std::equal_to<Key>>
using NodeHashMap = std::unordered_map<Key, Value, Hash, Eq>;

template <class Key, class Hash = std::hash<Key>, class Eq = std::equal_to<Key>>
using NodeHashSet = std::unordered_set<Key, Hash, Eq>;

#endif

}  // namespace colmap
