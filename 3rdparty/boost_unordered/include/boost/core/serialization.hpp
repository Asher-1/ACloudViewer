#ifndef BOOST_CORE_SERIALIZATION_HPP_INCLUDED
#define BOOST_CORE_SERIALIZATION_HPP_INCLUDED

// MS compatible compilers support #pragma once

#if defined(_MSC_VER) && (_MSC_VER >= 1020)
# pragma once
#endif

// Copyright 2023 Peter Dimov
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt
//
// Utilities needed to implement serialization support
// without including a Boost.Serialization header

// NOTE (ACloudViewer): this header shadows the system Boost header of the
// same name for every ColmapLib TU (the vendored boost/unordered 1.92
// closure is searched first, see libs/Reconstruction/src/CMakeLists.txt),
// and Boost ships two generations of consumers for it:
//   * Boost < 1.84: boost/serialization/nvp.hpp owns
//     boost::serialization::nvp — Boost < 1.74 defines its own legacy nvp
//     struct under the BOOST_SERIALIZATION_NVP_HPP guard, Boost 1.74-1.83
//     instead include boost/core/nvp.hpp (which resolves to this vendor
//     tree). boost/graph -> multi_index -> serialization/nvp.hpp pulls it
//     into every property_tree/graph TU.
//   * Boost >= 1.84: system multi_index_container.hpp (pulled by
//     property_tree -> ptree) includes <boost/core/serialization.hpp>
//     itself and instantiates serialization::nvp (complete type) plus
//     serialization::make_nvp from it — a forward declaration breaks the
//     build there (measured on macOS conda Boost 1.84).
// A vendor definition is therefore mandatory on >= 1.84, harmless on
// 1.74-1.83, and must yield to the legacy definition on < 1.74 — hence
// the guard handshake below: define through core/nvp.hpp and claim
// BOOST_SERIALIZATION_NVP_HPP only when no legacy system header has
// already defined nvp earlier in the TU. With exactly one definition per
// TU every supported Boost (1.71 ... 1.92) compiles; the residual
// mixed-shape ODR surface is confined to never-instantiated serialization
// paths on Boost < 1.74 (no ColmapLib TU boost-serializes these
// containers).
#if !defined(BOOST_SERIALIZATION_NVP_HPP)
#include <boost/core/nvp.hpp>
#define BOOST_SERIALIZATION_NVP_HPP
#endif
#include <cstddef>

namespace boost
{

namespace serialization
{

// Forward declarations (needed for specializations)

template<class T> struct version;

class access;

// Our own version_type replacement. This has to be in
// the `serialization` namespace, because its only purpose
// is to add `serialization` as an associated namespace.

struct core_version_type
{
    unsigned int version_;

    core_version_type( unsigned int version ): version_( version ) {}
    operator unsigned int () const { return version_; }
};

} // namespace serialization

namespace core
{

// nvp

using serialization::nvp;
using serialization::make_nvp;

// split_free

namespace detail
{

template<bool IsSaving> struct load_or_save_f;

template<> struct load_or_save_f<true>
{
    template<class A, class T> void operator()( A& a, T& t, unsigned int v ) const
    {
        save( a, t, serialization::core_version_type( v ) );
    }
};

template<> struct load_or_save_f<false>
{
    template<class A, class T> void operator()( A& a, T& t, unsigned int v ) const
    {
        load( a, t, serialization::core_version_type( v ) );
    }
};

} // namespace detail

template<class A, class T> inline void split_free( A& a, T& t, unsigned int v )
{
    detail::load_or_save_f< A::is_saving::value >()( a, t, v );
}

// split_member

namespace detail
{

template<bool IsSaving, class Access = serialization::access> struct load_or_save_m;

template<class Access> struct load_or_save_m<true, Access>
{
    template<class A, class T> void operator()( A& a, T const& t, unsigned int v ) const
    {
        Access::member_save( a, t, v );
    }
};

template<class Access> struct load_or_save_m<false, Access>
{
    template<class A, class T> void operator()( A& a, T& t, unsigned int v ) const
    {
        Access::member_load( a, t, v );
    }
};

} // namespace detail

template<class A, class T> inline void split_member( A& a, T& t, unsigned int v )
{
    detail::load_or_save_m< A::is_saving::value >()( a, t, v );
}

// load_construct_data_adl

template<class Ar, class T> void load_construct_data_adl( Ar& ar, T* t, unsigned int v )
{
    load_construct_data( ar, t, serialization::core_version_type( v ) );
}

// save_construct_data_adl

template<class Ar, class T> void save_construct_data_adl( Ar& ar, T const* t, unsigned int v )
{
    save_construct_data( ar, t, serialization::core_version_type( v ) );
}

} // namespace core
} // namespace boost

#endif  // #ifndef BOOST_CORE_SERIALIZATION_HPP_INCLUDED
