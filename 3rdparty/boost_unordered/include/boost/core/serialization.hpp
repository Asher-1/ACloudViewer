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

// NOTE (ACloudViewer): do NOT define boost::serialization::nvp from this
// vendor tree. Boost < 1.74's boost/serialization/nvp.hpp defines its own
// nvp class under a different include guard, so a vendor-defined nvp makes
// any TU that later pulls the system header hit "redefinition of struct
// boost::serialization::nvp" (measured on Ubuntu 20.04 / Boost 1.71 via
// boost/graph -> multi_index -> serialization/nvp.hpp). Forward-declaring
// the name keeps the serialize templates parseable without instantiating
// them (no ColmapLib TU boost-serializes these containers), while on
// Boost >= 1.74 the system serialization/nvp.hpp itself includes
// <boost/core/nvp.hpp>, which still lands on the vendor copy and defines
// the class exactly once.
#include <boost/core/addressof.hpp>
#include <cstddef>

namespace boost
{

namespace serialization
{

template<class T> class nvp;
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

// make_nvp is defined here directly instead of aliasing
// serialization::make_nvp: that name is owned by whichever header defines
// serialization::nvp for the system Boost in use (Boost < 1.74:
// boost/serialization/nvp.hpp; Boost >= 1.74: boost/core/nvp.hpp), and a
// duplicate declaration here would only make calls ambiguous. Constructing
// serialization::nvp<T> needs a complete type, so this is instantiated only
// on actual serialization use (none in ColmapLib).
template<class T>
inline const serialization::nvp<T> make_nvp(const char* name, T& value)
{
    return serialization::nvp<T>(name, value);
}

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
