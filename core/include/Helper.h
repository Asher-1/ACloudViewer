// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <cmath>
#include <cstdlib>
#include <functional>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "CVCoreLib.h"

namespace cloudViewer {
namespace utility {

/// The namespace hash_tuple defines a general hash function for std::tuple
/// See this post for details:
///   http://stackoverflow.com/questions/7110301
/// The hash_combine code is from boost
/// Reciprocal of the golden ratio helps spread entropy and handles duplicates.
/// See Mike Seymour in magic-numbers-in-boosthash-combine:
///   http://stackoverflow.com/questions/4948780

template <typename TT>
struct hash_tuple {
    size_t operator()(TT const& tt) const { return std::hash<TT>()(tt); }
};

namespace {

template <class T>
inline void hash_combine(std::size_t& hash_seed, T const& v) {
    hash_seed ^= std::hash<T>()(v) + 0x9e3779b9 + (hash_seed << 6) +
                 (hash_seed >> 2);
}

template <class Tuple, size_t Index = std::tuple_size<Tuple>::value - 1>
struct HashValueImpl {
    static void apply(size_t& seed, Tuple const& tuple) {
        HashValueImpl<Tuple, Index - 1>::apply(seed, tuple);
        hash_combine(seed, std::get<Index>(tuple));
    }
};

template <class Tuple>
struct HashValueImpl<Tuple, 0> {
    static void apply(size_t& seed, Tuple const& tuple) {
        hash_combine(seed, std::get<0>(tuple));
    }
};

}  // unnamed namespace

template <typename... TT>
struct hash_tuple<std::tuple<TT...>> {
    size_t operator()(std::tuple<TT...> const& tt) const {
        size_t hash_seed = 0;
        HashValueImpl<std::tuple<TT...>>::apply(hash_seed, tt);
        return hash_seed;
    }
};

template <typename T>
struct hash_eigen {
    std::size_t operator()(T const& matrix) const {
        size_t hash_seed = 0;
        for (int i = 0; i < (int)matrix.size(); i++) {
            auto elem = *(matrix.data() + i);
            hash_seed ^= std::hash<typename T::Scalar>()(elem) + 0x9e3779b9 +
                         (hash_seed << 6) + (hash_seed >> 2);
        }
        return hash_seed;
    }
};

// Hash function for enum class for C++ standard less than C++14
// https://stackoverflow.com/a/24847480/1255535
struct CV_CORE_LIB_API hash_enum_class {
    template <typename T>
    std::size_t operator()(T t) const {
        return static_cast<std::size_t>(t);
    }
};

// Format string by replacing embedded format specifiers with their respective
// values, see `printf` for more details. This is a modified implementation
// of Google's BSD-licensed StringPrintf function.
std::string CV_CORE_LIB_API StringPrintf(const char* format, ...);

// Replace all occurrences of `old_str` with `new_str` in the given string.
std::string CV_CORE_LIB_API StringReplace(const std::string& str,
                                          const std::string& old_str,
                                          const std::string& new_str);

/// Returns true of the source string contains the destination string.
/// \param src Source string.
/// \param dst Destination string.
bool CV_CORE_LIB_API StringContains(const std::string& src,
                                    const std::string& dst);

std::string CV_CORE_LIB_API StringReplaceFirst(const std::string& str,
                                               const std::string& old_str,
                                               const std::string& new_str);

std::string CV_CORE_LIB_API StringReplaceLast(const std::string& str,
                                              const std::string& old_str,
                                              const std::string& new_str);

// Check whether a string starts with a certain prefix.
bool CV_CORE_LIB_API StringStartsWith(const std::string& str,
                                      const std::string& prefix);

// Check whether a string ends with a certain postfix.
bool CV_CORE_LIB_API StringEndsWith(const std::string& str,
                                    const std::string& postfix);

std::string CV_CORE_LIB_API JoinStrings(const std::vector<std::string>& strs,
                                        const std::string& delimiter = ", ");

/// Function to split a string
/// http://stackoverflow.com/questions/236129/split-a-string-in-c
std::vector<std::string> CV_CORE_LIB_API
StringSplit(const std::string& str,
            const std::string& delimiters = " ",
            bool trim_empty_str = true);

/// Function to split a string, mimics boost::split
/// http://stackoverflow.com/questions/236129/split-a-string-in-c
void CV_CORE_LIB_API SplitString(std::vector<std::string>& tokens,
                                 const std::string& str,
                                 const std::string& delimiters = " ",
                                 bool trim_empty_str = true);

/// Function to split a string, mimics boost::split
/// http://stackoverflow.com/questions/236129/split-a-string-in-c
std::vector<std::string> CV_CORE_LIB_API
SplitString(const std::string& str,
            const std::string& delimiters = " ",
            bool trim_empty_str = true);

/// String util: find length of current word staring from a position
/// By default, alpha numeric chars and chars in valid_chars are considered
/// as valid charactors in a word
size_t CV_CORE_LIB_API WordLength(const std::string& doc,
                                  size_t start_pos,
                                  const std::string& valid_chars = "_");

CV_CORE_LIB_API std::string& LeftStripString(
        std::string& str, const std::string& chars = "\t\n\v\f\r ");

CV_CORE_LIB_API std::string& RightStripString(
        std::string& str, const std::string& chars = "\t\n\v\f\r ");

/// Strip empty charactors in front and after string. Similar to Python's
/// str.strip()
CV_CORE_LIB_API std::string& StripString(
        std::string& str, const std::string& chars = "\t\n\v\f\r ");

/// Convert string to the lower case
std::string CV_CORE_LIB_API ToLower(const std::string& s);

/// Convert string to the upper case
std::string CV_CORE_LIB_API ToUpper(const std::string& s);

/// Format string
template <typename... Args>
inline std::string FormatString(const std::string& format, Args... args) {
    int size_s = std::snprintf(nullptr, 0, format.c_str(), args...) +
                 1;  // Extra space for '\0'
    if (size_s <= 0) {
        throw std::runtime_error("Error during formatting.");
    }
    auto size = static_cast<size_t>(size_s);
    auto buf = std::make_unique<char[]>(size);
    std::snprintf(buf.get(), size, format.c_str(), args...);
    return std::string(buf.get(),
                       buf.get() + size - 1);  // We don't want the '\0' inside
};

/// Format string fast (Unix / BSD Only)
template <typename... Args>
inline std::string FastFormatString(const std::string& format, Args... args) {
#ifdef _WIN32
    return FormatString(format, args...);
#else
    char* buffer = nullptr;
    int size_s = asprintf(&buffer, format.c_str(), args...);
    if (size_s == -1) {
        throw std::runtime_error("Error during formatting.");
    }
    auto ret = std::string(buffer,
                           buffer + size_s);  // no + 1 since we ignore the \0
    std::free(buffer);                        // asprintf calls malloc
    return ret;
#endif  // _WIN32
};

void CV_CORE_LIB_API Sleep(int milliseconds);

/// Computes the quotient of x/y with rounding up
inline int CV_CORE_LIB_API DivUp(int x, int y) {
    div_t tmp = std::div(x, y);
    return tmp.quot + (tmp.rem != 0 ? 1 : 0);
}

/// \class UniformRandIntGenerator
///
/// \brief Draw pseudo-random integers bounded by min and max (inclusive)
/// from a uniform distribution
class CV_CORE_LIB_API UniformRandIntGenerator {
public:
    UniformRandIntGenerator(
            const int min,
            const int max,
            std::mt19937::result_type seed = std::random_device{}())
        : distribution_(min, max), generator_(seed) {}
    int operator()() { return distribution_(generator_); }

protected:
    std::uniform_int_distribution<int> distribution_;
    std::mt19937 generator_;
};

/// Returns current time stamp.
std::string CV_CORE_LIB_API GetCurrentTimeStamp();

}  // namespace utility
}  // namespace cloudViewer
