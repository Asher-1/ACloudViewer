// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef COLMAP_SRC_UTIL_LOGGING_H_
#define COLMAP_SRC_UTIL_LOGGING_H_

#include <glog/logging.h>

#include <iostream>

#include "util/string.h"

// Option checker macros. In contrast to glog, this function does not abort the
// program, but simply returns false on failure.
#define CHECK_OPTION_IMPL(expr) \
    __CheckOptionImpl(__FILE__, __LINE__, (expr), #expr)
#define CHECK_OPTION(expr)                                       \
    if (!__CheckOptionImpl(__FILE__, __LINE__, (expr), #expr)) { \
        return false;                                            \
    }
#define CHECK_OPTION_OP(name, op, val1, val2)                                \
    if (!__CheckOptionOpImpl(__FILE__, __LINE__, (val1 op val2), val1, val2, \
                             #val1, #val2, #op)) {                           \
        return false;                                                        \
    }
#define CHECK_OPTION_EQ(val1, val2) CHECK_OPTION_OP(_EQ, ==, val1, val2)
#define CHECK_OPTION_NE(val1, val2) CHECK_OPTION_OP(_NE, !=, val1, val2)
#define CHECK_OPTION_LE(val1, val2) CHECK_OPTION_OP(_LE, <=, val1, val2)
#define CHECK_OPTION_LT(val1, val2) CHECK_OPTION_OP(_LT, <, val1, val2)
#define CHECK_OPTION_GE(val1, val2) CHECK_OPTION_OP(_GE, >=, val1, val2)
#define CHECK_OPTION_GT(val1, val2) CHECK_OPTION_OP(_GT, >, val1, val2)

namespace colmap {

// Initialize glog at the beginning of the program.
void InitializeGlog(char** argv);

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

const char* __GetConstFileBaseName(const char* file);

bool __CheckOptionImpl(const char* file,
                       const int line,
                       const bool result,
                       const char* expr_str);

template <typename T1, typename T2>
bool __CheckOptionOpImpl(const char* file,
                         const int line,
                         const bool result,
                         const T1& val1,
                         const T2& val2,
                         const char* val1_str,
                         const char* val2_str,
                         const char* op_str) {
    if (result) {
        return true;
    } else {
        std::cerr << StringPrintf("[%s:%d] Check failed: %s %s %s (%s vs. %s)",
                                  __GetConstFileBaseName(file), line, val1_str,
                                  op_str, val2_str,
                                  std::to_string(val1).c_str(),
                                  std::to_string(val2).c_str())
                  << std::endl;
        return false;
    }
}

}  // namespace colmap

#endif  // COLMAP_SRC_UTIL_LOGGING_H_
