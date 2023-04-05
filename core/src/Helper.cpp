// ----------------------------------------------------------------------------
// -                        cloudViewer: asher-1.github.io                          -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 asher-1.github.io
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#include "Helper.h"

#include <fmt/chrono.h>

#include <algorithm>
#include <cctype>
#include <unordered_set>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif  // _WIN32

namespace cloudViewer {
namespace utility {

namespace {
// The StringAppendV function is borrowed from Google under the BSD license:
//
// Copyright 2012 Google Inc.  All rights reserved.
// https://developers.google.com/protocol-buffers/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
//       copyright notice, this list of conditions and the following disclaimer
//       in the documentation and/or other materials provided with the
//       distribution.
//     * Neither the name of Google Inc. nor the names of its
//       contributors may be used to endorse or promote products derived from
//       this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

void StringAppendV(std::string *dst, const char *format, va_list ap) {
    // First try with a small fixed size buffer.
    static const int kFixedBufferSize = 1024;
    char fixed_buffer[kFixedBufferSize];

    // It is possible for methods that use a va_list to invalidate
    // the data in it upon use.  The fix is to make a copy
    // of the structure before using it and use that copy instead.
    va_list backup_ap;
    va_copy(backup_ap, ap);
    int result = vsnprintf(fixed_buffer, kFixedBufferSize, format, backup_ap);
    va_end(backup_ap);

    if (result < kFixedBufferSize) {
        if (result >= 0) {
            // Normal case - everything fits.
            dst->append(fixed_buffer, result);
            return;
        }

#ifdef _MSC_VER
        // Error or MSVC running out of space.  MSVC 8.0 and higher
        // can be asked about space needed with the special idiom below:
        va_copy(backup_ap, ap);
        result = vsnprintf(nullptr, 0, format, backup_ap);
        va_end(backup_ap);
#endif
    }

    // Increase the buffer size to the size requested by vsnprintf,
    // plus one for the closing \0.
    const int variable_buffer_size = result + 1;
    std::unique_ptr<char[]> variable_buffer(new char[variable_buffer_size]);

    // Restore the va_list before we use it again.
    va_copy(backup_ap, ap);
    result =
            vsnprintf(variable_buffer.get(), variable_buffer_size, format, backup_ap);
    va_end(backup_ap);

    if (result >= 0 && result < variable_buffer_size) {
        dst->append(variable_buffer.get(), result);
    }
}

bool IsNotWhiteSpace(const int character) {
    return character != ' ' && character != '\n' && character != '\r' &&
           character != '\t';
}

}  // namespace

std::string StringPrintf(const char *format, ...) {
    va_list ap;
    va_start(ap, format);
    std::string result;
    StringAppendV(&result, format, ap);
    va_end(ap);
    return result;
}

std::string StringReplace(const std::string &str, const std::string &old_str,
                          const std::string &new_str) {
    if (old_str.empty()) {
        return str;
    }
    size_t position = 0;
    std::string mod_str = str;
    while ((position = mod_str.find(old_str, position)) != std::string::npos) {
        mod_str.replace(position, old_str.size(), new_str);
        position += new_str.size();
    }
    return mod_str;
}

std::string StringReplaceFirst(const std::string &str,
                               const std::string &old_str,
                               const std::string &new_str) {
    std::string mod_str = str;
    size_t start_pos = mod_str.find(old_str);
    if (start_pos == std::string::npos) return mod_str;
    mod_str.replace(start_pos, old_str.length(), new_str);
    return mod_str;
}

std::string StringReplaceLast(const std::string &str,
                              const std::string &old_str,
                              const std::string &new_str) {
    std::string mod_str = str;
    size_t start_pos = mod_str.rfind(old_str);
    if (start_pos == std::string::npos) return mod_str;
    mod_str.replace(start_pos, old_str.length(), new_str);
    return mod_str;
}

bool StringContains(const std::string &src, const std::string &dst) {
    return src.find(dst) != std::string::npos;
}

bool StringStartsWith(const std::string &str, const std::string &prefix) {
    return !prefix.empty() && prefix.size() <= str.size() &&
           str.substr(0, prefix.size()) == prefix;
}

bool StringEndsWith(const std::string &str, const std::string &postfix) {
    return !postfix.empty() && postfix.size() <= str.size() &&
           str.substr(str.size() - postfix.size(), str.size()) == postfix;
}

std::string JoinStrings(const std::vector<std::string> &strs,
                        const std::string &delimiter) {
    std::ostringstream oss;
    for (size_t i = 0; i < strs.size(); ++i) {
        oss << strs[i];
        if (i != strs.size() - 1) {
            oss << delimiter;
        }
    }
    return oss.str();
}

std::vector<std::string> StringSplit(const std::string &str,
                                     const std::string &delimiters /* = " "*/,
                                     bool trim_empty_str /* = true*/) {
    std::vector<std::string> tokens;
    std::string::size_type pos = 0, last_pos = 0;
    std::string::size_type new_pos;
    while (pos != std::string::npos) {
        pos = str.find_first_of(delimiters, last_pos);
        new_pos = (pos == std::string::npos ? str.length() : pos);
        if (new_pos != last_pos || !trim_empty_str) {
            tokens.push_back(str.substr(last_pos, new_pos - last_pos));
        }
        last_pos = new_pos + 1;
    }
    return tokens;
}

void SplitString(std::vector<std::string>& tokens,
                 const std::string& str,
                 const std::string& delimiters /* = " "*/,
                 bool trim_empty_str /* = true*/) {
    std::string::size_type pos = 0, new_pos = 0, last_pos = 0;
    while (pos != std::string::npos) {
        pos = str.find_first_of(delimiters, last_pos);
        new_pos = (pos == std::string::npos ? str.length() : pos);
        if (new_pos != last_pos || !trim_empty_str) {
            tokens.push_back(str.substr(last_pos, new_pos - last_pos));
        }
        last_pos = new_pos + 1;
    }
}

std::vector<std::string> SplitString(const std::string& str,
                                     const std::string& delimiters /* = " "*/,
                                     bool trim_empty_str /* = true*/) {
    std::vector<std::string> tokens;
    std::string::size_type pos = 0, new_pos = 0, last_pos = 0;
    while (pos != std::string::npos) {
        pos = str.find_first_of(delimiters, last_pos);
        new_pos = (pos == std::string::npos ? str.length() : pos);
        if (new_pos != last_pos || !trim_empty_str) {
            tokens.push_back(str.substr(last_pos, new_pos - last_pos));
        }
        last_pos = new_pos + 1;
    }
    return tokens;
}

std::string& LeftStripString(std::string& str, const std::string& chars) {
    str.erase(0, str.find_first_not_of(chars));
    return str;
}

std::string& RightStripString(std::string& str, const std::string& chars) {
    str.erase(str.find_last_not_of(chars) + 1);
    return str;
}

std::string& StripString(std::string& str, const std::string& chars) {
    return LeftStripString(RightStripString(str, chars), chars);
}

std::string ToLower(const std::string& str) {
    std::string out = str;
    std::transform(str.begin(), str.end(), out.begin(),
        [](unsigned char c) { return std::tolower(c); });
    return out;
}

std::string ToUpper(const std::string& str) {
    std::string out = str;
    std::transform(str.begin(), str.end(), out.begin(),
        [](unsigned char c) { return std::toupper(c); });
    return out;
}

// Count the length of current word starting from start_pos
size_t WordLength(const std::string& doc,
                  size_t start_pos,
                  const std::string& valid_chars) {
    std::unordered_set<char> valid_chars_set;
    for (const char& c : valid_chars) {
        valid_chars_set.insert(c);
    }
    auto is_word_char = [&valid_chars_set](const char& c) {
        return std::isalnum(c) ||
               valid_chars_set.find(c) != valid_chars_set.end();
    };
    size_t length = 0;
    for (size_t pos = start_pos; pos < doc.size(); ++pos) {
        if (!is_word_char(doc[pos])) {
            break;
        }
        length++;
    }
    return length;
}

void Sleep(int milliseconds) {
#ifdef _WIN32
    ::Sleep(milliseconds);
#else
    usleep(milliseconds * 1000);
#endif  // _WIN32
}

std::string GetCurrentTimeStamp() {
    std::time_t t = std::time(nullptr);
    return fmt::format("{:%Y-%m-%d-%H-%M-%S}", *std::localtime(&t));
}

}  // namespace utility
}  // namespace cloudViewer
