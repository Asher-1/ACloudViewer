// Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: Asher (Dahai Lu)

#include "util/misc.h"
#include "util/option_manager.h"

namespace cloudViewer {

class OptionsParser {
public:
    OptionsParser() {
        argc = 0;
        argv = nullptr;
        reset();
    }

    ~OptionsParser() { reset(); }

    bool parseOptions() { return parseOptions(this->argc, this->argv); }

    inline int getArgc() { return this->argc; }
    inline char** getArgv() { return this->argv; }

    template <typename T>
    void registerOption(const std::string& name, const T* option) {
        if (std::is_same<T, bool>::value) {
            options_bool_.emplace_back(name,
                                       reinterpret_cast<const bool*>(option));
        } else if (std::is_same<T, int>::value) {
            options_int_.emplace_back(name,
                                      reinterpret_cast<const int*>(option));
        } else if (std::is_same<T, double>::value) {
            options_double_.emplace_back(
                    name, reinterpret_cast<const double*>(option));
        } else if (std::is_same<T, std::string>::value) {
            if (!reinterpret_cast<const std::string*>(option)->empty()) {
                options_string_.emplace_back(
                        name, reinterpret_cast<const std::string*>(option));
            }
        } else {
            std::cerr << "Unsupported option type" << std::endl;
        }
    }

    void reset() {
        releaseOptions();
        options_bool_.clear();
        options_int_.clear();
        options_double_.clear();
        options_string_.clear();
    }

    bool parseOptions(int& argc, char**& argv) {
        // First, put all options without a section and then those with a
        // section. This is necessary as otherwise older Boost versions will
        // write the options without a section in between other sections and
        // therefore the errors will be assigned to the wrong section if read
        // later.

        ReleaseOptions(argc, argv);
        unsigned long capacity = 2 * getParametersCount() + 1;
        if (capacity == 0) {
            return false;
        }

        // add application name
        argv = new char*[capacity];
        setValue("options", 0, argv);
        argc = 1;

        // add other optinons
        for (const auto& option : options_bool_) {
            setValue("--" + option.first, argc, argv);
            argc += 1;
            bool bool_flag = *option.second;
            if (bool_flag)
                setValue("true", argc, argv);
            else
                setValue("false", argc, argv);
            argc += 1;
        }

        for (const auto& option : options_int_) {
            setValue("--" + option.first, argc, argv);
            argc += 1;
            setValue(std::to_string(*option.second), argc, argv);
            argc += 1;
        }

        for (const auto& option : options_double_) {
            setValue("--" + option.first, argc, argv);
            argc += 1;
            setValue(std::to_string(*option.second), argc, argv);
            argc += 1;
        }

        for (const auto& option : options_string_) {
            setValue("--" + option.first, argc, argv);
            argc += 1;
            setValue(*option.second, argc, argv);
            argc += 1;
        }

        if (argc == 0 || !argv) return false;

        return true;
    }

    static void ReleaseOptions(int argc, char** argv) {
        if (argv) {
            for (int i = 0; i < argc; ++i) {
                if (argv[i]) {
                    delete[] argv[i];
                }
            }
            argv = nullptr;
        }
    }

private:
    void releaseOptions() {
        if (this->argc > 0 && this->argv) {
            ReleaseOptions(this->argc, this->argv);
        }
        this->argc = 0;
    }

    void setValue(const std::string& value, int argc, char** argv) {
        unsigned long size = value.length() * sizeof(char);
        argv[argc] = static_cast<char*>(malloc(size));
        strncpy(argv[argc], value.c_str(), value.length());
    }

    unsigned long getParametersCount() {
        return static_cast<unsigned long>(
                options_int_.size() + options_bool_.size() +
                options_double_.size() + options_string_.size());
    }

    int argc;
    char** argv;
    std::vector<std::pair<std::string, const bool*>> options_bool_;
    std::vector<std::pair<std::string, const int*>> options_int_;
    std::vector<std::pair<std::string, const double*>> options_double_;
    std::vector<std::pair<std::string, const std::string*>> options_string_;
};

}  // namespace cloudViewer
