// ----------------------------------------------------------------------------
// -                        CloudViewer: asher-1.github.io                    -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 asher-1.github.io
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

#include <iostream>

#include "CloudViewer.h"

void PrintHelp() {
    using namespace cloudViewer;

    PrintCloudViewerVersion();
    // clang-format off
    utility::LogInfo("Usage:");
    utility::LogInfo("    > FileSystem ls [dir]");
    utility::LogInfo("    > FileSystem mkdir [dir]");
    utility::LogInfo("    > FileSystem rmdir [dir]");
    utility::LogInfo("    > FileSystem rmfile [file]");
    utility::LogInfo("    > FileSystem fileexists [file]");
    // clang-format on
    utility::LogInfo("");
}

int main(int argc, char *argv[]) {
    using namespace cloudViewer;
    using namespace cloudViewer::utility::filesystem;
    if (!(argc == 2 || argc == 3) ||
        utility::ProgramOptionExistsAny(argc, argv, {"-h", "--help"})) {
        PrintHelp();
        return 1;
    }

    std::string directory, function;
    function = std::string(argv[1]);
    if (argc == 2) {
        directory = ".";
    } else {
        directory = std::string(argv[2]);
    }

    if (function == "ls") {
        std::vector<std::string> filenames;
        ListFilesInDirectory(directory, filenames);

        for (const auto &filename : filenames) {
            std::cout << filename << std::endl;
            std::cout << "parent dir name is : "
                      << GetFileParentDirectory(filename) << std::endl;
            std::cout << "file name only is : "
                      << GetFileNameWithoutDirectory(filename) << std::endl;
            std::cout << "extension name is : "
                      << GetFileExtensionInLowerCase(filename) << std::endl;
            std::cout << "file name without extension is : "
                      << GetFileNameWithoutExtension(filename) << std::endl;
            std::cout << std::endl;
        }
    } else if (function == "mkdir") {
        bool success = MakeDirectoryHierarchy(directory);
        std::cout << "mkdir " << (success ? "succeeded" : "failed")
                  << std::endl;
    } else if (function == "rmdir") {
        bool success = DeleteDirectory(directory);
        std::cout << "rmdir " << (success ? "succeeded" : "failed")
                  << std::endl;
    } else if (function == "rmfile") {
        bool success = RemoveFile(directory);
        std::cout << "rmfile " << (success ? "succeeded" : "failed")
                  << std::endl;
    } else if (function == "fileexists") {
        bool success = FileExists(directory);
        std::cout << "fileexists " << (success ? "yes" : "no") << std::endl;
    }
    return 1;
}
