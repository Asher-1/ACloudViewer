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

#include <FileSystem.h>

#include <fcntl.h>
#include <sys/stat.h>

#include <algorithm>

#include "utility/Console.h"
#include "tests/UnitTest.h"

namespace cloudViewer {
namespace tests {

// ----------------------------------------------------------------------------
// Get the file extension and convert to lower case.
// ----------------------------------------------------------------------------
TEST(FileSystem, GetFileExtensionInLowerCase) {
    std::string path;
    std::string result;

    // empty
    path = "";
    result = utility::filesystem::GetFileExtensionInLowerCase(path);
    EXPECT_EQ("", result);

    // no folder tree
    path = "fileName.EXT";
    result = utility::filesystem::GetFileExtensionInLowerCase(path);
    EXPECT_EQ("ext", result);

    // just a dot
    path = "fileName.";
    result = utility::filesystem::GetFileExtensionInLowerCase(path);
    EXPECT_EQ("", result);

    path = "test/test_dir/fileName.EXT";
    result = utility::filesystem::GetFileExtensionInLowerCase(path);
    EXPECT_EQ("ext", result);

    // no extension
    path = "test/test_dir/fileName";
    result = utility::filesystem::GetFileExtensionInLowerCase(path);
    EXPECT_EQ("", result);

    // multiple extensions
    path = "test/test_dir/fileName.EXT.EXT";
    result = utility::filesystem::GetFileExtensionInLowerCase(path);
    EXPECT_EQ("ext", result);

    // multiple dots
    path = "test/test_dir/fileName..EXT";
    result = utility::filesystem::GetFileExtensionInLowerCase(path);
    EXPECT_EQ("ext", result);

    // dot before the /
    path = "test/utility::filesystem.EXT/fileName";
    result = utility::filesystem::GetFileExtensionInLowerCase(path);
    EXPECT_EQ("", result);

    // space in file name
    path = "test/test_dir/fileName .EXT";
    result = utility::filesystem::GetFileExtensionInLowerCase(path);
    EXPECT_EQ("ext", result);

    // space in extension
    path = "test/test_dir/fileName. EXT";
    result = utility::filesystem::GetFileExtensionInLowerCase(path);
    EXPECT_EQ(" ext", result);
}

// ----------------------------------------------------------------------------
// Should return the file name only, without extension.
// What it actually does is return the full path without the extension.
// ----------------------------------------------------------------------------
TEST(FileSystem, GetFileNameWithoutExtension) {
    std::string path;
    std::string result;

    // empty
    path = "";
    result = utility::filesystem::GetFileNameWithoutExtension(path);
    EXPECT_EQ("", result);

    // no folder tree
    path = "fileName.EXT";
    result = utility::filesystem::GetFileNameWithoutExtension(path);
    EXPECT_EQ("fileName", result);

    path = "test/test_dir/fileName.EXT";
    result = utility::filesystem::GetFileNameWithoutExtension(path);
    EXPECT_EQ("test/test_dir/fileName", result);

    // no extension
    path = "test/test_dir/fileName";
    result = utility::filesystem::GetFileNameWithoutExtension(path);
    EXPECT_EQ("test/test_dir/fileName", result);

    // multiple extensions
    path = "test/test_dir/fileName.EXT.EXT";
    result = utility::filesystem::GetFileNameWithoutExtension(path);
    EXPECT_EQ("test/test_dir/fileName.EXT", result);

    // multiple dots
    path = "test/test_dir/fileName..EXT";
    result = utility::filesystem::GetFileNameWithoutExtension(path);
    EXPECT_EQ("test/test_dir/fileName.", result);

    // space in file name
    path = "test/test_dir/fileName .EXT";
    result = utility::filesystem::GetFileNameWithoutExtension(path);
    EXPECT_EQ("test/test_dir/fileName ", result);
}

TEST(FileSystem, GetFileNameWithoutDirectory) {
    std::string path;
    std::string result;

    // empty
    path = "";
    result = utility::filesystem::GetFileNameWithoutDirectory(path);
    EXPECT_EQ("", result);

    // no folder tree
    path = "fileName.EXT";
    result = utility::filesystem::GetFileNameWithoutDirectory(path);
    EXPECT_EQ("fileName.EXT", result);

    path = "test/test_dir/fileName.EXT";
    result = utility::filesystem::GetFileNameWithoutDirectory(path);
    EXPECT_EQ("fileName.EXT", result);

    // no extension
    path = "test/test_dir/fileName";
    result = utility::filesystem::GetFileNameWithoutDirectory(path);
    EXPECT_EQ("fileName", result);

    // multiple extensions
    path = "test/test_dir/fileName.EXT.EXT";
    result = utility::filesystem::GetFileNameWithoutDirectory(path);
    EXPECT_EQ("fileName.EXT.EXT", result);

    // multiple dots
    path = "test/test_dir/fileName..EXT";
    result = utility::filesystem::GetFileNameWithoutDirectory(path);
    EXPECT_EQ("fileName..EXT", result);

    // space in file name
    path = "test/test_dir/fileName .EXT";
    result = utility::filesystem::GetFileNameWithoutDirectory(path);
    EXPECT_EQ("fileName .EXT", result);
}

// ----------------------------------------------------------------------------
// Get parent directory, terminated in '/'.
// ----------------------------------------------------------------------------
TEST(FileSystem, GetFileParentDirectory) {
    std::string path;
    std::string result;

    // empty
    path = "";
    result = utility::filesystem::GetFileParentDirectory(path);
    EXPECT_EQ("", result);

    // no folder tree
    path = "fileName.EXT";
    result = utility::filesystem::GetFileParentDirectory(path);
    EXPECT_EQ("", result);

    path = "test/test_dir/fileName.EXT";
    result = utility::filesystem::GetFileParentDirectory(path);
    EXPECT_EQ("test/test_dir/", result);

    // no extension
    path = "test/test_dir/fileName";
    result = utility::filesystem::GetFileParentDirectory(path);
    EXPECT_EQ("test/test_dir/", result);

    // multiple extensions
    path = "test/test_dir/fileName.EXT.EXT";
    result = utility::filesystem::GetFileParentDirectory(path);
    EXPECT_EQ("test/test_dir/", result);

    // multiple dots
    path = "test/test_dir/fileName..EXT";
    result = utility::filesystem::GetFileParentDirectory(path);
    EXPECT_EQ("test/test_dir/", result);

    // space in file name
    path = "test/test_dir/fileName .EXT";
    result = utility::filesystem::GetFileParentDirectory(path);
    EXPECT_EQ("test/test_dir/", result);
}

// ----------------------------------------------------------------------------
// Add '/' at the end of the input path, if missing.
// ----------------------------------------------------------------------------
TEST(FileSystem, GetRegularizedDirectoryName) {
    std::string path;
    std::string result;

    path = "";
    result = utility::filesystem::GetRegularizedDirectoryName(path);
    EXPECT_EQ("/", result);

    path = "test/test_dir";
    result = utility::filesystem::GetRegularizedDirectoryName(path);
    EXPECT_EQ("test/test_dir/", result);

    path = "test/test_dir/";
    result = utility::filesystem::GetRegularizedDirectoryName(path);
    EXPECT_EQ("test/test_dir/", result);
}

// ----------------------------------------------------------------------------
// Get/Change the working directory.
// ----------------------------------------------------------------------------
TEST(FileSystem, ChangeWorkingDirectory) {
    std::string path = "test";

    bool status;

    status = utility::filesystem::MakeDirectoryHierarchy(path);
    EXPECT_TRUE(status);

    status = utility::filesystem::ChangeWorkingDirectory(path);
    EXPECT_TRUE(status);

    std::string cwd = utility::filesystem::GetWorkingDirectory();

    EXPECT_EQ(path, utility::filesystem::GetFileNameWithoutDirectory(cwd));

    status = utility::filesystem::ChangeWorkingDirectory("..");
    EXPECT_TRUE(status);

    status = utility::filesystem::DeleteDirectory("test");
    EXPECT_TRUE(status);
}

// ----------------------------------------------------------------------------
// Check if a path exists.
// ----------------------------------------------------------------------------
TEST(FileSystem, DirectoryExists) {
    std::string path = "test/test_dir";

    bool status;

    // path doesn't exist yet
    status = utility::filesystem::DirectoryExists(path);
    EXPECT_FALSE(status);

    // create the path
    status = utility::filesystem::MakeDirectoryHierarchy(path);
    EXPECT_TRUE(status);

    // path exists
    status = utility::filesystem::DirectoryExists(path);
    EXPECT_TRUE(status);

    // clean-up in reverse order, DeleteDirectory can delete one dir at a time.
    status = utility::filesystem::ChangeWorkingDirectory("test");
    EXPECT_TRUE(status);

    status = utility::filesystem::DeleteDirectory("test_dir");
    EXPECT_TRUE(status);

    status = utility::filesystem::ChangeWorkingDirectory("..");
    EXPECT_TRUE(status);

    status = utility::filesystem::DeleteDirectory("test");
    EXPECT_TRUE(status);
}

// ----------------------------------------------------------------------------
// Make a directory.
// Return true if the directory was created.
// Return false otherwise. This could mean that the directory already exists.
// ----------------------------------------------------------------------------
TEST(FileSystem, MakeDirectory) {
    std::string path = "test";

    bool status;

    status = utility::filesystem::MakeDirectory(path);
    EXPECT_TRUE(status);

    status = utility::filesystem::MakeDirectory(path);
    EXPECT_FALSE(status);

    status = utility::filesystem::DeleteDirectory("test");
    EXPECT_TRUE(status);
}

// ----------------------------------------------------------------------------
// Make a hierarchy of directories. Equivalent to 'mkdir -p ...'.
// ----------------------------------------------------------------------------
TEST(FileSystem, MakeDirectoryHierarchy) {
    std::string path = "test/test_dir";

    bool status;

    status = utility::filesystem::MakeDirectoryHierarchy(path);
    EXPECT_TRUE(status);

    // clean-up in reverse order, DeleteDirectory can delete one dir at a time.
    status = utility::filesystem::ChangeWorkingDirectory("test");
    EXPECT_TRUE(status);

    status = utility::filesystem::DeleteDirectory("test_dir");
    EXPECT_TRUE(status);

    status = utility::filesystem::ChangeWorkingDirectory("..");
    EXPECT_TRUE(status);

    status = utility::filesystem::DeleteDirectory("test");
    EXPECT_TRUE(status);
}

// ----------------------------------------------------------------------------
// Note: DeleteDirectory can delete one dir at a time.
// ----------------------------------------------------------------------------
TEST(FileSystem, DeleteDirectory) {
    std::string path = "test";

    bool status;

    status = utility::filesystem::MakeDirectory(path);
    EXPECT_TRUE(status);

    status = utility::filesystem::DeleteDirectory(path);
    EXPECT_TRUE(status);

    status = utility::filesystem::DeleteDirectory(path);
    EXPECT_FALSE(status);
}

TEST(FileSystem, File_Exists_Remove) {
    std::string path = "test/test_dir";
    std::string fileName = "fileName.ext";

    bool status;

    status = utility::filesystem::MakeDirectoryHierarchy(path);
    EXPECT_TRUE(status);

    status = utility::filesystem::ChangeWorkingDirectory(path);
    EXPECT_TRUE(status);

    status = utility::filesystem::FileExists(fileName);
    EXPECT_FALSE(status);

    creat(fileName.c_str(), 0);

    status = utility::filesystem::FileExists(fileName);
    EXPECT_TRUE(status);

    status = utility::filesystem::RemoveFile(fileName);
    EXPECT_TRUE(status);

    status = utility::filesystem::FileExists(fileName);
    EXPECT_FALSE(status);

    // clean-up in reverse order, DeleteDirectory can delete one dir at a time.
    status = utility::filesystem::ChangeWorkingDirectory("..");
    EXPECT_TRUE(status);

    status = utility::filesystem::DeleteDirectory("test_dir");
    EXPECT_TRUE(status);

    status = utility::filesystem::ChangeWorkingDirectory("..");
    EXPECT_TRUE(status);

    status = utility::filesystem::DeleteDirectory("test");
    EXPECT_TRUE(status);
}

// ----------------------------------------------------------------------------
// List all files in the specified directory.
// ----------------------------------------------------------------------------
TEST(FileSystem, ListFilesInDirectory) {
    std::string path = "test/test_dir";
    std::vector<std::string> fileNames = {"fileName00.ext", "fileName01.ext",
                                          "fileName02.ext", "fileName03.ext"};

    bool status;

    status = utility::filesystem::MakeDirectoryHierarchy(path);
    EXPECT_TRUE(status);

    status = utility::filesystem::ChangeWorkingDirectory(path);
    EXPECT_TRUE(status);

    for (size_t i = 0; i < fileNames.size(); i++)
        creat(fileNames[i].c_str(), 0);

    std::vector<std::string> list;
    status = utility::filesystem::ListFilesInDirectory(".", list);
    EXPECT_TRUE(status);

    sort(list.begin(), list.end());

    for (size_t i = 0; i < fileNames.size(); i++) {
        EXPECT_EQ(fileNames[i],
                  utility::filesystem::GetFileNameWithoutDirectory(list[i]));
    }

    // clean-up
    for (size_t i = 0; i < fileNames.size(); i++) {
        status = utility::filesystem::RemoveFile(fileNames[i]);
        EXPECT_TRUE(status);
    }

    // clean-up in reverse order, DeleteDirectory can delete one dir at a time.
    status = utility::filesystem::ChangeWorkingDirectory("..");
    EXPECT_TRUE(status);

    status = utility::filesystem::DeleteDirectory("test_dir");
    EXPECT_TRUE(status);

    status = utility::filesystem::ChangeWorkingDirectory("..");
    EXPECT_TRUE(status);

    status = utility::filesystem::DeleteDirectory("test");
    EXPECT_TRUE(status);
}

// ----------------------------------------------------------------------------
// List all files of a specific extension in the specified directory.
// ----------------------------------------------------------------------------
TEST(FileSystem, ListFilesInDirectoryWithExtension) {
    std::string path = "test/test_dir";
    std::vector<std::string> fileNames = {"fileName00.ext0", "fileName01.ext0",
                                          "fileName02.ext0", "fileName03.ext0",
                                          "fileName04.ext1", "fileName05.ext1",
                                          "fileName06.ext1", "fileName07.ext1"};

    bool status;

    status = utility::filesystem::MakeDirectoryHierarchy(path);
    EXPECT_TRUE(status);

    status = utility::filesystem::ChangeWorkingDirectory(path);
    EXPECT_TRUE(status);

    for (size_t i = 0; i < fileNames.size(); i++)
        creat(fileNames[i].c_str(), 0);

    std::vector<std::string> list;
    status = utility::filesystem::ListFilesInDirectory(".", list);
    EXPECT_TRUE(status);

    sort(list.begin(), list.end());

    for (size_t i = 0; i < list.size(); i++) {
        EXPECT_EQ(fileNames[i],
                  utility::filesystem::GetFileNameWithoutDirectory(list[i]));
    }

    // clean-up
    for (size_t i = 0; i < fileNames.size(); i++) {
        status = utility::filesystem::RemoveFile(fileNames[i]);
        EXPECT_TRUE(status);
    }

    // clean-up in reverse order, DeleteDirectory can delete one dir at a time.
    status = utility::filesystem::ChangeWorkingDirectory("..");
    EXPECT_TRUE(status);

    status = utility::filesystem::DeleteDirectory("test_dir");
    EXPECT_TRUE(status);

    status = utility::filesystem::ChangeWorkingDirectory("..");
    EXPECT_TRUE(status);

    status = utility::filesystem::DeleteDirectory("test");
    EXPECT_TRUE(status);
}

// ----------------------------------------------------------------------------
// Split path into components
// ----------------------------------------------------------------------------
TEST(FileSystem, GetPathComponents) {
    // setup
    std::string cwd = utility::filesystem::GetWorkingDirectory();
    std::vector<std::string> cwd_components =
            utility::filesystem::GetPathComponents(cwd);
    if (cwd_components.size() < 2) {
        utility::LogError("Please do not run unit test from root directory.");
    }
    std::vector<std::string> parent_components(cwd_components.begin(),
                                               cwd_components.end() - 1);

    // test
    std::vector<std::string> expected;
    std::vector<std::string> result;

    result = utility::filesystem::GetPathComponents("");
    EXPECT_EQ(result, cwd_components);

    result = utility::filesystem::GetPathComponents("/");
    expected = {"/"};
    EXPECT_EQ(result, expected);

    result = utility::filesystem::GetPathComponents("c:\\");
    expected = {"c:"};
    EXPECT_EQ(result, expected);

    result = utility::filesystem::GetPathComponents("../bogus/test.abc");
    expected = parent_components;
    expected.push_back("bogus");
    expected.push_back("test.abc");
    EXPECT_EQ(result, expected);

    result = utility::filesystem::GetPathComponents("/usr/lib/../local/bin");
    expected = {"/", "usr", "local", "bin"};
    EXPECT_EQ(result, expected);

    result = utility::filesystem::GetPathComponents(
            "c:\\windows\\system\\winnt.dll");
    expected = {"c:", "windows", "system", "winnt.dll"};
    EXPECT_EQ(result, expected);
}

}  // namespace tests
}  // namespace cloudViewer
