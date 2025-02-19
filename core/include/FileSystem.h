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

#ifndef CV_FILESYSTEM_HEADER
#define CV_FILESYSTEM_HEADER

#include "CVCoreLib.h"
#include "Helper.h"
#include <functional>
#include <string>
#include <vector>

namespace cloudViewer {
namespace utility {
namespace filesystem {

// Join multiple paths into one path.
template <typename... T>
std::string JoinPaths(T const &...paths) {
    std::string result;
    if (sizeof...(paths) == 1) {
        int unpack[]{0, (result = std::string(paths), 0)...};
        static_cast<void>(unpack);
        return result;
    } else {
        int unpack[]{0,
                     (result = result.empty() ? std::string(paths)
                                              : result + "/" + std::string(paths),
                      0)...};
        static_cast<void>(unpack);
    }

    if (StringContains(result, "///")) {
        result = StringReplace(result, "///", "/");
    }
    if (StringContains(result, "//")) {
        result = StringReplace(result, "//", "/");
    }
    if (StringContains(result, "\\\\")) {
        result = StringReplace(result, "\\\\", "/");
    }
    if (StringContains(result, "\\")) {
        result = StringReplace(result, "\\", "/");
    }

    if (StringEndsWith(result, "//")) {
        result = StringReplaceLast(result, "//", "");
    } else if (StringEndsWith(result, "/")) {
        result = StringReplaceLast(result, "/", "");
    }
    return result;
}

std::string CV_CORE_LIB_API GetEnvVar(const std::string &env_var);

// Append trailing slash to string if it does not yet end with a slash.
std::string CV_CORE_LIB_API EnsureTrailingSlash(const std::string &str);

// Check whether file name has the file extension (case insensitive).
bool CV_CORE_LIB_API HasFileExtension(const std::string &file_name, const std::string &ext);

// Split the path into its root and extension, for example,
// "dir/file.jpg" into "dir/file" and ".jpg".
void CV_CORE_LIB_API SplitFileExtension(const std::string &path, std::string *root,
                        std::string *ext);

/**
 * @brief Copy a file.
 * @param from The file path to copy from.
 * @param to The file path to copy to.
 * @return If the action is successful.
 */
bool CV_CORE_LIB_API CopyFile(const std::string &from, const std::string &to);

/**
 * @brief Copy a directory.
 * @param from The path to copy from [Note: not including 'from' folder].
 * @param to The path to copy to.
 * @return If the action is successful.
 */
bool CV_CORE_LIB_API CopyDir(const std::string &from, const std::string &to);

/**
 * @brief Copy a file or directory.
 * @param from The path to copy from.
 * @param to The path to copy to.
 * @param include_parent_dir Whether copy parent directory or not.
 * @return If the action is successful.
 */
bool CV_CORE_LIB_API Copy(const std::string &from, const std::string &to,
          bool include_parent_dir = false, const std::string &extname = "");

std::string CV_CORE_LIB_API GetFileExtensionInLowerCase(const std::string &filename);

std::string CV_CORE_LIB_API GetFileNameWithoutExtension(const std::string &filename);

std::string CV_CORE_LIB_API GetFileNameWithoutDirectory(const std::string &filename);

std::string CV_CORE_LIB_API GetFileParentDirectory(const std::string &filename);

std::string CV_CORE_LIB_API GetRegularizedDirectoryName(const std::string &directory);

std::string CV_CORE_LIB_API GetFileBaseName(const std::string &filename);

std::string CV_CORE_LIB_API GetWorkingDirectory();

std::vector<std::string> CV_CORE_LIB_API GetPathComponents(const std::string& path);

bool CV_CORE_LIB_API ChangeWorkingDirectory(const std::string &directory);

bool CV_CORE_LIB_API IsFile(const std::string &filename);

bool CV_CORE_LIB_API IsDirectory(const std::string &directory);

bool CV_CORE_LIB_API DirectoryExists(const std::string &directory);

/**
 * @brief Check if a specified directory specified by directory_path exists.
 *        If not, recursively create the directory (and its parents).
 * @param directory_path Directory path.
 * @return If the directory does exist or its creation is successful.
 */
bool CV_CORE_LIB_API EnsureDirectory(const std::string &directory_path);

bool CV_CORE_LIB_API MakeDirectory(const std::string &directory);

bool CV_CORE_LIB_API MakeDirectoryHierarchy(const std::string &directory);

bool CV_CORE_LIB_API DeleteDirectory(const std::string &directory);

bool CV_CORE_LIB_API FileExists(const std::string &filename);

bool CV_CORE_LIB_API RemoveFile(const std::string &filename);

bool CV_CORE_LIB_API ListDirectory(const std::string& directory,
                                   std::vector<std::string>& subdirs,
                                   std::vector<std::string>& filenames);

bool CV_CORE_LIB_API ListFilesInDirectory(const std::string &directory,
                                          std::vector<std::string> &filenames);

bool CV_CORE_LIB_API ListFilesInDirectoryWithExtension(const std::string &directory,
                                                       const std::string &extname,
                                                       std::vector<std::string> &filenames);

CV_CORE_LIB_API std::vector<std::string> FindFilesRecursively(
                            const std::string &directory,
                            std::function<bool(const std::string &)> is_match);

// wrapper for fopen that enables unicode paths on Windows
CV_CORE_LIB_API FILE* FOpen(const std::string &filename, const std::string &mode);
std::string CV_CORE_LIB_API GetIOErrorString(const int errnoVal);
bool CV_CORE_LIB_API FReadToBuffer(const std::string& path,
    std::vector<char>& bytes,
    std::string* errorStr);

/// RAII Wrapper for C FILE*
/// Throws exceptions in situations where the caller is not usually going to
/// have proper handling code:
/// - using an unopened CFile
/// - errors and ferror from underlying calls (fclose, ftell, fseek, fread,
///   fwrite, fgetpos, fsetpos)
/// - reading a line that is too long, caller is unlikely to have proper code to
///   handle a partial next line
/// If you would like to handle any of these issues by not having your code
/// throw, use try/catch (const std::exception &e) { ... }
class CV_CORE_LIB_API CFile {
public:
    /// The destructor closes the file automatically.
    ~CFile();

    /// Open a file.
    bool Open(const std::string &filename, const std::string &mode);

    /// Returns the last encountered error for this file.
    std::string GetError();

    /// Close the file.
    void Close();

    /// Returns current position in the file (ftell).
    int64_t CurPos();

    /// Returns the file size in bytes.
    int64_t GetFileSize();

    /// Returns the number of lines in the file.
    int64_t GetNumLines();

    /// Throws if we hit buffer maximum. In most cases, calling code is only
    /// capable of processing a complete line, if it receives a partial line it
    /// will probably fail and it is very likely to fail/corrupt on the next
    /// call that receives the remainder of the line.
    const char *ReadLine();

    /// Read data to a buffer.
    /// \param data The data buffer to be written into.
    /// \param num_elements Number of elements to be read. The byte size of the
    /// element is determined by the size of buffer type.
    template <class T>
    size_t ReadData(T *data, size_t num_elems) {
        return ReadData(data, sizeof(T), num_elems);
    }

    /// Read data to a buffer.
    /// \param data The data buffer to be written into.
    /// \param elem_size Element size in bytes.
    /// \param num_elems Number of elements to read.
    size_t ReadData(void *data, size_t elem_size, size_t num_elems);

    /// Returns the underlying C FILE pointer.
    FILE *GetFILE() { return file_; }

private:
    FILE *file_ = nullptr;
    int error_code_ = 0;
    std::vector<char> line_buffer_;
};

}  // namespace filesystem
}  // namespace utility
}  // namespace cloudViewer

#endif // CV_FILESYSTEM_HEADER
