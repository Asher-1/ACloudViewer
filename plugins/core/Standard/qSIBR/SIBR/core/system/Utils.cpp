// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "core/system/Utils.hpp"

#include <tinyfiledialogs/tinyfiledialogs.h>

#include <boost/filesystem.hpp>
#include <fstream>
#include <sstream>
#include <vector>

#ifdef SIBR_OS_WINDOWS
#include <Windows.h>
#include <shlobj.h>
#include <stdio.h>
#ifndef ENABLE_VIRTUAL_TERMINAL_PROCESSING
#define ENABLE_VIRTUAL_TERMINAL_PROCESSING 0x0004
#endif
#else
#include <libgen.h>
#include <limits.h>
#include <pwd.h>
#include <sys/types.h>
#include <unistd.h>
#endif

namespace sibr {
#ifdef SIBR_OS_WINDOWS
static HANDLE stdoutHandle;
static DWORD outModeInit;

void setupConsole(void) {
    DWORD outMode = 0;
    stdoutHandle = GetStdHandle(STD_OUTPUT_HANDLE);

    if (stdoutHandle == INVALID_HANDLE_VALUE) {
        exit(GetLastError());
    }

    if (!GetConsoleMode(stdoutHandle, &outMode)) {
        exit(GetLastError());
    }

    outModeInit = outMode;

    // Enable ANSI escape codes
    outMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;

    if (!SetConsoleMode(stdoutHandle, outMode)) {
        exit(GetLastError());
    }
}

void restoreConsole(void) {
    // Reset colors
    printf("\x1b[0m");

    // Reset console mode
    if (!SetConsoleMode(stdoutHandle, outModeInit)) {
        exit(GetLastError());
    }
}
#endif

std::string loadFile(const std::string& fname) {
    std::ifstream file(fname.c_str(), std::ios::binary);
    if (!file || !file.is_open()) {
        SIBR_ERR << "File not found: " << fname << std::endl;
        return "";
    }
    file.seekg(0, std::ios::end);

    std::streampos length = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(length);
    file.read(&buffer[0], length);
    file.close();

    return std::string(buffer.begin(), buffer.end());
}

void makeDirectory(const std::string& path) {
    boost::filesystem::path p(path);
    if (boost::filesystem::exists(p) == false)
        boost::filesystem::create_directories(p);
}

std::vector<std::string> listFiles(
        const std::string& path,
        const bool listHidden,
        const bool includeSubdirectories,
        const std::vector<std::string>& allowedExtensions) {
    if (!directoryExists(path)) {
        return {};
    }

    std::vector<std::string> files;
    bool shouldCheckExtension = !allowedExtensions.empty();

    try {
        boost::filesystem::directory_iterator end_iter;
        for (boost::filesystem::directory_iterator dir_itr(path);
             dir_itr != end_iter; ++dir_itr) {
            const std::string itemName = dir_itr->path().filename().string();
            if (includeSubdirectories &&
                boost::filesystem::is_directory(dir_itr->status())) {
                if (listHidden ||
                    (itemName.size() > 0 && itemName.at(0) != '.')) {
                    files.push_back(itemName);
                }
            } else if (boost::filesystem::is_regular_file(dir_itr->status())) {
                bool shouldKeep = !shouldCheckExtension;
                if (shouldCheckExtension) {
                    for (const auto& allowedExtension : allowedExtensions) {
                        if (dir_itr->path().extension() ==
                                    ("." + allowedExtension) ||
                            dir_itr->path().extension() == allowedExtension) {
                            shouldKeep = true;
                            break;
                        }
                    }
                }

                if (shouldKeep && (listHidden || (itemName.size() > 0 &&
                                                  itemName.at(0) != '.'))) {
                    files.push_back(itemName);
                }
            }
        }
    } catch (const boost::filesystem::filesystem_error&) {
        std::cout << "Can't access or find directory." << std::endl;
    }

    std::sort(files.begin(), files.end());

    return files;
}

std::vector<std::string> listSubdirectories(const std::string& path,
                                            const bool listHidden) {
    if (!directoryExists(path)) {
        return {};
    }

    std::vector<std::string> dirs;

    try {
        boost::filesystem::directory_iterator end_iter;
        for (boost::filesystem::directory_iterator dir_itr(path);
             dir_itr != end_iter; ++dir_itr) {
            const std::string itemName = dir_itr->path().filename().string();
            if (boost::filesystem::is_directory(dir_itr->status())) {
                if (listHidden ||
                    (itemName.size() > 0 && itemName.at(0) != '.')) {
                    dirs.push_back(itemName);
                }
            }
        }
    } catch (const boost::filesystem::filesystem_error&) {
        std::cout << "Can't access or find directory." << std::endl;
    }

    std::sort(dirs.begin(), dirs.end());

    return dirs;
}

bool copyDirectory(const std::string& src, const std::string& dst) {
    boost::filesystem::path source = src;
    boost::filesystem::path destination = dst;
    namespace fs = boost::filesystem;
    try {
        // Check whether the function call is valid
        if (!fs::exists(source) || !fs::is_directory(source)) {
            std::cerr << "Source directory " << source.string()
                      << " does not exist or is not a directory." << '\n';
            return false;
        }
        if (fs::exists(destination)) {
            std::cerr << "Destination directory " << destination.string()
                      << " already exists." << '\n';
            return false;
        }
        // Create the destination directory
        if (!fs::create_directory(destination)) {
            std::cerr << "Unable to create destination directory"
                      << destination.string() << '\n';
            return false;
        }
    } catch (fs::filesystem_error const& e) {
        std::cerr << e.what() << '\n';
        return false;
    }
    // Iterate through the source directory
    for (fs::directory_iterator file(source); file != fs::directory_iterator();
         ++file) {
        try {
            fs::path current(file->path());
            if (fs::is_directory(current)) {
                // Found directory: Recursion
                if (!copyDirectory(
                            current.string(),
                            (destination / current.filename()).string())) {
                    return false;
                }
            } else {
                // Found file: Copy
                fs::copy_file(current, destination / current.filename());
            }
        } catch (fs::filesystem_error const& e) {
            std::cerr << e.what() << '\n';
        }
    }
    return true;
}

bool copyFile(const std::string& src,
              const std::string& dst,
              const bool overwrite) {
    boost::filesystem::path source = src;
    boost::filesystem::path destination = dst;
    namespace fs = boost::filesystem;
    try {
        // Check whether the function call is valid
        if (!fs::exists(source) || !fs::is_regular_file(source)) {
            std::cerr << "Source file " << source.string()
                      << " does not exist or is not a regular file." << '\n';
            return false;
        }

        // If the destination is a directory, we copy the file into this
        // directory, with the same name.
        if (fs::is_directory(destination)) {
            destination = destination / source.filename();
        }

        if (fs::exists(destination) && !overwrite) {
            std::cerr << "Destination file " << destination.string()
                      << " already exists." << '\n';
            return false;
        }
        if (overwrite) {
            fs::copy_file(source, destination,
                          boost::filesystem::copy_option::overwrite_if_exists);
        } else {
            fs::copy_file(source, destination);
        }

    } catch (fs::filesystem_error const& e) {
        std::cerr << e.what() << '\n';
        return false;
    }

    return true;
}

void emptyDirectory(const std::string& path) {
    boost::filesystem::path p(path);
    for (boost::filesystem::directory_iterator end_dir_it, it(p);
         it != end_dir_it; ++it) {
        boost::filesystem::remove_all(it->path());
    }
}

bool fileExists(const std::string& path) {
    boost::filesystem::path p(path);
    return boost::filesystem::exists(p) &&
           boost::filesystem::is_regular_file(path);
}

bool directoryExists(const std::string& path) {
    boost::filesystem::path p(path);
    return boost::filesystem::exists(p) &&
           boost::filesystem::is_directory(path);
}

size_t getAvailableMem() {
#define DIV 1024

#ifdef SIBR_OS_WINDOWS
    MEMORYSTATUSEX statex;
    statex.dwLength = sizeof(statex);
    GlobalMemoryStatusEx(&statex);
    return static_cast<size_t>(statex.ullAvailPhys) / DIV;
#else
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    return static_cast<size_t>(pages * page_size) / DIV;
#endif
}

namespace {

std::string queryExecutableDirectory() {
#ifdef SIBR_OS_WINDOWS
    char exePath[MAX_PATH];
    unsigned int len =
            GetModuleFileNameA(GetModuleHandleA(0x0), exePath, MAX_PATH);
    if (len == 0 || len >= MAX_PATH) {
        SIBR_ERR << "Can't find install folder! Please specify as command-line "
                    "option using --appPath option!"
                 << std::endl;
        return "";
    }
    return parentDirectory(exePath);
#else
    char result[PATH_MAX];
    ssize_t c = readlink("/proc/self/exe", result, PATH_MAX - 1);
    if (c <= 0) {
        SIBR_ERR << "Can't find executable path. Please specify as "
                    "command-line option using --appPath option!"
                 << std::endl;
        return "";
    }
    result[c] = '\0';
    return std::string(dirname(result));
#endif
}

/** True if \p root contains the SIBR tree copied by qSIBR CMake (see
 * SIBR_BIN_DIR). */
bool looksLikeSibrAssetRoot(const std::string& root) {
    if (root.empty()) return false;
    return directoryExists(root + "/sibr_resources") ||
           fileExists(root + "/ibr_resources.ini");
}

}  // namespace

SIBR_SYSTEM_EXPORT std::string getInstallDirectory() {
    // ACloudViewer copies shaders/, sibr_resources/, ibr_resources.ini under
    // CMAKE_BINARY_DIR/bin (SIBR_BIN_DIR). With VS/Xcode multi-config, the
    // executable lives in bin/Release or bin/Debug, so the asset root is the
    // parent of the exe directory. Fall back to exe dir when assets sit there.

    const std::string exeDir = queryExecutableDirectory();
    if (exeDir.empty()) return "";

    if (looksLikeSibrAssetRoot(exeDir)) return exeDir;

    const std::string parentOfExe = parentDirectory(exeDir);
    if (!parentOfExe.empty() && parentOfExe != exeDir &&
        looksLikeSibrAssetRoot(parentOfExe))
        return parentOfExe;

    return exeDir;
}

SIBR_SYSTEM_EXPORT std::string getBinDirectory() {
    return getInstallSubDirectory("bin");
}

SIBR_SYSTEM_EXPORT std::string getShadersDirectory(
        const std::string& subfolder) {
    return getInstallSubDirectory("shaders" +
                                  ((subfolder != "") ? "/" + subfolder : ""));
}

SIBR_SYSTEM_EXPORT std::string getScriptsDirectory() {
    return getInstallSubDirectory("scripts");
}

SIBR_SYSTEM_EXPORT std::string getResourcesDirectory() {
    return getInstallSubDirectory("sibr_resources");
}

SIBR_SYSTEM_EXPORT std::string getAppDataDirectory() {
    std::string appDataDirectory = "";
#ifdef SIBR_OS_WINDOWS
    PWSTR path_tmp;

    /* Attempt to get user's AppData folder
     *
     * Microsoft Docs:
     * https://docs.microsoft.com/en-us/windows/win32/api/shlobj_core/nf-shlobj_core-shgetknownfolderpath
     * https://docs.microsoft.com/en-us/windows/win32/shell/knownfolderid
     */
    auto get_folder_path_ret = SHGetKnownFolderPath(FOLDERID_RoamingAppData, 0,
                                                    nullptr, &path_tmp);

    /* Error check */
    if (get_folder_path_ret != S_OK) {
        CoTaskMemFree(path_tmp);
        SIBR_ERR << "Could not access AppData folder.";
    }

    std::wstring path_wtmp(path_tmp);
    appDataDirectory += std::string(path_wtmp.begin(), path_wtmp.end());
    appDataDirectory += "\\sibr";
    CoTaskMemFree(path_tmp);
#else
    struct passwd* pw = getpwuid(getuid());
    appDataDirectory += pw->pw_dir + std::string("/.sibr");
#endif

    makeDirectory(appDataDirectory);

    return appDataDirectory;
}

SIBR_SYSTEM_EXPORT std::string getInstallSubDirectory(
        const std::string& subfolder) {
    std::string installDirectory = getInstallDirectory();
    std::string installSubDirectory = installDirectory + "/" + subfolder;

    if (!directoryExists(installSubDirectory)) {
        // try subdirs GD LINUX issue
        installSubDirectory = installDirectory + "/install/" + subfolder;
        if (!directoryExists(installSubDirectory))
            SIBR_ERR << "Can't find subfolder " << subfolder << " in "
                     << installDirectory
                     << ". Please specify correct app folder as command-line "
                        "option using --appPath option!"
                     << std::endl;
    }

    return installSubDirectory;
}

bool showFilePicker(std::string& selectedElement,

                    const FilePickerMode mode,
                    const std::string& directoryPath,
                    const std::string& extensionsAllowed) {
    const char* result = NULL;

    std::vector<const char*> filterPatterns;
    std::vector<std::string> patternStorage;
    if (!extensionsAllowed.empty()) {
        std::stringstream ss(extensionsAllowed);
        std::string ext;
        while (std::getline(ss, ext, ',')) {
            if (!ext.empty()) {
                patternStorage.push_back("*." + ext);
            }
        }
        for (const auto& p : patternStorage) {
            filterPatterns.push_back(p.c_str());
        }
    }

    const char* defaultPath =
            directoryPath.empty() ? "" : directoryPath.c_str();
    int numFilters = static_cast<int>(filterPatterns.size());
    const char* const* filters = numFilters > 0 ? filterPatterns.data() : NULL;

    if (mode == Directory) {
        result = tinyfd_selectFolderDialog("Select Folder", defaultPath);
    } else if (mode == Save) {
        result = tinyfd_saveFileDialog("Save File", defaultPath, numFilters,
                                       filters, NULL);
    } else {
        result = tinyfd_openFileDialog("Open File", defaultPath, numFilters,
                                       filters, NULL, 0);
    }

    if (result != NULL) {
        selectedElement = std::string(result);
        return true;
    }

    return false;
}

SIBR_SYSTEM_EXPORT std::istream& safeGetline(std::istream& is, std::string& t) {
#ifdef SIBR_OS_WINDOWS
    return std::getline(is, t);
#else
    t.clear();

    // The characters in the stream are read one-by-one using a std::streambuf.
    // That is faster than reading them one-by-one using the std::istream.
    // Code that uses streambuf this way must be guarded by a sentry object.
    // The sentry object performs various tasks,
    // such as thread synchronization and updating the stream state.

    std::istream::sentry se(is, true);
    std::streambuf* sb = is.rdbuf();

    for (;;) {
        int c = sb->sbumpc();
        switch (c) {
            case '\n':
                return is;
            case '\r':
                if (sb->sgetc() == '\n') sb->sbumpc();
                return is;
            case std::streambuf::traits_type::eof():
                // Also handle the case when the last line has no line ending
                is.setstate(std::ios::eofbit);
                // this helps ignore the last line if it's empty (otherwise it's
                // a different behavior from std::get_line)
                if (t.empty()) is.setstate(std::ios::badbit);
                return is;
            default:
                t += (char)c;
        }
    }
#endif
}

}  // namespace sibr
