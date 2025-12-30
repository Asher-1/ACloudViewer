// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "CPUInfo.h"

#include <fstream>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <cstring>

#ifdef __linux__
#include <sys/sysinfo.h>
#elif __APPLE__
#include <sys/sysctl.h>
#include <sys/types.h>
#elif _WIN32
#include <Windows.h>
#include <intrin.h>
#endif

#include <Helper.h>
#include <Logging.h>

namespace cloudViewer {
namespace utility {

/// Returns the number of physical CPU cores.
static int PhysicalConcurrency() {
    try {
#ifdef __linux__
        // Ref: boost::thread::physical_concurrency().
        std::ifstream proc_cpuinfo("/proc/cpuinfo");
        const std::string physical_id("physical id");
        const std::string core_id("core id");

        // [physical ID, core id]
        using CoreEntry = std::pair<int, int>;
        std::set<CoreEntry> cores;
        CoreEntry current_core_entry;

        std::string line;
        while (std::getline(proc_cpuinfo, line)) {
            if (line.empty()) {
                continue;
            }
            std::vector<std::string> key_val =
                    utility::SplitString(line, ":", /*trim_empty_str=*/false);
            if (key_val.size() != 2) {
                return std::thread::hardware_concurrency();
            }
            std::string key = utility::StripString(key_val[0]);
            std::string value = utility::StripString(key_val[1]);
            if (key == physical_id) {
                current_core_entry.first = std::stoi(value);
                continue;
            }
            if (key == core_id) {
                current_core_entry.second = std::stoi(value);
                cores.insert(current_core_entry);
                continue;
            }
        }
        // Fall back to hardware_concurrency() in case
        // /proc/cpuinfo is formatted differently than we expect.
        return cores.size() != 0 ? cores.size()
                                 : std::thread::hardware_concurrency();
#elif __APPLE__
        // Ref: boost::thread::physical_concurrency().
        int count;
        size_t size = sizeof(count);
        return sysctlbyname("hw.physicalcpu", &count, &size, NULL, 0) ? 0
                                                                      : count;
#elif _WIN32
        // Ref: https://stackoverflow.com/a/44247223/1255535.
        DWORD length = 0;
        const BOOL result_first = GetLogicalProcessorInformationEx(
                RelationProcessorCore, nullptr, &length);
        assert(GetLastError() == ERROR_INSUFFICIENT_BUFFER);

        std::unique_ptr<uint8_t[]> buffer(new uint8_t[length]);
        const PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX info =
                reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(
                        buffer.get());
        const BOOL result_second = GetLogicalProcessorInformationEx(
                RelationProcessorCore, info, &length);
        assert(result_second != FALSE);

        int nb_physical_cores = 0;
        int offset = 0;
        do {
            const PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX current_info =
                    reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(
                            buffer.get() + offset);
            offset += current_info->Size;
            ++nb_physical_cores;
        } while (offset < static_cast<int>(length));
        return nb_physical_cores;
#else
        return std::thread::hardware_concurrency();
#endif
    } catch (...) {
        return std::thread::hardware_concurrency();
    }
}

/// Returns the CPU model name/brand string.
static std::string GetCPUModelName() {
#ifdef __linux__
    std::ifstream proc_cpuinfo("/proc/cpuinfo");
    std::string line;
    while (std::getline(proc_cpuinfo, line)) {
        if (line.find("model name") == 0) {
            size_t colon_pos = line.find(':');
            if (colon_pos != std::string::npos) {
                std::string model_name = line.substr(colon_pos + 1);
                return utility::StripString(model_name);
            }
        }
    }
    return "";
#elif __APPLE__
    char brand_string[256];
    size_t size = sizeof(brand_string);
    if (sysctlbyname("machdep.cpu.brand_string", brand_string, &size, NULL, 0) == 0) {
        return std::string(brand_string);
    }
    return "";
#elif _WIN32
    // Use CPUID instruction to get CPU brand string
    int cpuInfo[4] = {-1};
    char brand_string[0x40] = {0};
    
    // Get extended CPUID info
    __cpuid(cpuInfo, 0x80000000);
    unsigned int nExIds = cpuInfo[0];
    
    if (nExIds >= 0x80000004) {
        // Get brand string in 3 parts
        for (unsigned int i = 0x80000002; i <= 0x80000004; ++i) {
            __cpuid(cpuInfo, i);
            memcpy(brand_string + (i - 0x80000002) * 16, cpuInfo, sizeof(cpuInfo));
        }
        std::string result(brand_string);
        // Trim whitespace
        size_t first = result.find_first_not_of(" \t\n\r");
        size_t last = result.find_last_not_of(" \t\n\r");
        if (first != std::string::npos && last != std::string::npos) {
            return result.substr(first, last - first + 1);
        }
        return result;
    }
    return "";
#else
    return "";
#endif
}

CPUInfo::CPUInfo() : impl_(new CPUInfo::Impl()) {
    impl_->num_cores_ = PhysicalConcurrency();
    impl_->num_threads_ = std::thread::hardware_concurrency();
    impl_->model_name_ = GetCPUModelName();
}

CPUInfo& CPUInfo::GetInstance() {
    static CPUInfo instance;
    return instance;
}

int CPUInfo::NumCores() const { return impl_->num_cores_; }

int CPUInfo::NumThreads() const { return impl_->num_threads_; }

const std::string& CPUInfo::ModelName() const { return impl_->model_name_; }

void CPUInfo::Print() const {
    if (!impl_->model_name_.empty()) {
        utility::LogInfo("CPUInfo: {} ({} cores, {} threads).",
                         impl_->model_name_, NumCores(), NumThreads());
    } else {
        utility::LogInfo("CPUInfo: {} cores, {} threads.", NumCores(),
                         NumThreads());
    }
}

}  // namespace utility
}  // namespace cloudViewer
