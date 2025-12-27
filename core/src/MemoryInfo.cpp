// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "MemoryInfo.h"

#include <CVPlatform.h>

#include <cmath>
#include <iomanip>

#if defined(CV_WINDOWS)
#include <windows.h>

#include <sstream>
#elif defined(CV_LINUX)
#include <sys/sysinfo.h>

#include <fstream>
#include <limits>
#elif defined(CV_MAC_OS)
#include <mach/host_info.h>
#include <mach/mach_host.h>
#include <mach/mach_init.h>
#include <mach/mach_types.h>
#include <mach/vm_statistics.h>
#include <sys/sysctl.h>
#include <sys/types.h>

#include <sstream>
#else
#warning "System unrecognized. Can't found memory infos."
#include <limits>
#endif

namespace cloudViewer {
namespace system {

#if defined(CV_LINUX)
unsigned long linuxGetAvailableRam() {
    std::string token;
    std::ifstream file("/proc/meminfo");
    while (file >> token) {
        if (token == "MemAvailable:") {
            unsigned long mem;
            if (file >> mem) {
                // read in kB and convert to bytes
                return mem * 1024;
            } else {
                return 0;
            }
        }
        // ignore rest of the line
        file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }
    return 0;  // nothing found
}
#endif

MemoryInfo getMemoryInfo() {
    MemoryInfo infos;

#if defined(CV_WINDOWS)
    MEMORYSTATUS memory;
    GlobalMemoryStatus(&memory);

    // memory.dwMemoryLoad;
    infos.totalRam = memory.dwTotalPhys;
    infos.availableRam = infos.freeRam = memory.dwAvailPhys;
    // memory.dwTotalPageFile;
    // memory.dwAvailPageFile;
    infos.totalSwap = memory.dwTotalVirtual;
    infos.freeSwap = memory.dwAvailVirtual;
#elif defined(CV_LINUX)
    struct sysinfo sys_info;
    sysinfo(&sys_info);

    infos.totalRam = sys_info.totalram * sys_info.mem_unit;
    infos.freeRam = sys_info.freeram * sys_info.mem_unit;

    infos.availableRam = linuxGetAvailableRam();
    if (infos.availableRam == 0) infos.availableRam = infos.freeRam;

    // infos.sharedRam = sys_info.sharedram * sys_info.mem_unit;
    // infos.bufferRam = sys_info.bufferram * sys_info.mem_unit;
    infos.totalSwap = sys_info.totalswap * sys_info.mem_unit;
    infos.freeSwap = sys_info.freeswap * sys_info.mem_unit;
#elif defined(CV_MAC_OS)
    uint64_t physmem;
    size_t len = sizeof physmem;
    int mib[2] = {CTL_HW, HW_MEMSIZE};
    size_t miblen = sizeof(mib) / sizeof(mib[0]);

    // Total physical memory.
    if (sysctl(mib, miblen, &physmem, &len, NULL, 0) == 0 &&
        len == sizeof(physmem))
        infos.totalRam = physmem;

    // Virtual memory.
    mib[0] = CTL_VM;
    mib[1] = VM_SWAPUSAGE;
    struct xsw_usage swap;
    len = sizeof(struct xsw_usage);
    if (sysctl(mib, miblen, &swap, &len, NULL, 0) == 0) {
        infos.totalSwap = swap.xsu_total;
        infos.freeSwap = swap.xsu_avail;
    }

    // In use - Reference ParaView/VTK implementation for macOS memory
    // statistics Use 64-bit API for accurate memory statistics on modern macOS
    // systems
    mach_port_t stat_port = mach_host_self();
    vm_size_t page_size;
    vm_statistics64_data_t vm_stat;
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    if (KERN_SUCCESS == host_page_size(stat_port, &page_size) &&
        KERN_SUCCESS == host_statistics64(stat_port, HOST_VM_INFO64,
                                          (host_info64_t)&vm_stat, &count)) {
        // Free RAM: completely free memory pages
        infos.freeRam = (int64_t)vm_stat.free_count * (int64_t)page_size;

        // Available RAM: ParaView/VTK style calculation for macOS
        // ParaView uses "App Memory" concept but excludes purgeable and
        // compressed memory App Memory = active + wired - purgeable -
        // compressor This represents memory actively used by applications,
        // excluding reclaimable memory
        uint64_t app_memory =
                ((uint64_t)vm_stat.active_count + (uint64_t)vm_stat.wire_count -
                 (uint64_t)vm_stat.purgeable_count -
                 (uint64_t)vm_stat.compressor_page_count) *
                (uint64_t)page_size;
        infos.availableRam = infos.totalRam - app_memory;
    }
#else
    // TODO: could be done on FreeBSD too
    // see https://github.com/xbmc/xbmc/blob/master/xbmc/linux/XMemUtils.cpp
    infos.totalRam = infos.freeRam = infos.availableRam = infos.totalSwap =
            infos.freeSwap = std::numeric_limits<std::size_t>::max();
#endif

    return infos;
}

std::ostream& operator<<(std::ostream& os, const MemoryInfo& infos) {
    const double convertionGb = std::pow(2, 30);
    os << std::setw(5)
       << "\t- Total RAM:     " << (infos.totalRam / convertionGb) << " GB"
       << std::endl
       << "\t- Free RAM:      " << (infos.freeRam / convertionGb) << " GB"
       << std::endl
       << "\t- Available RAM: " << (infos.availableRam / convertionGb) << " GB"
       << std::endl
       << "\t- Total swap:    " << (infos.totalSwap / convertionGb) << " GB"
       << std::endl
       << "\t- Free swap:     " << (infos.freeSwap / convertionGb) << " GB"
       << std::endl;
    return os;
}

std::string MemoryInfo::toString() {
    std::stringstream info;
    info << *this;
    return info.str();
}

}  // namespace system
}  // namespace cloudViewer
