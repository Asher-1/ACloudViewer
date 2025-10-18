// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "CloudViewer.h"

using namespace cloudViewer;

int main(int argc, char **argv) {
    // command-line parameters
    utility::SetVerbosityLevel(utility::VerbosityLevel::Debug);

    // print GPU Information
    utility::LogInfo("{}", gpu::gpuInformationCUDA().c_str());

    // print CPU Information
    system::MemoryInfo memoryInformation = system::getMemoryInfo();
    utility::LogInfo("hardware.cpu.freq {}",
                     system::cpu_clock_by_os());  // cpu frequency
    utility::LogInfo("hardware.cpu.cores {}",
                     system::get_total_cpus());  // cpu cores
    utility::LogInfo("hardware.ram.size {}",
                     memoryInformation.totalRam);  // ram size

    utility::LogInfo("Memory information:\n {}",
                     memoryInformation.toString().c_str());

    if (memoryInformation.availableRam == 0) {
        utility::LogWarning(
                "Cannot find available system memory, this can be due to OS "
                "limitation.\n"
                "Use only one thread for CPU feature extraction.");
    } else {
        const double oneGB = 1024.0 * 1024.0 * 1024.0;
        if (memoryInformation.availableRam < 0.5 * memoryInformation.totalRam) {
            utility::LogWarning(
                    "More than half of the RAM is used by other applications. "
                    "It would be more efficient to close them.");
            utility::LogWarning(
                    " => {} GB are used by other applications for a total RAM "
                    "capacity of {} GB.",
                    std::size_t(
                            std::round(double(memoryInformation.totalRam -
                                              memoryInformation.availableRam) /
                                       oneGB)),
                    std::size_t(std::round(double(memoryInformation.totalRam) /
                                           oneGB)));
        }
    }

    return 0;
}
