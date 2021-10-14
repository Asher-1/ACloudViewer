// ----------------------------------------------------------------------------
// -                        CloudViewer: asher-1.github.io                          -
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

#include "CloudViewer.h"

using namespace cloudViewer;

int main(int argc, char **argv) {
    // command-line parameters
    utility::SetVerbosityLevel(utility::VerbosityLevel::Debug);

    // print GPU Information
    utility::LogInfo("{}", gpu::gpuInformationCUDA().c_str());

    // print CPU Information
    system::MemoryInfo memoryInformation = system::getMemoryInfo();
    utility::LogInfo("hardware.cpu.freq {}", system::cpu_clock_by_os());        // cpu frequency
    utility::LogInfo("hardware.cpu.cores {}", system::get_total_cpus());        // cpu cores
    utility::LogInfo("hardware.ram.size {}", memoryInformation.totalRam); // ram size

    utility::LogInfo("Memory information:\n {}", memoryInformation.toString().c_str());

    if(memoryInformation.availableRam == 0)
    {
      utility::LogWarning("Cannot find available system memory, this can be due to OS limitation.\n"
                              "Use only one thread for CPU feature extraction.");
    }
    else
    {
        const double oneGB = 1024.0 * 1024.0 * 1024.0;
        if(memoryInformation.availableRam < 0.5 * memoryInformation.totalRam)
        {
            utility::LogWarning("More than half of the RAM is used by other applications. It would be more efficient to close them.");
            utility::LogWarning(" => {} GB are used by other applications for a total RAM capacity of {} GB.",
                                std::size_t(std::round(double(memoryInformation.totalRam - memoryInformation.availableRam) / oneGB)),
                                std::size_t(std::round(double(memoryInformation.totalRam) / oneGB)));
        }
    }

    return 0;
}
