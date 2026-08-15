// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "PcdExport.h"

#include <CVLog.h>

#include <cstdint>
#include <cstring>
#include <fstream>

#include "mcalib_portability.h"

namespace mcalib {

bool writePcdBinaryXYZIRT(const std::string& path,
                          const std::vector<PointXYZIRT>& points,
                          const std::string& frame_id) {
    std::ofstream out;
    if (!openOutputFile(out, path)) {
        CVLog::Error("[PcdExport] cannot write %s", path.c_str());
        return false;
    }

    out << "# .PCD v0.7 - Point Cloud Data file format\n";
    out << "VERSION 0.7\n";
    out << "FIELDS x y z intensity ring timestamp\n";
    out << "SIZE 4 4 4 1 1 2\n";
    out << "TYPE F F F U U U\n";
    out << "COUNT 1 1 1 1 1 1\n";
    out << "WIDTH " << points.size() << "\n";
    out << "HEIGHT 1\n";
    out << "VIEWPOINT 0 0 0 1 0 0 0\n";
    out << "POINTS " << points.size() << "\n";
    out << "DATA binary\n";

    for (const auto& pt : points) {
        out.write(reinterpret_cast<const char*>(&pt.x), sizeof(float));
        out.write(reinterpret_cast<const char*>(&pt.y), sizeof(float));
        out.write(reinterpret_cast<const char*>(&pt.z), sizeof(float));
        out.write(reinterpret_cast<const char*>(&pt.intensity),
                  sizeof(uint8_t));
        out.write(reinterpret_cast<const char*>(&pt.ring), sizeof(uint8_t));
        out.write(reinterpret_cast<const char*>(&pt.timestamp),
                  sizeof(uint16_t));
    }

    (void)frame_id;
    return out.good();
}

bool writePcdBinaryXYZIRGB(const std::string& path,
                           const std::vector<PointXYZIRT>& points,
                           const std::vector<uint32_t>& rgb_packed,
                           const std::string& frame_id) {
    if (rgb_packed.size() != points.size()) {
        CVLog::Error("[PcdExport] rgb_packed size mismatch: %zu vs %zu",
                     rgb_packed.size(), points.size());
        return false;
    }
    std::ofstream out;
    if (!openOutputFile(out, path)) {
        CVLog::Error("[PcdExport] cannot write %s", path.c_str());
        return false;
    }

    // PCL convention: rgb is a single float (4 bytes) with the low 24 bits
    // holding r<<16 | g<<8 | b. CloudCompare/PCL/rviz all read it this way.
    out << "# .PCD v0.7 - Point Cloud Data file format\n";
    out << "VERSION 0.7\n";
    out << "FIELDS x y z intensity ring timestamp rgb\n";
    out << "SIZE 4 4 4 1 1 2 4\n";
    out << "TYPE F F F U U U F\n";
    out << "COUNT 1 1 1 1 1 1 1\n";
    out << "WIDTH " << points.size() << "\n";
    out << "HEIGHT 1\n";
    out << "VIEWPOINT 0 0 0 1 0 0 0\n";
    out << "POINTS " << points.size() << "\n";
    out << "DATA binary\n";

    for (size_t i = 0; i < points.size(); ++i) {
        const auto& pt = points[i];
        out.write(reinterpret_cast<const char*>(&pt.x), sizeof(float));
        out.write(reinterpret_cast<const char*>(&pt.y), sizeof(float));
        out.write(reinterpret_cast<const char*>(&pt.z), sizeof(float));
        out.write(reinterpret_cast<const char*>(&pt.intensity),
                  sizeof(uint8_t));
        out.write(reinterpret_cast<const char*>(&pt.ring), sizeof(uint8_t));
        out.write(reinterpret_cast<const char*>(&pt.timestamp),
                  sizeof(uint16_t));
        // Reinterpret packed uint32 as float — PCL reads rgb as float.
        float rgb_float;
        std::memcpy(&rgb_float, &rgb_packed[i], sizeof(float));
        out.write(reinterpret_cast<const char*>(&rgb_float), sizeof(float));
    }

    (void)frame_id;
    return out.good();
}

}  // namespace mcalib
