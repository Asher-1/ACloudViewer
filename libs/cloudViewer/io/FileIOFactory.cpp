// ----------------------------------------------------------------------------
// -                                  ECV_IO                            -
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

#include <Eigen.h>
#include <FileSystem.h>
#include <Helper.h>
#include <Logging.h>
#include <ProgressReporters.h>
#include <liblzf/lzf.h>
#include <rply.h>

#include <cstdint>
#include <cstdio>
#include <sstream>

#include "io/FileFormatIO.h"
#include "io/PointCloudIO.h"

// ECV_IO_LIB
#include <ecvMesh.h>
#include <ecvPointCloud.h>
#include <ecvScalarField.h>

// QT
#include <QFileInfo>

// References for PCD file IO
// http://pointclouds.org/documentation/tutorials/pcd_file_format.php
// https://github.com/PointCloudLibrary/pcl/blob/master/io/src/pcd_io.cpp
// https://www.mathworks.com/matlabcentral/fileexchange/40382-matlab-to-point-cloud-library

namespace cloudViewer {

namespace {
using namespace io;
using namespace cloudViewer;

namespace ply_pointcloud_reader {

struct PLYReaderState {
    utility::CountingProgressReporter *progress_bar;
    ccPointCloud *pointcloud_ptr;
    long vertex_index;
    long vertex_num;
    long normal_index;
    long normal_num;
    long color_index;
    long color_num;
    double normal[3];
};

int ReadVertexCallback(p_ply_argument argument) {
    PLYReaderState *state_ptr;
    long index;
    ply_get_argument_user_data(argument, reinterpret_cast<void **>(&state_ptr),
                               &index);
    if (state_ptr->vertex_index >= state_ptr->vertex_num) {
        return 0;  // some sanity check
    }

    double value = ply_get_argument_value(argument);
    CCVector3 &P = *state_ptr->pointcloud_ptr->getPointPtr(
            static_cast<size_t>(state_ptr->vertex_index));
    P(index) = static_cast<PointCoordinateType>(value);
    if (index == 2) {  // reading 'z'
        state_ptr->vertex_index++;
        ++(*state_ptr->progress_bar);
    }
    return 1;
}

int ReadNormalCallback(p_ply_argument argument) {
    PLYReaderState *state_ptr;
    long index;
    ply_get_argument_user_data(argument, reinterpret_cast<void **>(&state_ptr),
                               &index);
    if (state_ptr->normal_index >= state_ptr->normal_num) {
        return 0;
    }

    double value = ply_get_argument_value(argument);
    state_ptr->normal[index] = value;

    if (index == 2) {  // reading 'nz'
        state_ptr->pointcloud_ptr->addEigenNorm(
                Eigen::Vector3d(state_ptr->normal[0], state_ptr->normal[1],
                                state_ptr->normal[2]));
        state_ptr->normal_index++;
    }
    return 1;
}

int ReadColorCallback(p_ply_argument argument) {
    PLYReaderState *state_ptr;
    long index;
    ply_get_argument_user_data(argument, reinterpret_cast<void **>(&state_ptr),
                               &index);
    if (state_ptr->color_index >= state_ptr->color_num) {
        return 0;
    }

    double value = ply_get_argument_value(argument);
    ecvColor::Rgb &C = state_ptr->pointcloud_ptr->getPointColorPtr(
            static_cast<size_t>(state_ptr->color_index));
    C(index) = static_cast<ColorCompType>(value);

    if (index == 2) {  // reading 'blue'
        state_ptr->color_index++;
    }
    return 1;
}

}  // namespace ply_pointcloud_reader

/********************************* PCD UTILITY
 * *************************************/
enum PCDDataType {
    PCD_DATA_ASCII = 0,
    PCD_DATA_BINARY = 1,
    PCD_DATA_BINARY_COMPRESSED = 2
};

struct PCLPointField {
public:
    std::string name;
    int size;
    char type;
    int count;
    // helper variable
    int count_offset;
    int offset;
};

struct PCDHeader {
public:
    std::string version;
    std::vector<PCLPointField> fields;
    int width;
    int height;
    int points;
    PCDDataType datatype;
    std::string viewpoint;
    // helper variables
    int elementnum;
    int pointsize;
    bool has_points;
    bool has_normals;
    bool has_colors;
    bool has_ring;
    bool has_intensity;
};

bool CheckHeader(PCDHeader &header) {
    if (header.points <= 0 || header.pointsize <= 0) {
        utility::LogWarning("[CheckHeader] PCD has no data.");
        return false;
    }
    if (header.fields.size() == 0 || header.pointsize <= 0) {
        utility::LogWarning("[CheckHeader] PCD has no fields.");
        return false;
    }
    header.has_points = false;
    header.has_normals = false;
    header.has_colors = false;
    header.has_ring = false;
    header.has_intensity = false;
    bool has_x = false;
    bool has_y = false;
    bool has_z = false;
    bool has_normal_x = false;
    bool has_normal_y = false;
    bool has_normal_z = false;
    bool has_rgb = false;
    bool has_rgba = false;
    bool has_ring = false;
    bool has_intensity = false;
    for (const auto &field : header.fields) {
        if (field.name == "x") {
            has_x = true;
        } else if (field.name == "y") {
            has_y = true;
        } else if (field.name == "z") {
            has_z = true;
        } else if (field.name == "normal_x") {
            has_normal_x = true;
        } else if (field.name == "normal_y") {
            has_normal_y = true;
        } else if (field.name == "normal_z") {
            has_normal_z = true;
        } else if (field.name == "rgb") {
            has_rgb = true;
        } else if (field.name == "rgba") {
            has_rgba = true;
        } else if (field.name == "intensity") {
            has_intensity = true;
        } else if (field.name == "ring") {
            has_ring = true;
        }
    }
    header.has_points = (has_x && has_y && has_z);
    header.has_normals = (has_normal_x && has_normal_y && has_normal_z);
    header.has_colors = (has_rgb || has_rgba);
    header.has_intensity = has_intensity;
    header.has_ring = has_ring;
    if (header.has_points == false) {
        utility::LogWarning(
                "[CheckHeader] Fields for point data are not complete.");
        return false;
    }
    return true;
}

bool ReadPCDHeader(FILE *file, PCDHeader &header) {
    char line_buffer[DEFAULT_IO_BUFFER_SIZE];
    size_t specified_channel_count = 0;

    while (fgets(line_buffer, DEFAULT_IO_BUFFER_SIZE, file)) {
        std::string line(line_buffer);
        if (line == "") {
            continue;
        }
        std::vector<std::string> st;
        utility::SplitString(st, line, "\t\r\n ");
        std::stringstream sstream(line);
        sstream.imbue(std::locale::classic());
        std::string line_type;
        sstream >> line_type;
        if (line_type.substr(0, 1) == "#") {
        } else if (line_type.substr(0, 7) == "VERSION") {
            if (st.size() >= 2) {
                header.version = st[1];
            }
        } else if (line_type.substr(0, 6) == "FIELDS" ||
                   line_type.substr(0, 7) == "COLUMNS") {
            specified_channel_count = st.size() - 1;
            if (specified_channel_count == 0) {
                utility::LogWarning("[ReadPCDHeader] Bad PCD file format.");
                return false;
            }
            header.fields.resize(specified_channel_count);
            int count_offset = 0, offset = 0;
            for (size_t i = 0; i < specified_channel_count;
                 i++, count_offset += 1, offset += 4) {
                header.fields[i].name = st[i + 1];
                header.fields[i].size = 4;
                header.fields[i].type = 'F';
                header.fields[i].count = 1;
                header.fields[i].count_offset = count_offset;
                header.fields[i].offset = offset;
            }
            header.elementnum = count_offset;
            header.pointsize = offset;
        } else if (line_type.substr(0, 4) == "SIZE") {
            if (specified_channel_count != st.size() - 1) {
                utility::LogWarning("[ReadPCDHeader] Bad PCD file format.");
                return false;
            }
            int offset = 0, col_type = 0;
            for (size_t i = 0; i < specified_channel_count;
                 i++, offset += col_type) {
                sstream >> col_type;
                header.fields[i].size = col_type;
                header.fields[i].offset = offset;
            }
            header.pointsize = offset;
        } else if (line_type.substr(0, 4) == "TYPE") {
            if (specified_channel_count != st.size() - 1) {
                utility::LogWarning("[ReadPCDHeader] Bad PCD file format.");
                return false;
            }
            for (size_t i = 0; i < specified_channel_count; i++) {
                header.fields[i].type = st[i + 1].c_str()[0];
            }
        } else if (line_type.substr(0, 5) == "COUNT") {
            if (specified_channel_count != st.size() - 1) {
                utility::LogWarning("[ReadPCDHeader] Bad PCD file format.");
                return false;
            }
            int count_offset = 0, offset = 0, col_count = 0;
            for (size_t i = 0; i < specified_channel_count; i++) {
                sstream >> col_count;
                header.fields[i].count = col_count;
                header.fields[i].count_offset = count_offset;
                header.fields[i].offset = offset;
                count_offset += col_count;
                offset += col_count * header.fields[i].size;
            }
            header.elementnum = count_offset;
            header.pointsize = offset;
        } else if (line_type.substr(0, 5) == "WIDTH") {
            sstream >> header.width;
        } else if (line_type.substr(0, 6) == "HEIGHT") {
            sstream >> header.height;
            header.points = header.width * header.height;
        } else if (line_type.substr(0, 9) == "VIEWPOINT") {
            if (st.size() >= 2) {
                header.viewpoint = st[1];
            }
        } else if (line_type.substr(0, 6) == "POINTS") {
            sstream >> header.points;
        } else if (line_type.substr(0, 4) == "DATA") {
            header.datatype = PCD_DATA_ASCII;
            if (st.size() >= 2) {
                if (st[1].substr(0, 17) == "binary_compressed") {
                    header.datatype = PCD_DATA_BINARY_COMPRESSED;
                } else if (st[1].substr(0, 6) == "binary") {
                    header.datatype = PCD_DATA_BINARY;
                }
            }
            break;
        }
    }
    if (CheckHeader(header) == false) {
        return false;
    }
    return true;
}

PointCoordinateType UnpackBinaryPCDElement(const char *data_ptr,
                                           const char type,
                                           const int size) {
    const char type_uppercase = std::toupper(type, std::locale());
    if (type_uppercase == 'I') {
        if (size == 1) {
            std::int8_t data;
            memcpy(&data, data_ptr, sizeof(data));
            return static_cast<PointCoordinateType>(data);
        } else if (size == 2) {
            std::int16_t data;
            memcpy(&data, data_ptr, sizeof(data));
            return static_cast<PointCoordinateType>(data);
        } else if (size == 4) {
            std::int32_t data;
            memcpy(&data, data_ptr, sizeof(data));
            return static_cast<PointCoordinateType>(data);
        } else {
            return static_cast<PointCoordinateType>(0.0);
        }
    } else if (type_uppercase == 'U') {
        if (size == 1) {
            std::uint8_t data;
            memcpy(&data, data_ptr, sizeof(data));
            return static_cast<PointCoordinateType>(data);
        } else if (size == 2) {
            std::uint16_t data;
            memcpy(&data, data_ptr, sizeof(data));
            return static_cast<PointCoordinateType>(data);
        } else if (size == 4) {
            std::uint32_t data;
            memcpy(&data, data_ptr, sizeof(data));
            return static_cast<PointCoordinateType>(data);
        } else {
            return static_cast<PointCoordinateType>(0.0);
        }
    } else if (type_uppercase == 'F') {
        if (size == 4) {
            float data;
            memcpy(&data, data_ptr, sizeof(data));
            return static_cast<PointCoordinateType>(data);
        } else {
            return static_cast<PointCoordinateType>(0.0);
        }
    }
    return static_cast<PointCoordinateType>(0.0);
}

ecvColor::Rgb UnpackBinaryPCDColor(const char *data_ptr,
                                   const char type,
                                   const int size) {
    if (size == 4) {
        std::uint8_t data[4];
        memcpy(data, data_ptr, 4);
        // color data is packed in BGR order.
        return ecvColor::Rgb(data[2], data[1], data[0]);
    } else {
        return ecvColor::Rgb();
    }
}

ScalarType UnpackBinaryPCDScalar(const char *data_ptr,
                                 const char type,
                                 const int size) {
    if (size == 1) {
        std::uint8_t data;
        memcpy(&data, data_ptr, 1);
        ScalarType scalar = static_cast<ScalarType>(data);
        return scalar;
    } else {
        return static_cast<ScalarType>(0);
    }
}

PointCoordinateType UnpackASCIIPCDElement(const char *data_ptr,
                                          const char type,
                                          const int size) {
    char *end;
    const char type_uppercase = std::toupper(type, std::locale());
    if (type_uppercase == 'I') {
        return static_cast<PointCoordinateType>(std::strtol(data_ptr, &end, 0));
    } else if (type_uppercase == 'U') {
        return static_cast<PointCoordinateType>(
                std::strtoul(data_ptr, &end, 0));
    } else if (type_uppercase == 'F') {
        return static_cast<PointCoordinateType>(std::strtod(data_ptr, &end));
    }
    return static_cast<PointCoordinateType>(0.0);
}

ecvColor::Rgb UnpackASCIIPCDColor(const char *data_ptr,
                                  const char type,
                                  const int size) {
    if (size == 4) {
        std::uint8_t data[4] = {0, 0, 0, 0};
        char *end;
        const char type_uppercase = std::toupper(type, std::locale());
        if (type_uppercase == 'I') {
            std::int32_t value = std::strtol(data_ptr, &end, 0);
            memcpy(data, &value, 4);
        } else if (type_uppercase == 'U') {
            std::uint32_t value = std::strtoul(data_ptr, &end, 0);
            memcpy(data, &value, 4);
        } else if (type_uppercase == 'F') {
            float value = std::strtof(data_ptr, &end);
            memcpy(data, &value, 4);
        }
        return ecvColor::Rgb(data[2], data[1], data[0]);
    } else {
        return ecvColor::Rgb();
    }
}

ScalarType UnpackASCIIPCDScalar(const char *data_ptr,
                                const char type,
                                const int size) {
    if (size == 1) {
        std::uint8_t data;
        char *end;
        if (type == 'I') {
            std::int32_t value = std::strtol(data_ptr, &end, 0);
            memcpy(&data, &value, 1);
        } else if (type == 'U') {
            std::uint32_t value = std::strtoul(data_ptr, &end, 0);
            memcpy(&data, &value, 1);
        } else if (type == 'F') {
            float value = std::strtof(data_ptr, &end);
            memcpy(&data, &value, 1);
        }
        ScalarType scalar = static_cast<ScalarType>(data);
        return scalar;
    } else {
        return static_cast<ScalarType>(0);
    }
}

bool ReadPCDData(FILE *file,
                 const PCDHeader &header,
                 ccPointCloud &pointcloud,
                 const ReadPointCloudOption &params) {
    pointcloud.clear();
    // The header should have been checked
    if (header.has_points) {
        pointcloud.resize(static_cast<unsigned int>(header.points));
    } else {
        utility::LogWarning(
                "[ReadPCDData] Fields for point data are not complete.");
        return false;
    }
    if (header.has_normals) {
        pointcloud.reserveTheNormsTable();
    }
    if (header.has_colors) {
        pointcloud.resizeTheRGBTable();
    }

    // if the input field already exists...
    size_t pointCount = static_cast<size_t>(header.points);

    // create new scalar field
    ccScalarField *cc_intensity_field = nullptr;
    if (header.has_intensity) {
        int id = pointcloud.getScalarFieldIndexByName("intensity");
        if (id >= 0) {
            pointcloud.deleteScalarField(id);
        }

        cc_intensity_field = new ccScalarField("intensity");
        if (!cc_intensity_field->reserveSafe(
                    static_cast<unsigned>(pointCount))) {
            cc_intensity_field->release();
            return false;
        }
    }
    ccScalarField *cc_ring_field = nullptr;
    if (header.has_ring) {
        int id = pointcloud.getScalarFieldIndexByName("ring");
        if (id >= 0) {
            pointcloud.deleteScalarField(id);
        }
        cc_ring_field = new ccScalarField("ring");
        if (!cc_ring_field->reserveSafe(static_cast<unsigned>(pointCount))) {
            cc_ring_field->release();
            return false;
        }
    }

    utility::CountingProgressReporter reporter(params.update_progress);
    reporter.SetTotal(header.points);

    CCVector3 P(0, 0, 0);
    CCVector3 N(0, 0, 0);
    ecvColor::Rgb col;
    ScalarType ring;
    ScalarType intensity;

    if (header.datatype == PCD_DATA_ASCII) {
        char line_buffer[DEFAULT_IO_BUFFER_SIZE];
        unsigned int idx = 0;
        while (fgets(line_buffer, DEFAULT_IO_BUFFER_SIZE, file) &&
               idx < (unsigned int)header.points) {
            std::string line(line_buffer);
            std::vector<std::string> strs;
            utility::SplitString(strs, line, "\t\r\n ");
            if ((int)strs.size() < header.elementnum) {
                continue;
            }

            bool find_point = false;
            bool find_normal = false;
            bool find_color = false;
            bool find_ring = false;
            bool find_intensity = false;
            for (size_t i = 0; i < header.fields.size(); i++) {
                const auto &field = header.fields[i];
                if (field.name == "x") {
                    find_point = true;
                    P.x = UnpackASCIIPCDElement(
                            strs[field.count_offset].c_str(), field.type,
                            field.size);
                } else if (field.name == "y") {
                    find_point = true;
                    P.y = UnpackASCIIPCDElement(
                            strs[field.count_offset].c_str(), field.type,
                            field.size);
                } else if (field.name == "z") {
                    find_point = true;
                    P.z = UnpackASCIIPCDElement(
                            strs[field.count_offset].c_str(), field.type,
                            field.size);
                } else if (field.name == "normal_x") {
                    find_normal = true;
                    N.x = UnpackASCIIPCDElement(
                            strs[field.count_offset].c_str(), field.type,
                            field.size);
                } else if (field.name == "normal_y") {
                    find_normal = true;
                    N.y = UnpackASCIIPCDElement(
                            strs[field.count_offset].c_str(), field.type,
                            field.size);
                } else if (field.name == "normal_z") {
                    find_normal = true;
                    N.z = UnpackASCIIPCDElement(
                            strs[field.count_offset].c_str(), field.type,
                            field.size);
                } else if (field.name == "rgb" || field.name == "rgba") {
                    find_color = true;
                    col = UnpackASCIIPCDColor(strs[field.count_offset].c_str(),
                                              field.type, field.size);
                } else if (field.name == "intensity") {
                    find_intensity = true;
                    intensity = UnpackASCIIPCDScalar(
                            strs[field.count_offset].c_str(), field.type,
                            field.size);
                } else if (field.name == "ring") {
                    find_ring = true;
                    ring = UnpackASCIIPCDScalar(
                            strs[field.count_offset].c_str(), field.type,
                            field.size);
                }
            }

            if (find_point) {
                pointcloud.setPoint(idx, P);
            }
            if (header.has_normals && find_normal) {
                pointcloud.addNorm(N);
            }
            if (header.has_colors && find_color) {
                pointcloud.setPointColor(idx, col);
            }
            if (header.has_intensity && find_intensity) {
                cc_intensity_field->addElement(intensity);
            }
            if (header.has_ring && find_ring) {
                cc_ring_field->addElement(ring);
            }

            idx++;
            if (idx % 1000 == 0) {
                reporter.Update(idx);
            }
        }
    } else if (header.datatype == PCD_DATA_BINARY) {
        std::unique_ptr<char[]> buffer(new char[header.pointsize]);
        for (unsigned int i = 0; i < (unsigned int)header.points; i++) {
            if (fread(buffer.get(), header.pointsize, 1, file) != 1) {
                utility::LogWarning(
                        "[ReadPCDData] Failed to read data record.");
                pointcloud.clear();
                return false;
            }

            bool find_point = false;
            bool find_normal = false;
            bool find_color = false;
            bool find_ring = false;
            bool find_intensity = false;
            for (const auto &field : header.fields) {
                if (field.name == "x") {
                    find_point = true;
                    P.x = UnpackBinaryPCDElement(buffer.get() + field.offset,
                                                 field.type, field.size);
                } else if (field.name == "y") {
                    find_point = true;
                    P.y = UnpackBinaryPCDElement(buffer.get() + field.offset,
                                                 field.type, field.size);
                } else if (field.name == "z") {
                    find_point = true;
                    P.z = UnpackBinaryPCDElement(buffer.get() + field.offset,
                                                 field.type, field.size);
                } else if (field.name == "normal_x") {
                    find_normal = true;
                    N.x = UnpackBinaryPCDElement(buffer.get() + field.offset,
                                                 field.type, field.size);
                } else if (field.name == "normal_y") {
                    find_normal = true;
                    N.y = UnpackBinaryPCDElement(buffer.get() + field.offset,
                                                 field.type, field.size);
                } else if (field.name == "normal_z") {
                    find_normal = true;
                    N.z = UnpackBinaryPCDElement(buffer.get() + field.offset,
                                                 field.type, field.size);
                } else if (field.name == "rgb" || field.name == "rgba") {
                    find_color = true;
                    col = UnpackBinaryPCDColor(buffer.get() + field.offset,
                                               field.type, field.size);
                } else if (field.name == "intensity") {
                    find_intensity = true;
                    intensity =
                            UnpackBinaryPCDScalar(buffer.get() + field.offset,
                                                  field.type, field.size);
                } else if (field.name == "ring") {
                    find_ring = true;
                    ring = UnpackBinaryPCDScalar(buffer.get() + field.offset,
                                                 field.type, field.size);
                }
            }

            if (find_point) {
                pointcloud.setPoint(i, P);
            }
            if (header.has_normals && find_normal) {
                pointcloud.addNorm(N);
            }
            if (header.has_colors && find_color) {
                pointcloud.setPointColor(i, col);
            }
            if (header.has_intensity && find_intensity) {
                cc_intensity_field->addElement(intensity);
            }
            if (header.has_ring && find_ring) {
                cc_ring_field->addElement(ring);
            }

            if (i % 1000 == 0) {
                reporter.Update(i);
            }
        }
    } else if (header.datatype == PCD_DATA_BINARY_COMPRESSED) {
        double reporter_total = 100.0;
        reporter.SetTotal(int(reporter_total));
        reporter.Update(int(reporter_total * 0.01));

        std::uint32_t compressed_size;
        std::uint32_t uncompressed_size;
        if (fread(&compressed_size, sizeof(compressed_size), 1, file) != 1) {
            utility::LogWarning("[ReadPCDData] Failed to read data record.");
            pointcloud.clear();
            return false;
        }
        if (fread(&uncompressed_size, sizeof(uncompressed_size), 1, file) !=
            1) {
            utility::LogWarning("[ReadPCDData] Failed to read data record.");
            pointcloud.clear();
            return false;
        }
        utility::LogDebug(
                "PCD data with {:d} compressed size, and {:d} uncompressed "
                "size.",
                compressed_size, uncompressed_size);
        std::unique_ptr<char[]> buffer_compressed(new char[compressed_size]);
        reporter.Update(int(reporter_total * .1));
        if (fread(buffer_compressed.get(), 1, compressed_size, file) !=
            compressed_size) {
            utility::LogWarning("[ReadPCDData] Failed to read data record.");
            pointcloud.clear();
            return false;
        }
        std::unique_ptr<char[]> buffer(new char[uncompressed_size]);
        reporter.Update(int(reporter_total * .2));
        if (lzf_decompress(buffer_compressed.get(),
                           (unsigned int)compressed_size, buffer.get(),
                           (unsigned int)uncompressed_size) !=
            uncompressed_size) {
            utility::LogWarning("[ReadPCDData] Uncompression failed.");
            pointcloud.clear();
            return false;
        }

        std::vector<CCVector3> normals;
        if (header.has_normals) {
            normals.resize(static_cast<unsigned int>(header.points));
        }

        for (const auto &field : header.fields) {
            const char *base_ptr = buffer.get() + field.offset * header.points;
            double progress =
                    double(base_ptr - buffer.get()) / uncompressed_size;
            reporter.Update(int(reporter_total * (progress + .2)));

            if (field.name == "x") {
                for (size_t i = 0; i < (size_t)header.points; i++) {
                    pointcloud.getPointPtr(i)->x = UnpackBinaryPCDElement(
                            base_ptr + i * field.size * field.count, field.type,
                            field.size);
                }
            } else if (field.name == "y") {
                for (size_t i = 0; i < (size_t)header.points; i++) {
                    pointcloud.getPointPtr(i)->y = UnpackBinaryPCDElement(
                            base_ptr + i * field.size * field.count, field.type,
                            field.size);
                }
            } else if (field.name == "z") {
                for (size_t i = 0; i < (size_t)header.points; i++) {
                    pointcloud.getPointPtr(i)->z = UnpackBinaryPCDElement(
                            base_ptr + i * field.size * field.count, field.type,
                            field.size);
                }
            } else if (field.name == "normal_x") {
                for (size_t i = 0; i < (size_t)header.points; i++) {
                    normals[i].x = UnpackBinaryPCDElement(
                            base_ptr + i * field.size * field.count, field.type,
                            field.size);
                }
            } else if (field.name == "normal_y") {
                for (size_t i = 0; i < (size_t)header.points; i++) {
                    normals[i].y = UnpackBinaryPCDElement(
                            base_ptr + i * field.size * field.count, field.type,
                            field.size);
                }
            } else if (field.name == "normal_z") {
                for (size_t i = 0; i < (size_t)header.points; i++) {
                    normals[i].z = UnpackBinaryPCDElement(
                            base_ptr + i * field.size * field.count, field.type,
                            field.size);
                }
            } else if (field.name == "rgb" || field.name == "rgba") {
                for (unsigned int i = 0; i < (unsigned int)header.points; i++) {
                    ecvColor::Rgb color = UnpackBinaryPCDColor(
                            base_ptr + i * field.size * field.count, field.type,
                            field.size);
                    pointcloud.setPointColor(i, color);
                }
            } else if (field.name == "intensity") {
                for (unsigned int i = 0; i < (unsigned int)header.points; i++) {
                    ScalarType intensity = UnpackBinaryPCDScalar(
                            base_ptr + i * field.size * field.count, field.type,
                            field.size);
                    cc_intensity_field->addElement(intensity);
                }
            } else if (field.name == "ring") {
                for (unsigned int i = 0; i < (unsigned int)header.points; i++) {
                    ScalarType ring = UnpackBinaryPCDScalar(
                            base_ptr + i * field.size * field.count, field.type,
                            field.size);
                    cc_ring_field->addElement(ring);
                }
            }
        }

        if (header.has_normals) {
            pointcloud.addNorms(normals);
        }
    }

    if (header.has_intensity && cc_intensity_field) {
        cc_intensity_field->computeMinAndMax();
        pointcloud.addScalarField(cc_intensity_field);
        pointcloud.setCurrentDisplayedScalarField(0);
        pointcloud.showSF(true);
    }

    if (header.has_ring && cc_ring_field) {
        cc_ring_field->computeMinAndMax();
        pointcloud.addScalarField(cc_ring_field);
        pointcloud.setCurrentDisplayedScalarField(0);
        pointcloud.showSF(true);
    }

    reporter.Finish();
    return true;
}

bool GenerateHeader(const ccPointCloud &pointcloud,
                    const bool write_ascii,
                    const bool compressed,
                    PCDHeader &header) {
    if (!pointcloud.hasPoints()) {
        return false;
    }

    header.version = "0.7";
    header.width = (int)pointcloud.size();
    header.height = 1;
    header.points = header.width;
    header.fields.clear();
    PCLPointField field;
    field.type = 'F';
    field.size = 4;
    field.count = 1;
    field.name = "x";
    header.fields.push_back(field);
    field.name = "y";
    header.fields.push_back(field);
    field.name = "z";
    header.fields.push_back(field);
    header.elementnum = 3;
    header.pointsize = 12;
    if (pointcloud.hasNormals()) {
        field.name = "normal_x";
        header.fields.push_back(field);
        field.name = "normal_y";
        header.fields.push_back(field);
        field.name = "normal_z";
        header.fields.push_back(field);
        header.elementnum += 3;
        header.pointsize += 12;
    }
    if (pointcloud.hasColors()) {
        field.name = "rgb";
        header.fields.push_back(field);
        header.elementnum++;
        header.pointsize += 4;
    }
    if (write_ascii) {
        header.datatype = PCD_DATA_ASCII;
    } else {
        if (compressed) {
            header.datatype = PCD_DATA_BINARY_COMPRESSED;
        } else {
            header.datatype = PCD_DATA_BINARY;
        }
    }
    return true;
}

bool WritePCDHeader(FILE *file, const PCDHeader &header) {
    fprintf(file, "# .PCD v%s - Point Cloud Data file format\n",
            header.version.c_str());
    fprintf(file, "VERSION %s\n", header.version.c_str());
    fprintf(file, "FIELDS");
    for (const auto &field : header.fields) {
        fprintf(file, " %s", field.name.c_str());
    }
    fprintf(file, "\n");
    fprintf(file, "SIZE");
    for (const auto &field : header.fields) {
        fprintf(file, " %d", field.size);
    }
    fprintf(file, "\n");
    fprintf(file, "TYPE");
    for (const auto &field : header.fields) {
        fprintf(file, " %c", field.type);
    }
    fprintf(file, "\n");
    fprintf(file, "COUNT");
    for (const auto &field : header.fields) {
        fprintf(file, " %d", field.count);
    }
    fprintf(file, "\n");
    fprintf(file, "WIDTH %d\n", header.width);
    fprintf(file, "HEIGHT %d\n", header.height);
    fprintf(file, "VIEWPOINT 0 0 0 1 0 0 0\n");
    fprintf(file, "POINTS %d\n", header.points);

    switch (header.datatype) {
        case PCD_DATA_BINARY:
            fprintf(file, "DATA binary\n");
            break;
        case PCD_DATA_BINARY_COMPRESSED:
            fprintf(file, "DATA binary_compressed\n");
            break;
        case PCD_DATA_ASCII:
        default:
            fprintf(file, "DATA ascii\n");
            break;
    }
    return true;
}

float ConvertRGBToFloat(const ecvColor::Rgb &color) {
    std::uint8_t rgba[4] = {0, 0, 0, 0};
    rgba[2] = (std::uint8_t)std::max(std::min((int)color.r, 255), 0);
    rgba[1] = (std::uint8_t)std::max(std::min((int)color.g, 255), 0);
    rgba[0] = (std::uint8_t)std::max(std::min((int)color.b, 255), 0);
    float value;
    memcpy(&value, rgba, 4);
    return value;
}

bool WritePCDData(FILE *file,
                  const PCDHeader &header,
                  const ccPointCloud &pointcloud,
                  const WritePointCloudOption &params) {
    bool has_normal = pointcloud.hasNormals();
    bool has_color = pointcloud.hasColors();

    utility::CountingProgressReporter reporter(params.update_progress);
    reporter.SetTotal(pointcloud.size());

    if (header.datatype == PCD_DATA_ASCII) {
        for (unsigned int i = 0; i < pointcloud.size(); i++) {
            const auto &point = *pointcloud.getPoint(i);
            fprintf(file, "%.10g %.10g %.10g", point.x, point.y, point.z);
            if (has_normal) {
                const auto &normal = pointcloud.getPointNormal(i);
                fprintf(file, " %.10g %.10g %.10g", normal.x, normal.y,
                        normal.z);
            }
            if (has_color) {
                const auto &color = pointcloud.getPointColor(i);
                fprintf(file, " %.10g", ConvertRGBToFloat(color));
            }
            fprintf(file, "\n");
            if (i % 1000 == 0) {
                reporter.Update(i);
            }
        }
    } else if (header.datatype == PCD_DATA_BINARY) {
        std::unique_ptr<float[]> data(new float[header.elementnum]);
        for (unsigned int i = 0; i < pointcloud.size(); i++) {
            const auto &point = *pointcloud.getPoint(i);
            data[0] = (float)point[0];
            data[1] = (float)point[1];
            data[2] = (float)point[2];
            int idx = 3;
            if (has_normal) {
                const auto &normal = pointcloud.getPointNormal(i);
                data[idx + 0] = (float)normal[0];
                data[idx + 1] = (float)normal[1];
                data[idx + 2] = (float)normal[2];
                idx += 3;
            }
            if (has_color) {
                const auto &color = pointcloud.getPointColor(i);
                data[idx] = ConvertRGBToFloat(color);
            }
            fwrite(data.get(), sizeof(float), header.elementnum, file);
            if (i % 1000 == 0) {
                reporter.Update(i);
            }
        }
    } else if (header.datatype == PCD_DATA_BINARY_COMPRESSED) {
        double report_total = double(pointcloud.size() * 2);
        // 0%-50% packing into buffer
        // 50%-75% compressing buffer
        // 75%-100% writing compressed buffer
        reporter.SetTotal(int(report_total));

        int strip_size = header.points;
        std::uint32_t buffer_size =
                (std::uint32_t)(header.elementnum * header.points);
        std::unique_ptr<float[]> buffer(new float[buffer_size]);
        std::unique_ptr<float[]> buffer_compressed(new float[buffer_size * 2]);
        for (unsigned int i = 0; i < pointcloud.size(); i++) {
            const auto &point = *pointcloud.getPoint(i);
            buffer[0 * strip_size + i] = (float)point(0);
            buffer[1 * strip_size + i] = (float)point(1);
            buffer[2 * strip_size + i] = (float)point(2);
            int idx = 3;
            if (has_normal) {
                const auto &normal = pointcloud.getPointNormal(i);
                buffer[(idx + 0) * strip_size + i] = (float)normal(0);
                buffer[(idx + 1) * strip_size + i] = (float)normal(1);
                buffer[(idx + 2) * strip_size + i] = (float)normal(2);
                idx += 3;
            }
            if (has_color) {
                const auto &color = pointcloud.getPointColor(i);
                buffer[idx * strip_size + i] = ConvertRGBToFloat(color);
            }
            if (i % 1000 == 0) {
                reporter.Update(i);
            }
        }
        std::uint32_t buffer_size_in_bytes = buffer_size * sizeof(float);
        std::uint32_t size_compressed =
                lzf_compress(buffer.get(), buffer_size_in_bytes,
                             buffer_compressed.get(), buffer_size_in_bytes * 2);
        if (size_compressed == 0) {
            utility::LogWarning("[WritePCDData] Failed to compress data.");
            return false;
        }
        utility::LogDebug(
                "[WritePCDData] {:d} bytes data compressed into {:d} bytes.",
                buffer_size_in_bytes, size_compressed);
        reporter.Update(int(report_total * 0.75));
        fwrite(&size_compressed, sizeof(size_compressed), 1, file);
        fwrite(&buffer_size_in_bytes, sizeof(buffer_size_in_bytes), 1, file);
        fwrite(buffer_compressed.get(), 1, size_compressed, file);
    }
    reporter.Finish();
    return true;
}

/********************************* PCD UTILITY
 * *************************************/

}  // unnamed namespace

namespace io {

/********************************* PCD IO *************************************/
FileGeometry ReadFileGeometryTypePCD(const std::string &path) {
    return CONTAINS_POINTS;
}

bool ReadPointCloudFromPCD(const std::string &filename,
                           ccPointCloud &pointcloud,
                           const ReadPointCloudOption &params) {
    PCDHeader header;
    FILE *file = utility::filesystem::FOpen(filename.c_str(), "rb");
    if (file == NULL) {
        utility::LogWarning("Read PCD failed: unable to open file: {}",
                            filename);
        return false;
    }
    if (ReadPCDHeader(file, header) == false) {
        utility::LogWarning("Read PCD failed: unable to parse header.");
        fclose(file);
        return false;
    }
    utility::LogDebug(
            "PCD header indicates {:d} fields, {:d} bytes per point, and {:d} "
            "points in total.",
            (int)header.fields.size(), header.pointsize, header.points);
    for (const auto &field : header.fields) {
        utility::LogDebug("{}, {}, {:d}, {:d}, {:d}", field.name.c_str(),
                          field.type, field.size, field.count, field.offset);
    }
    utility::LogDebug("Compression method is {:d}.", (int)header.datatype);
    utility::LogDebug("Points: {};  normals: {};  colors: {}",
                      header.has_points ? "yes" : "no",
                      header.has_normals ? "yes" : "no",
                      header.has_colors ? "yes" : "no");
    if (!ReadPCDData(file, header, pointcloud, params)) {
        utility::LogWarning("Read PCD failed: unable to read data.");
        fclose(file);
        return false;
    }
    fclose(file);
    return true;
}

bool WritePointCloudToPCD(const std::string &filename,
                          const ccPointCloud &pointcloud,
                          const WritePointCloudOption &params) {
    PCDHeader header;
    if (!GenerateHeader(pointcloud, bool(params.write_ascii),
                        bool(params.compressed), header)) {
        utility::LogWarning("Write PCD failed: unable to generate header.");
        return false;
    }
    FILE *file = utility::filesystem::FOpen(filename.c_str(), "wb");
    if (file == NULL) {
        utility::LogWarning("Write PCD failed: unable to open file.");
        return false;
    }
    if (!WritePCDHeader(file, header)) {
        utility::LogWarning("Write PCD failed: unable to write header.");
        fclose(file);
        return false;
    }
    if (!WritePCDData(file, header, pointcloud, params)) {
        utility::LogWarning("Write PCD failed: unable to write data.");
        fclose(file);
        return false;
    }
    fclose(file);
    return true;
}

/********************************* PCD IO *************************************/

/********************************* XYZ IO *************************************/
FileGeometry ReadFileGeometryTypeXYZ(const std::string &path) {
    return CONTAINS_POINTS;
}

bool ReadPointCloudFromXYZ(const std::string &filename,
                           ccPointCloud &pointcloud,
                           const ReadPointCloudOption &params) {
    try {
        utility::filesystem::CFile file;
        if (!file.Open(filename, "r")) {
            utility::LogWarning("Read XYZ failed: unable to open file: {}",
                                filename);
            return false;
        }
        utility::CountingProgressReporter reporter(params.update_progress);
        reporter.SetTotal(file.GetFileSize());

        pointcloud.clear();
        int i = 0;
        double x, y, z;
        const char *line_buffer;
        while ((line_buffer = file.ReadLine())) {
            if (sscanf(line_buffer, "%lf %lf %lf", &x, &y, &z) == 3) {
                pointcloud.addEigenPoint(Eigen::Vector3d(x, y, z));
            }
            if (++i % 1000 == 0) {
                reporter.Update(file.CurPos());
            }
        }
        reporter.Finish();

        return true;
    } catch (const std::exception &e) {
        utility::LogWarning("Read XYZ failed with exception: {}", e.what());
        return false;
    }
}

bool ReadPointCloudInMemoryFromXYZ(const unsigned char *buffer,
                                   const size_t length,
                                   ccPointCloud &pointcloud,
                                   const ReadPointCloudOption &params) {
    try {
        utility::CountingProgressReporter reporter(params.update_progress);
        reporter.SetTotal(static_cast<int64_t>(length));

        std::string content(reinterpret_cast<const char *>(buffer), length);
        std::istringstream ibs(content);
        pointcloud.clear();
        int i = 0;
        double x, y, z;
        std::string line;
        while (std::getline(ibs, line)) {
            if (sscanf(line.c_str(), "%lf %lf %lf", &x, &y, &z) == 3) {
                pointcloud.addEigenPoint(Eigen::Vector3d(x, y, z));
            }
            if (++i % 1000 == 0) {
                reporter.Update(ibs.tellg());
            }
        }
        reporter.Finish();

        return true;
    } catch (const std::exception &e) {
        utility::LogWarning("Read XYZ failed with exception: {}", e.what());
        return false;
    }
}

bool WritePointCloudToXYZ(const std::string &filename,
                          const ccPointCloud &pointcloud,
                          const WritePointCloudOption &params) {
    try {
        utility::filesystem::CFile file;
        if (!file.Open(filename, "w")) {
            utility::LogWarning("Write XYZ failed: unable to open file: {}",
                                filename);
            return false;
        }
        utility::CountingProgressReporter reporter(params.update_progress);
        reporter.SetTotal(pointcloud.size());

        for (size_t i = 0; i < pointcloud.size(); i++) {
            const Eigen::Vector3d &point = pointcloud.getEigenPoint(i);
            if (fprintf(file.GetFILE(), "%.10f %.10f %.10f\n", point(0),
                        point(1), point(2)) < 0) {
                utility::LogWarning(
                        "Write XYZ failed: unable to write file: {}", filename);
                return false;  // error happened during writing.
            }
            if (i % 1000 == 0) {
                reporter.Update(i);
            }
        }
        reporter.Finish();
        return true;
    } catch (const std::exception &e) {
        utility::LogWarning("Write XYZ failed with exception: {}", e.what());
        return false;
    }
}

bool WritePointCloudInMemoryToXYZ(unsigned char *&buffer,
                                  size_t &length,
                                  const ccPointCloud &pointcloud,
                                  const WritePointCloudOption &params) {
    try {
        utility::CountingProgressReporter reporter(params.update_progress);
        reporter.SetTotal(pointcloud.size());

        std::string content;
        for (size_t i = 0; i < pointcloud.size(); i++) {
            const Eigen::Vector3d &point = pointcloud.getEigenPoint(i);
            std::string line = utility::FastFormatString(
                    "%.10f %.10f %.10f\n", point(0), point(1), point(2));
            content.append(line);
            if (i % 1000 == 0) {
                reporter.Update(i);
            }
        }
        // nothing to report...
        if (content.length() == 0) {
            reporter.Finish();
            return false;
        }
        length = content.length();
        buffer = new unsigned char[length];  // we do this for the caller
        std::memcpy(buffer, content.c_str(), length);

        reporter.Finish();
        return true;
    } catch (const std::exception &e) {
        utility::LogWarning("Write XYZ failed with exception: {}", e.what());
        return false;
    }
}
/********************************* XYZ IO *************************************/

/********************************* XYZN IO
 * *************************************/
FileGeometry ReadFileGeometryTypeXYZN(const std::string &path) {
    return CONTAINS_POINTS;
}

bool ReadPointCloudFromXYZN(const std::string &filename,
                            ccPointCloud &pointcloud,
                            const ReadPointCloudOption &params) {
    try {
        utility::filesystem::CFile file;
        if (!file.Open(filename, "r")) {
            utility::LogWarning("Read XYZN failed: unable to open file: {}",
                                filename);
            return false;
        }
        utility::CountingProgressReporter reporter(params.update_progress);
        reporter.SetTotal(file.GetFileSize());

        pointcloud.clear();
        int i = 0;
        double x, y, z, nx, ny, nz;
        const char *line_buffer;
        while ((line_buffer = file.ReadLine())) {
            if (sscanf(line_buffer, "%lf %lf %lf %lf %lf %lf", &x, &y, &z, &nx,
                       &ny, &nz) == 6) {
                pointcloud.addEigenPoint(Eigen::Vector3d(x, y, z));
                pointcloud.addEigenNorm(Eigen::Vector3d(nx, ny, nz));
            }
            if (++i % 1000 == 0) {
                reporter.Update(file.CurPos());
            }
        }
        reporter.Finish();

        return true;
    } catch (const std::exception &e) {
        utility::LogWarning("Read XYZN failed with exception: {}", e.what());
        return false;
    }
}

bool WritePointCloudToXYZN(const std::string &filename,
                           const ccPointCloud &pointcloud,
                           const WritePointCloudOption &params) {
    if (!pointcloud.hasNormals()) {
        return false;
    }

    try {
        utility::filesystem::CFile file;
        if (!file.Open(filename, "w")) {
            utility::LogWarning("Write XYZN failed: unable to open file: {}",
                                filename);
            return false;
        }
        utility::CountingProgressReporter reporter(params.update_progress);
        reporter.SetTotal(pointcloud.size());

        for (size_t i = 0; i < pointcloud.size(); i++) {
            const Eigen::Vector3d &point = pointcloud.getEigenPoint(i);
            const Eigen::Vector3d &normal = pointcloud.getEigenNormal(i);
            if (fprintf(file.GetFILE(), "%.10f %.10f %.10f %.10f %.10f %.10f\n",
                        point(0), point(1), point(2), normal(0), normal(1),
                        normal(2)) < 0) {
                utility::LogWarning(
                        "Write XYZN failed: unable to write file: {}",
                        filename);
                return false;  // error happened during writing.
            }
            if (i % 1000 == 0) {
                reporter.Update(i);
            }
        }
        reporter.Finish();
        return true;
    } catch (const std::exception &e) {
        utility::LogWarning("Write XYZN failed with exception: {}", e.what());
        return false;
    }
}
/********************************* XYZN IO
 * *************************************/

/********************************* XYZRGB IO
 * *************************************/
FileGeometry ReadFileGeometryTypeXYZRGB(const std::string &path) {
    return CONTAINS_POINTS;
}

bool ReadPointCloudFromXYZRGB(const std::string &filename,
                              ccPointCloud &pointcloud,
                              const ReadPointCloudOption &params) {
    try {
        utility::filesystem::CFile file;
        if (!file.Open(filename, "r")) {
            utility::LogWarning("Read XYZRGB failed: unable to open file: {}",
                                filename);
            return false;
        }
        utility::CountingProgressReporter reporter(params.update_progress);
        reporter.SetTotal(file.GetFileSize());

        pointcloud.clear();
        int i = 0;
        double x, y, z, r, g, b;
        const char *line_buffer;
        while ((line_buffer = file.ReadLine())) {
            if (sscanf(line_buffer, "%lf %lf %lf %lf %lf %lf", &x, &y, &z, &r,
                       &g, &b) == 6) {
                pointcloud.addEigenPoint(Eigen::Vector3d(x, y, z));
                pointcloud.addEigenColor(Eigen::Vector3d(r, g, b));
            }
            if (++i % 1000 == 0) {
                reporter.Update(file.CurPos());
            }
        }
        reporter.Finish();

        return true;
    } catch (const std::exception &e) {
        utility::LogWarning("Read XYZ failed with exception: {}", e.what());
        return false;
    }
}

bool WritePointCloudToXYZRGB(const std::string &filename,
                             const ccPointCloud &pointcloud,
                             const WritePointCloudOption &params) {
    if (!pointcloud.hasColors()) {
        return false;
    }

    try {
        utility::filesystem::CFile file;
        if (!file.Open(filename, "w")) {
            utility::LogWarning("Write XYZRGB failed: unable to open file: {}",
                                filename);
            return false;
        }
        utility::CountingProgressReporter reporter(params.update_progress);
        reporter.SetTotal(pointcloud.size());

        for (size_t i = 0; i < pointcloud.size(); i++) {
            const Eigen::Vector3d &point = pointcloud.getEigenPoint(i);
            const Eigen::Vector3d &color = pointcloud.getEigenColor(i);
            if (fprintf(file.GetFILE(), "%.10f %.10f %.10f %.10f %.10f %.10f\n",
                        point(0), point(1), point(2), color(0), color(1),
                        color(2)) < 0) {
                utility::LogWarning(
                        "Write XYZRGB failed: unable to write file: {}",
                        filename);
                return false;  // error happened during writing.
            }
            if (i % 1000 == 0) {
                reporter.Update(i);
            }
        }
        reporter.Finish();
        return true;
    } catch (const std::exception &e) {
        utility::LogWarning("Write XYZRGB failed with exception: {}", e.what());
        return false;
    }
}
/********************************* XYZRGB IO
 * *************************************/

/********************************* PLY IO *************************************/
FileGeometry ReadFileGeometryTypePLY(const std::string &path) {
    p_ply ply_file = ply_open(path.c_str(), NULL, 0, NULL);
    if (!ply_file) {
        return CONTENTS_UNKNOWN;
    }
    if (!ply_read_header(ply_file)) {
        ply_close(ply_file);
        return CONTENTS_UNKNOWN;
    }

    auto nVertices =
            ply_set_read_cb(ply_file, "vertex", "x", nullptr, nullptr, 0);
    auto nLines =
            ply_set_read_cb(ply_file, "edge", "vertex1", nullptr, nullptr, 0);
    auto nFaces = ply_set_read_cb(ply_file, "face", "vertex_indices", nullptr,
                                  nullptr, 0);
    if (nFaces == 0) {
        nFaces = ply_set_read_cb(ply_file, "face", "vertex_index", nullptr,
                                 nullptr, 0);
    }
    ply_close(ply_file);

    int contents = 0;
    if (nFaces > 0) {
        contents |= CONTAINS_TRIANGLES;
    } else if (nLines > 0) {
        contents |= CONTAINS_LINES;
    } else if (nVertices > 0) {
        contents |= CONTAINS_POINTS;
    }
    return FileGeometry(contents);
}

bool ReadPointCloudFromPLY(const std::string &filename,
                           ccPointCloud &pointcloud,
                           const ReadPointCloudOption &params) {
    using namespace ply_pointcloud_reader;

    p_ply ply_file = ply_open(filename.c_str(), NULL, 0, NULL);
    if (!ply_file) {
        utility::LogWarning("Read PLY failed: unable to open file: {}",
                            filename.c_str());
        return false;
    }
    if (!ply_read_header(ply_file)) {
        utility::LogWarning("Read PLY failed: unable to parse header.");
        ply_close(ply_file);
        return false;
    }

    PLYReaderState state;
    state.pointcloud_ptr = &pointcloud;
    state.vertex_num = ply_set_read_cb(ply_file, "vertex", "x",
                                       ReadVertexCallback, &state, 0);
    ply_set_read_cb(ply_file, "vertex", "y", ReadVertexCallback, &state, 1);
    ply_set_read_cb(ply_file, "vertex", "z", ReadVertexCallback, &state, 2);

    state.normal_num = ply_set_read_cb(ply_file, "vertex", "nx",
                                       ReadNormalCallback, &state, 0);
    ply_set_read_cb(ply_file, "vertex", "ny", ReadNormalCallback, &state, 1);
    ply_set_read_cb(ply_file, "vertex", "nz", ReadNormalCallback, &state, 2);

    state.color_num = ply_set_read_cb(ply_file, "vertex", "red",
                                      ReadColorCallback, &state, 0);
    ply_set_read_cb(ply_file, "vertex", "green", ReadColorCallback, &state, 1);
    ply_set_read_cb(ply_file, "vertex", "blue", ReadColorCallback, &state, 2);

    if (state.vertex_num <= 0) {
        utility::LogWarning("Read PLY failed: number of vertex <= 0.");
        ply_close(ply_file);
        return false;
    }

    state.vertex_index = 0;
    state.normal_index = 0;
    state.color_index = 0;

    pointcloud.clear();
    pointcloud.resize(state.vertex_num);

    if (state.color_num > 1) {
        pointcloud.resizeTheRGBTable();
    }

    if (state.normal_num > 1) {
        pointcloud.reserveTheNormsTable();
    }

    utility::CountingProgressReporter reporter(params.update_progress);
    reporter.SetTotal(state.vertex_num);
    state.progress_bar = &reporter;

    if (!ply_read(ply_file)) {
        utility::LogWarning("Read PLY failed: unable to read file: {}",
                            filename);
        ply_close(ply_file);
        return false;
    }

    ply_close(ply_file);
    reporter.Finish();
    return true;
}

bool WritePointCloudToPLY(const std::string &filename,
                          const ccPointCloud &pointcloud,
                          const WritePointCloudOption &params) {
    if (pointcloud.isEmpty()) {
        utility::LogWarning("Write PLY failed: point cloud has 0 points.");
        return false;
    }

    p_ply ply_file =
            ply_create(filename.c_str(),
                       bool(params.write_ascii) ? PLY_ASCII : PLY_LITTLE_ENDIAN,
                       NULL, 0, NULL);
    if (!ply_file) {
        utility::LogWarning("Write PLY failed: unable to open file: {}",
                            filename);
        return false;
    }
    ply_add_comment(ply_file, "Created by CloudViewer");
    ply_add_element(ply_file, "vertex", static_cast<long>(pointcloud.size()));
    ply_add_property(ply_file, "x", PLY_DOUBLE, PLY_DOUBLE, PLY_DOUBLE);
    ply_add_property(ply_file, "y", PLY_DOUBLE, PLY_DOUBLE, PLY_DOUBLE);
    ply_add_property(ply_file, "z", PLY_DOUBLE, PLY_DOUBLE, PLY_DOUBLE);
    if (pointcloud.hasNormals()) {
        ply_add_property(ply_file, "nx", PLY_DOUBLE, PLY_DOUBLE, PLY_DOUBLE);
        ply_add_property(ply_file, "ny", PLY_DOUBLE, PLY_DOUBLE, PLY_DOUBLE);
        ply_add_property(ply_file, "nz", PLY_DOUBLE, PLY_DOUBLE, PLY_DOUBLE);
    }
    if (pointcloud.hasColors()) {
        ply_add_property(ply_file, "red", PLY_UCHAR, PLY_UCHAR, PLY_UCHAR);
        ply_add_property(ply_file, "green", PLY_UCHAR, PLY_UCHAR, PLY_UCHAR);
        ply_add_property(ply_file, "blue", PLY_UCHAR, PLY_UCHAR, PLY_UCHAR);
    }
    if (!ply_write_header(ply_file)) {
        utility::LogWarning("Write PLY failed: unable to write header.");
        ply_close(ply_file);
        return false;
    }

    utility::CountingProgressReporter reporter(params.update_progress);
    reporter.SetTotal(pointcloud.size());

    bool printed_color_warning = false;
    for (size_t i = 0; i < pointcloud.size(); i++) {
        const Eigen::Vector3d &point = pointcloud.getEigenPoint(i);
        ply_write(ply_file, point(0));
        ply_write(ply_file, point(1));
        ply_write(ply_file, point(2));
        if (pointcloud.hasNormals()) {
            const Eigen::Vector3d &normal = pointcloud.getEigenNormal(i);
            ply_write(ply_file, normal(0));
            ply_write(ply_file, normal(1));
            ply_write(ply_file, normal(2));
        }
        if (pointcloud.hasColors()) {
            const Eigen::Vector3d &color = pointcloud.getEigenColor(i);
            if (!printed_color_warning &&
                (color(0) < 0 || color(0) > 1 || color(1) < 0 || color(1) > 1 ||
                 color(2) < 0 || color(2) > 1)) {
                utility::LogWarning(
                        "Write Ply clamped color value to valid range");
                printed_color_warning = true;
            }
            ply_write(ply_file,
                      std::min(255.0, std::max(0.0, color(0) * 255.0)));
            ply_write(ply_file,
                      std::min(255.0, std::max(0.0, color(1) * 255.0)));
            ply_write(ply_file,
                      std::min(255.0, std::max(0.0, color(2) * 255.0)));
        }
        if (i % 1000 == 0) {
            reporter.Update(i);
        }
    }

    reporter.Finish();
    ply_close(ply_file);
    return true;
}
/********************************* PLY IO *************************************/

/********************************* PTS IO *************************************/
FileGeometry ReadFileGeometryTypePTS(const std::string &path) {
    return CONTAINS_POINTS;
}

bool ReadPointCloudFromPTS(const std::string &filename,
                           ccPointCloud &pointcloud,
                           const ReadPointCloudOption &params) {
    try {
        utility::filesystem::CFile file;
        if (!file.Open(filename, "r")) {
            utility::LogWarning("Read PTS failed: unable to open file: {}",
                                filename);
            return false;
        }
        size_t num_of_pts = 0;
        int num_of_fields = 0;
        const char *line_buffer;
        if ((line_buffer = file.ReadLine())) {
            sscanf(line_buffer, "%zu", &num_of_pts);
        }
        if (num_of_pts <= 0) {
            utility::LogWarning("Read PTS failed: unable to read header.");
            return false;
        }
        utility::CountingProgressReporter reporter(params.update_progress);
        reporter.SetTotal(num_of_pts);

        pointcloud.clear();
        size_t idx = 0;
        while (idx < num_of_pts && (line_buffer = file.ReadLine())) {
            if (num_of_fields == 0) {
                std::vector<std::string> st;
                utility::SplitString(st, line_buffer, " ");
                num_of_fields = (int)st.size();
                if (num_of_fields < 3) {
                    utility::LogWarning(
                            "Read PTS failed: insufficient data fields.");
                    return false;
                }
                pointcloud.resize(static_cast<unsigned>(num_of_pts));
                if (num_of_fields >= 7) {
                    // X Y Z I R G B
                    pointcloud.resizeTheRGBTable();
                }
            }
            double x, y, z;
            int i, r, g, b;
            if (num_of_fields < 7) {
                if (sscanf(line_buffer, "%lf %lf %lf", &x, &y, &z) == 3) {
                    pointcloud.setPoint(idx, Eigen::Vector3d(x, y, z));
                }
            } else {
                if (sscanf(line_buffer, "%lf %lf %lf %d %d %d %d", &x, &y, &z,
                           &i, &r, &g, &b) == 7) {
                    pointcloud.setPoint(idx, Eigen::Vector3d(x, y, z));
                    pointcloud.setPointColor(static_cast<unsigned>(idx),
                                             ecvColor::Rgb(r, g, b));
                }
            }
            idx++;
            if (idx % 1000 == 0) {
                reporter.Update(idx);
            }
        }
        reporter.Finish();

        return true;
    } catch (const std::exception &e) {
        utility::LogWarning("Read PTS failed with exception: {}", e.what());
        return false;
    }
}

bool WritePointCloudToPTS(const std::string &filename,
                          const ccPointCloud &pointcloud,
                          const WritePointCloudOption &params) {
    try {
        utility::filesystem::CFile file;
        if (!file.Open(filename, "w")) {
            utility::LogWarning("Write PTS failed: unable to open file: {}",
                                filename);
            return false;
        }
        utility::CountingProgressReporter reporter(params.update_progress);
        reporter.SetTotal(pointcloud.size());

        if (fprintf(file.GetFILE(), "%zu\r\n", (size_t)pointcloud.size()) < 0) {
            utility::LogWarning("Write PTS failed: unable to write file: {}",
                                filename);
            return false;
        }
        for (size_t i = 0; i < pointcloud.size(); i++) {
            const auto &point = pointcloud.getEigenPoint(i);
            if (!pointcloud.hasColors()) {
                if (fprintf(file.GetFILE(), "%.10f %.10f %.10f\r\n", point(0),
                            point(1), point(2)) < 0) {
                    utility::LogWarning(
                            "Write PTS failed: unable to write file: {}",
                            filename);
                    return false;
                }
            } else {
                auto color = utility::ColorToUint8(pointcloud.getEigenColor(i));
                if (fprintf(file.GetFILE(), "%.10f %.10f %.10f %d %d %d %d\r\n",
                            point(0), point(1), point(2), 0, (int)color(0),
                            (int)color(1), (int)(color(2))) < 0) {
                    utility::LogWarning(
                            "Write PTS failed: unable to write file: {}",
                            filename);
                    return false;
                }
            }
            if (i % 1000 == 0) {
                reporter.Update(i);
            }
        }
        reporter.Finish();
        return true;
    } catch (const std::exception &e) {
        utility::LogWarning("Write PTS failed with exception: {}", e.what());
        return false;
    }
}

/********************************* PTS IO *************************************/

}  // namespace io
}  // namespace cloudViewer
