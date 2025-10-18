// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "PcdFilter.h"

#include <liblzf/lzf.h>

// Dialog
#include <ui_PCDOutputFormatDlg.h>
#include <ui_savePCDFileDlg.h>

// PclUtils
#include <PCLConv.h>
#include <cc2sm.h>
#include <sm2cc.h>

// CV_CORE_LIB
#include <CVTools.h>
#include <FileSystem.h>
#include <Helper.h>
#include <ProgressReporters.h>

// ECV_DB_LIB
#include <ecvGBLSensor.h>
#include <ecvHObjectCaster.h>
#include <ecvPointCloud.h>
#include <ecvScalarField.h>

// Qt
#include <QDialog>
#include <QFileInfo>
#include <QPushButton>
#include <QSettings>

// Boost
#include <boost/bind.hpp>
#include <boost/make_shared.hpp>

// System
#include <iostream>

// pcl
#include <pcl/filters/passthrough.h>
#include <pcl/io/pcd_io.h>

using namespace cloudViewer;
namespace {

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
    bool has_timestamp;
};

bool CheckHeader(PCDHeader& header) {
    if (header.points <= 0 || header.pointsize <= 0) {
        CVLog::Warning(QString("[CheckHeader] PCD has no data."));
        return false;
    }
    if (header.fields.size() == 0 || header.pointsize <= 0) {
        CVLog::Warning(QString("[CheckHeader] PCD has no fields."));
        return false;
    }
    header.has_points = false;
    header.has_normals = false;
    header.has_colors = false;
    header.has_ring = false;
    header.has_intensity = false;
    header.has_timestamp = false;
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
    bool has_timestamp = false;
    for (const auto& field : header.fields) {
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
        } else if (field.name == "timestamp") {
            has_timestamp = true;
        } else if (field.name == "ring") {
            has_ring = true;
        }
    }
    header.has_points = (has_x && has_y && has_z);
    header.has_normals = (has_normal_x && has_normal_y && has_normal_z);
    header.has_colors = (has_rgb || has_rgba);
    header.has_intensity = has_intensity;
    header.has_timestamp = has_timestamp;
    header.has_ring = has_ring;
    if (header.has_points == false) {
        CVLog::Warning(QString(
                "[CheckHeader] Fields for point data are not complete."));
        return false;
    }
    return true;
}

bool ReadPCDHeader(FILE* file, PCDHeader& header) {
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
                CVLog::Warning(QString("[ReadPCDHeader] Bad PCD file format."));
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
                CVLog::Warning(QString("[ReadPCDHeader] Bad PCD file format."));
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
                CVLog::Warning(QString("[ReadPCDHeader] Bad PCD file format."));
                return false;
            }
            for (size_t i = 0; i < specified_channel_count; i++) {
                header.fields[i].type = st[i + 1].c_str()[0];
            }
        } else if (line_type.substr(0, 5) == "COUNT") {
            if (specified_channel_count != st.size() - 1) {
                CVLog::Warning(QString("[ReadPCDHeader] Bad PCD file format."));
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

PointCoordinateType UnpackBinaryPCDElement(const char* data_ptr,
                                           const char type,
                                           const int size) {
    if (type == 'I') {
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
    } else if (type == 'U') {
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
    } else if (type == 'F') {
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

ecvColor::Rgb UnpackBinaryPCDColor(const char* data_ptr,
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

ScalarType UnpackBinaryPCDScalar(const char* data_ptr,
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

PointCoordinateType UnpackASCIIPCDElement(const char* data_ptr,
                                          const char type,
                                          const int size) {
    char* end;
    if (type == 'I') {
        return static_cast<PointCoordinateType>(std::strtol(data_ptr, &end, 0));
    } else if (type == 'U') {
        return static_cast<PointCoordinateType>(
                std::strtoul(data_ptr, &end, 0));
    } else if (type == 'F') {
        return static_cast<PointCoordinateType>(std::strtod(data_ptr, &end));
    }
    return static_cast<PointCoordinateType>(0.0);
}

ecvColor::Rgb UnpackASCIIPCDColor(const char* data_ptr,
                                  const char type,
                                  const int size) {
    if (size == 4) {
        std::uint8_t data[4] = {0, 0, 0, 0};
        char* end;
        if (type == 'I') {
            std::int32_t value = std::strtol(data_ptr, &end, 0);
            memcpy(data, &value, 4);
        } else if (type == 'U') {
            std::uint32_t value = std::strtoul(data_ptr, &end, 0);
            memcpy(data, &value, 4);
        } else if (type == 'F') {
            float value = std::strtof(data_ptr, &end);
            memcpy(data, &value, 4);
        }
        return ecvColor::Rgb(data[2], data[1], data[0]);
    } else {
        return ecvColor::Rgb();
    }
}

ScalarType UnpackASCIIPCDScalar(const char* data_ptr,
                                const char type,
                                const int size) {
    if (size == 1) {
        std::uint8_t data;
        char* end;
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

bool ReadPCDData(FILE* file,
                 const PCDHeader& header,
                 ccPointCloud& pointcloud) {
    pointcloud.clear();
    // The header should have been checked
    if (header.has_points) {
        pointcloud.resize(static_cast<unsigned int>(header.points));
    } else {
        CVLog::Warning(QString(
                "[ReadPCDData] Fields for point data are not complete."));
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
    ccScalarField* cc_intensity_field = nullptr;
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
    ccScalarField* cc_timestamp_field = nullptr;
    if (header.has_timestamp) {
        int id = pointcloud.getScalarFieldIndexByName("timestamp");
        if (id >= 0) {
            pointcloud.deleteScalarField(id);
        }

        cc_timestamp_field = new ccScalarField("timestamp");
        if (!cc_timestamp_field->reserveSafe(
                    static_cast<unsigned>(pointCount))) {
            cc_timestamp_field->release();
            return false;
        }
    }
    ccScalarField* cc_ring_field = nullptr;
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

    std::function<bool(double)> update_progress;
    utility::CountingProgressReporter reporter(update_progress);
    reporter.SetTotal(header.points);

    CCVector3 P(0, 0, 0);
    CCVector3 N(0, 0, 0);
    ecvColor::Rgb col;
    ScalarType ring;
    ScalarType intensity;
    ScalarType timestamp;

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
            bool find_timestamp = false;
            for (size_t i = 0; i < header.fields.size(); i++) {
                const auto& field = header.fields[i];
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
                } else if (field.name == "timestamp") {
                    find_timestamp = true;
                    timestamp = UnpackASCIIPCDScalar(
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
            if (header.has_timestamp && find_timestamp) {
                cc_timestamp_field->addElement(timestamp);
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
                CVLog::Warning(
                        QString("[ReadPCDData] Failed to read data record."));
                pointcloud.clear();
                return false;
            }

            bool find_point = false;
            bool find_normal = false;
            bool find_color = false;
            bool find_ring = false;
            bool find_intensity = false;
            bool find_timestamp = false;
            for (const auto& field : header.fields) {
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
                } else if (field.name == "timestamp") {
                    find_timestamp = true;
                    timestamp =
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
            if (header.has_timestamp && find_timestamp) {
                cc_timestamp_field->addElement(timestamp);
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
            CVLog::Warning(
                    QString("[ReadPCDData] Failed to read data record."));
            pointcloud.clear();
            return false;
        }
        if (fread(&uncompressed_size, sizeof(uncompressed_size), 1, file) !=
            1) {
            CVLog::Warning(
                    QString("[ReadPCDData] Failed to read data record."));
            pointcloud.clear();
            return false;
        }

        CVLog::PrintDebug(
                QString("PCD data with %1 compressed size, and %2 uncompressed "
                        "size.")
                        .arg(compressed_size)
                        .arg(uncompressed_size));
        std::unique_ptr<char[]> buffer_compressed(new char[compressed_size]);
        reporter.Update(int(reporter_total * .1));
        if (fread(buffer_compressed.get(), 1, compressed_size, file) !=
            compressed_size) {
            CVLog::Warning(
                    QString("[ReadPCDData] Failed to read data record."));
            pointcloud.clear();
            return false;
        }
        std::unique_ptr<char[]> buffer(new char[uncompressed_size]);
        reporter.Update(int(reporter_total * .2));
        if (lzf_decompress(buffer_compressed.get(),
                           (unsigned int)compressed_size, buffer.get(),
                           (unsigned int)uncompressed_size) !=
            uncompressed_size) {
            CVLog::Warning(QString("[ReadPCDData] Uncompression failed."));
            pointcloud.clear();
            return false;
        }

        std::vector<CCVector3> normals;
        if (header.has_normals) {
            normals.resize(static_cast<unsigned int>(header.points));
        }

        for (const auto& field : header.fields) {
            const char* base_ptr = buffer.get() + field.offset * header.points;
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
            } else if (field.name == "timestamp") {
                for (unsigned int i = 0; i < (unsigned int)header.points; i++) {
                    ScalarType timestamp = UnpackBinaryPCDScalar(
                            base_ptr + i * field.size * field.count, field.type,
                            field.size);
                    cc_timestamp_field->addElement(timestamp);
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

    if (header.has_ring && cc_ring_field) {
        cc_ring_field->computeMinAndMax();
        int sfIdex = pointcloud.addScalarField(cc_ring_field);
        pointcloud.setCurrentDisplayedScalarField(sfIdex);
        pointcloud.showSF(true);
    }

    if (header.has_intensity && cc_intensity_field) {
        cc_intensity_field->computeMinAndMax();
        int sfIdex = pointcloud.addScalarField(cc_intensity_field);
        pointcloud.setCurrentDisplayedScalarField(sfIdex);
        pointcloud.showSF(true);
    }

    if (header.has_timestamp && cc_timestamp_field) {
        cc_timestamp_field->computeMinAndMax();
        int sfIdex = pointcloud.addScalarField(cc_timestamp_field);
        pointcloud.setCurrentDisplayedScalarField(sfIdex);
        pointcloud.showSF(true);
    }

    reporter.Finish();
    return true;
}

bool ReadPointCloudFromPCD(const std::string& filename,
                           ccPointCloud& pointcloud) {
    PCDHeader header;
    FILE* file = utility::filesystem::FOpen(filename.c_str(), "rb");
    if (file == NULL) {
        CVLog::Warning(QString("Read PCD failed: unable to open file: %1")
                               .arg(qPrintable(filename.c_str())));
        return false;
    }
    if (ReadPCDHeader(file, header) == false) {
        CVLog::Warning(QString("Read PCD failed: unable to parse header."));
        fclose(file);
        return false;
    }

    CVLog::PrintDebug(QString("PCD header indicates %1 fields, %2 bytes "
                              "per point, and %3 points in total.")
                              .arg(header.fields.size())
                              .arg(header.pointsize)
                              .arg(header.points));
    for (const auto& field : header.fields) {
        CVLog::PrintDebug(QString("%1, %2, %3, %4, %5")
                                  .arg(field.name.c_str())
                                  .arg(field.type)
                                  .arg(field.size)
                                  .arg(field.count)
                                  .arg(field.offset));
    }
    CVLog::PrintDebug(
            QString("Compression method is %1.").arg(header.datatype));
    CVLog::PrintDebug(QString("Points: %1;  normals: %2;  colors: %3")
                              .arg(header.has_points ? "yes" : "no")
                              .arg(header.has_normals ? "yes" : "no")
                              .arg(header.has_colors ? "yes" : "no"));
    if (!ReadPCDData(file, header, pointcloud)) {
        CVLog::Warning(QString("Read PCD failed: unable to read data."));
        fclose(file);
        return false;
    }
    fclose(file);
    return true;
}

}  // namespace

PcdFilter::PcdFilter()
    : FileIOFilter({"_Point Cloud Library Filter",
                    13.0f,  // priority
                    QStringList{"pcd"}, "pcd",
                    QStringList{"Point Cloud Library cloud (*.pcd)"},
                    QStringList{"Point Cloud Library cloud (*.pcd)"},
                    Import | Export}) {}

bool PcdFilter::canSave(CV_CLASS_ENUM type,
                        bool& multiple,
                        bool& exclusive) const {
    // only one cloud per file
    if (type == CV_TYPES::POINT_CLOUD) {
        multiple = false;
        exclusive = true;
        return true;
    }

    return false;
}

static PcdFilter::PCDOutputFileFormat s_outputFileFormat = PcdFilter::AUTO;
void PcdFilter::SetOutputFileFormat(PCDOutputFileFormat format) {
    s_outputFileFormat = format;
}

//! Dialog for the PCV plugin
class ccPCDFileOutputForamtDialog : public QDialog,
                                    public Ui::PCDOutputFormatDialog {
public:
    explicit ccPCDFileOutputForamtDialog(QWidget* parent = nullptr)
        : QDialog(parent, Qt::Tool), Ui::PCDOutputFormatDialog() {
        setupUi(this);
    }
};

//! PCD File Save dialog
class SavePCDFileDialog : public QDialog, public Ui::SavePCDFileDlg {
public:
    //! Default constructor
    explicit SavePCDFileDialog(QWidget* parent = nullptr)
        : QDialog(parent), Ui::SavePCDFileDlg() {
        setupUi(this);
    }
};

CC_FILE_ERROR PcdFilter::saveToFile(ccHObject* entity,
                                    const QString& filename,
                                    const SaveParameters& parameters) {
    if (!entity || filename.isEmpty()) {
        return CC_FERR_BAD_ARGUMENT;
    }

    // the cloud to save
    ccPointCloud* ccCloud = ccHObjectCaster::ToPointCloud(entity);
    if (!ccCloud) {
        CVLog::Warning("[PCL] This filter can only save one cloud at a time!");
        return CC_FERR_BAD_ENTITY_TYPE;
    }

    QSettings settings("save pcd");
    settings.beginGroup("SavePcd");
    bool saveOriginOrientation =
            settings.value("saveOriginOrientation", true).toBool();
    bool saveBinary = settings.value("SavePCDBinary", true).toBool();
    bool compressedMode = settings.value("Compressed", true).toBool();

    PcdFilter::PCDOutputFileFormat outputFileFormat = s_outputFileFormat;
    if (outputFileFormat == AUTO) {
        if (nullptr == parameters.parentWidget) {
            // defaulting to compressed binary
            outputFileFormat = COMPRESSED_BINARY;
        } else {
            // GUI version: show a dialog to let the user choose the output
            // format
            ccPCDFileOutputForamtDialog dialog;
            static PcdFilter::PCDOutputFileFormat s_previousOutputFileFormat =
                    COMPRESSED_BINARY;
            switch (s_previousOutputFileFormat) {
                case COMPRESSED_BINARY:
                    dialog.compressedBinaryRadioButton->setChecked(true);
                    break;
                case BINARY:
                    dialog.binaryRadioButton->setChecked(true);
                    break;
                case ASCII:
                    dialog.asciiRadioButton->setChecked(true);
                    break;
            }

            QAbstractButton* clickedButton = nullptr;

            QObject::connect(
                    dialog.buttonBox, &QDialogButtonBox::clicked,
                    [&](QAbstractButton* button) { clickedButton = button; });

            if (dialog.exec()) {
                if (dialog.compressedBinaryRadioButton->isChecked()) {
                    outputFileFormat = COMPRESSED_BINARY;
                } else if (dialog.binaryRadioButton->isChecked()) {
                    outputFileFormat = BINARY;
                } else if (dialog.asciiRadioButton->isChecked()) {
                    outputFileFormat = ASCII;
                } else {
                    assert(false);
                }

                s_previousOutputFileFormat = outputFileFormat;

                if (clickedButton ==
                    dialog.buttonBox->button(
                            QDialogButtonBox::StandardButton::YesToAll)) {
                    // remember once and for all the output file format
                    s_outputFileFormat = outputFileFormat;
                }
            } else {
                return CC_FERR_CANCELED_BY_USER;
            }
        }

        saveOriginOrientation = true;
        switch (outputFileFormat) {
            case COMPRESSED_BINARY: {
                saveBinary = true;
                compressedMode = true;
                break;
            }
            case BINARY: {
                saveBinary = true;
                compressedMode = false;
                break;
            }
            case ASCII: {
                saveBinary = false;
                compressedMode = false;
                break;
            }
        }

    } else {
        // display pcd save dialog
        if (nullptr == parameters.parentWidget) {
            // defaulting to compressed binary
            saveOriginOrientation = saveOriginOrientation;
            saveBinary = saveBinary;
            compressedMode = compressedMode;
        } else {
            SavePCDFileDialog spfDlg(parameters.parentWidget);
            compressedMode = saveBinary && compressedMode ? true : false;
            {
                spfDlg.saveOriginOrientationCheckBox->setChecked(
                        saveOriginOrientation);
                spfDlg.saveBinaryCheckBox->setChecked(saveBinary);
                spfDlg.saveCompressedCheckBox->setChecked(compressedMode);

                if (!spfDlg.exec()) return CC_FERR_CANCELED_BY_USER;

                saveOriginOrientation =
                        spfDlg.saveOriginOrientationCheckBox->isChecked();
                saveBinary = spfDlg.saveBinaryCheckBox->isChecked();
                compressedMode = spfDlg.saveCompressedCheckBox->isChecked();
            }
        }
    }

    settings.setValue("saveOriginOrientation", saveOriginOrientation);
    settings.setValue("SavePCDBinary", saveBinary);
    settings.setValue("Compressed",
                      saveBinary && compressedMode ? true : false);
    settings.endGroup();

    PCLCloud::Ptr pclCloud = cc2smReader(ccCloud).getAsSM();
    if (!pclCloud) {
        return CC_FERR_THIRD_PARTY_LIB_FAILURE;
    }

    if (saveOriginOrientation) {
        // search for a sensor as child (we take the first if there are
        // several of them)
        ccSensor* sensor(nullptr);
        {
            for (unsigned i = 0; i < ccCloud->getChildrenNumber(); ++i) {
                ccHObject* child = ccCloud->getChild(i);

                // try to cast to a ccSensor
                sensor = ccHObjectCaster::ToSensor(child);
                if (sensor) break;
            }
        }

        Eigen::Vector4f pos;
        Eigen::Quaternionf ori;
        if (!sensor) {
            // we append to the cloud null sensor informations
            pos = Eigen::Vector4f::Zero();
            ori = Eigen::Quaternionf::Identity();
        } else {
            // we get out valid sensor informations
            ccGLMatrix mat = sensor->getRigidTransformation();
            CCVector3 trans = mat.getTranslationAsVec3D();
            pos(0) = trans.x;
            pos(1) = trans.y;
            pos(2) = trans.z;

            // also the rotation
            Eigen::Matrix3f eigrot;
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j) eigrot(i, j) = mat.getColumn(j)[i];

            // now translate to a quaternion notation
            ori = Eigen::Quaternionf(eigrot);
        }

        if (ccCloud->size() == 0) {
            pcl::PCDWriter w;
            QFile file(filename);
            if (!file.open(QFile::WriteOnly | QFile::Truncate))
                return CC_FERR_WRITING;
            QTextStream stream(&file);

            if (compressedMode) {
                stream << QString(w.generateHeaderBinaryCompressed(*pclCloud,
                                                                   pos, ori)
                                          .c_str())
                       << "DATA binary\n";
            } else {
                stream << QString(w.generateHeaderBinary(*pclCloud, pos, ori)
                                          .c_str())
                       << "DATA binary\n";
            }
            return CC_FERR_NO_ERROR;
        }

        if (compressedMode) {
            pcl::PCDWriter w;
            if (w.writeBinaryCompressed(
                        /*qPrintable*/ CVTools::FromQString(filename),
                        *pclCloud, pos, ori) < 0) {
                return CC_FERR_THIRD_PARTY_LIB_FAILURE;
            }
        } else {
            if (pcl::io::savePCDFile(
                        /*qPrintable*/ CVTools::FromQString(filename),
                        *pclCloud, pos, ori,
                        saveBinary) < 0)  // DGM: warning, toStdString doesn't
                                          // preserve "local" characters
            {
                return CC_FERR_THIRD_PARTY_LIB_FAILURE;
            }
        }
    } else {
        if (ccCloud->size() == 0) {
            pcl::PCDWriter w;
            QFile file(filename);
            if (!file.open(QFile::WriteOnly | QFile::Truncate))
                return CC_FERR_WRITING;
            QTextStream stream(&file);

            Eigen::Vector4f pos;
            Eigen::Quaternionf ori;

            if (compressedMode) {
                stream << QString(w.generateHeaderBinaryCompressed(*pclCloud,
                                                                   pos, ori)
                                          .c_str())
                       << "DATA binary\n";
            } else {
                stream << QString(w.generateHeaderBinary(*pclCloud, pos, ori)
                                          .c_str())
                       << "DATA binary\n";
            }
            return CC_FERR_NO_ERROR;
        }

        bool hasColor = ccCloud->hasColors();
        bool hasNormals = ccCloud->hasNormals();
        if (hasColor && !hasNormals) {
            PointCloudRGB::Ptr rgbCloud(new PointCloudRGB);
            FROM_PCL_CLOUD(*pclCloud, *rgbCloud);
            if (!rgbCloud) {
                return CC_FERR_THIRD_PARTY_LIB_FAILURE;
            }
            if (compressedMode) {
                if (pcl::io::savePCDFileBinaryCompressed(
                            CVTools::FromQString(filename), *rgbCloud) <
                    0)  // DGM: warning, toStdString doesn't preserve
                        // "local" characters
                {
                    return CC_FERR_THIRD_PARTY_LIB_FAILURE;
                }
            } else {
                if (pcl::io::savePCDFile(CVTools::FromQString(filename),
                                         *rgbCloud, saveBinary) <
                    0)  // DGM: warning, toStdString doesn't preserve
                        // "local" characters
                {
                    return CC_FERR_THIRD_PARTY_LIB_FAILURE;
                }
            }
        } else if (!hasColor && hasNormals) {
            PointCloudNormal::Ptr normalCloud(new PointCloudNormal);
            FROM_PCL_CLOUD(*pclCloud, *normalCloud);
            if (!normalCloud) {
                return CC_FERR_THIRD_PARTY_LIB_FAILURE;
            }
            if (compressedMode) {
                if (pcl::io::savePCDFileBinaryCompressed(
                            CVTools::FromQString(filename), *normalCloud) <
                    0)  // DGM: warning, toStdString doesn't preserve
                        // "local" characters
                {
                    return CC_FERR_THIRD_PARTY_LIB_FAILURE;
                }
            } else {
                if (pcl::io::savePCDFile(CVTools::FromQString(filename),
                                         *normalCloud, saveBinary) <
                    0)  // DGM: warning, toStdString doesn't preserve
                        // "local" characters
                {
                    return CC_FERR_THIRD_PARTY_LIB_FAILURE;
                }
            }
        } else if (hasColor && hasNormals) {
            PointCloudRGBNormal::Ptr rgbNormalCloud(new PointCloudRGBNormal);
            FROM_PCL_CLOUD(*pclCloud, *rgbNormalCloud);
            if (!rgbNormalCloud) {
                return CC_FERR_THIRD_PARTY_LIB_FAILURE;
            }
            if (compressedMode) {
                if (pcl::io::savePCDFileBinaryCompressed(
                            CVTools::FromQString(filename),
                            *rgbNormalCloud) <
                    0)  // DGM: warning, toStdString doesn't preserve
                        // "local" characters
                {
                    return CC_FERR_THIRD_PARTY_LIB_FAILURE;
                }
            } else {
                if (pcl::io::savePCDFile(CVTools::FromQString(filename),
                                         *rgbNormalCloud, saveBinary) <
                    0)  // DGM: warning, toStdString doesn't preserve
                        // "local" characters
                {
                    return CC_FERR_THIRD_PARTY_LIB_FAILURE;
                }
            }
        } else  // just save xyz coordinates
        {
            PointCloudT::Ptr xyzCloud(new PointCloudT);
            FROM_PCL_CLOUD(*pclCloud, *xyzCloud);
            if (!xyzCloud) {
                return CC_FERR_THIRD_PARTY_LIB_FAILURE;
            }
            if (compressedMode) {
                if (pcl::io::savePCDFileBinaryCompressed(
                            CVTools::FromQString(filename), *xyzCloud) <
                    0)  // DGM: warning, toStdString doesn't preserve
                        // "local" characters
                {
                    return CC_FERR_THIRD_PARTY_LIB_FAILURE;
                }
            } else {
                if (pcl::io::savePCDFile(CVTools::FromQString(filename),
                                         *xyzCloud, saveBinary) <
                    0)  // DGM: warning, toStdString doesn't preserve
                        // "local" characters
                {
                    return CC_FERR_THIRD_PARTY_LIB_FAILURE;
                }
            }
        }
    }

    return CC_FERR_NO_ERROR;
}

CC_FILE_ERROR PcdFilter::loadFile(const QString& filename,
                                  ccHObject& container,
                                  LoadParameters& parameters) {
    Eigen::Vector4f origin;
    Eigen::Quaternionf orientation;
    int pcd_version;
    int data_type;
    unsigned int data_idx;
    size_t pointCount = -1;
    PCLCloud::Ptr inputCloud(new PCLCloud);
    // Load the given file
    pcl::PCDReader p;

    const std::string& fileName = CVTools::FromQString(filename);

    if (p.readHeader(fileName, *inputCloud, origin, orientation, pcd_version,
                     data_type, data_idx) < 0) {
        CVLog::Warning(QString("[PCL] An error occurred while reading PCD "
                               "header and try to "
                               "mannually read %1")
                               .arg(qPrintable(filename)));
        ccPointCloud* ccCloud = new ccPointCloud("pointCloud");
        if (ReadPointCloudFromPCD(fileName, *ccCloud)) {
            container.addChild(ccCloud);
            return CC_FERR_NO_ERROR;
        } else {
            CVLog::Warning(QString("[PCL] Failed to "
                                   "mannually read %1")
                                   .arg(qPrintable(filename)));
        }
        return CC_FERR_THIRD_PARTY_LIB_FAILURE;
    }

    pointCount = inputCloud->width * inputCloud->height;
    CVLog::Print(QString("%1: Point Count: %2")
                         .arg(qPrintable(filename))
                         .arg(pointCount));

    if (pointCount == 0) {
        return CC_FERR_NO_LOAD;
    }

    // DGM: warning, toStdString doesn't preserve "local" characters
    if (pcl::io::loadPCDFile(fileName, *inputCloud, origin, orientation) < 0) {
        CVLog::Warning(QString("[PCL] An error occurred while "
                               "pcl::io::loadPCDFile and try to "
                               "mannually read %1")
                               .arg(qPrintable(filename)));
        ccPointCloud* ccCloud = new ccPointCloud("pointCloud");
        if (ReadPointCloudFromPCD(fileName, *ccCloud)) {
            container.addChild(ccCloud);
            return CC_FERR_NO_ERROR;
        } else {
            CVLog::Warning(QString("[PCL] Failed to "
                                   "mannually read %1")
                                   .arg(qPrintable(filename)));
        }
        return CC_FERR_THIRD_PARTY_LIB_FAILURE;
    }

    // data may contain NaNs --> remove them
    if (!inputCloud->is_dense) {
        // now we need to remove NaNs
        pcl::PassThrough<PCLCloud> passFilter;
        passFilter.setInputCloud(inputCloud);
        passFilter.filter(*inputCloud);
    }
    ccPointCloud* ccCloud = pcl2cc::Convert(*inputCloud);

    // convert to CC cloud
    if (!ccCloud) {
        CVLog::Warning(
                "[PCL] An error occurred while converting PCD cloud to "
                "CloudViewer  cloud!");
        return CC_FERR_CONSOLE_ERROR;
    }
    ccCloud->setName(QStringLiteral("unnamed"));

    // now we construct a ccGBLSensor
    {
        // get orientation as rot matrix and copy it into a ccGLMatrix
        ccGLMatrix ccRot;
        {
            Eigen::Matrix3f eigrot = orientation.toRotationMatrix();
            float* X = ccRot.getColumn(0);
            float* Y = ccRot.getColumn(1);
            float* Z = ccRot.getColumn(2);

            X[0] = eigrot(0, 0);
            X[1] = eigrot(1, 0);
            X[2] = eigrot(2, 0);
            Y[0] = eigrot(0, 1);
            Y[1] = eigrot(1, 1);
            Y[2] = eigrot(2, 1);
            Z[0] = eigrot(0, 2);
            Z[1] = eigrot(1, 2);
            Z[2] = eigrot(2, 2);

            ccRot.getColumn(3)[3] = 1.0f;
            ccRot.setTranslation(origin.data());
        }

        ccGBLSensor* sensor = new ccGBLSensor;
        sensor->setRigidTransformation(ccRot);
        sensor->setYawStep(static_cast<PointCoordinateType>(0.05));
        sensor->setPitchStep(static_cast<PointCoordinateType>(0.05));
        sensor->setVisible(true);
        // uncertainty to some default
        sensor->setUncertainty(static_cast<PointCoordinateType>(0.01));
        // graphic scale
        sensor->setGraphicScale(ccCloud->getOwnBB().getDiagNorm() / 10);

        // Compute parameters
        ccGenericPointCloud* pc = ccHObjectCaster::ToGenericPointCloud(ccCloud);
        sensor->computeAutoParameters(pc);

        sensor->setEnabled(false);

        ccCloud->addChild(sensor);
    }

    container.addChild(ccCloud);

    return CC_FERR_NO_ERROR;
}
