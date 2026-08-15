// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "CalibConfigParser.h"

#include <CVLog.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <regex>
#include <sstream>

#include "mcalib_portability.h"

namespace mcalib {

void CalibConfigParser::skipWhitespace(const std::string& s, size_t& pos) {
    while (pos < s.size()) {
        if (std::isspace(s[pos])) {
            ++pos;
        } else if (s[pos] == '#') {
            while (pos < s.size() && s[pos] != '\n') ++pos;
        } else {
            break;
        }
    }
}

std::string CalibConfigParser::readToken(const std::string& s, size_t& pos) {
    skipWhitespace(s, pos);
    size_t start = pos;
    while (pos < s.size() && !std::isspace(s[pos]) && s[pos] != ':' &&
           s[pos] != '{' && s[pos] != '}') {
        ++pos;
    }
    return s.substr(start, pos - start);
}

std::string CalibConfigParser::readValue(const std::string& s, size_t& pos) {
    skipWhitespace(s, pos);
    if (pos < s.size() && s[pos] == '"') {
        ++pos;
        size_t start = pos;
        while (pos < s.size() && s[pos] != '"') {
            if (s[pos] == '\\') ++pos;
            ++pos;
        }
        std::string val = s.substr(start, pos - start);
        if (pos < s.size()) ++pos;
        return val;
    }

    size_t start = pos;
    while (pos < s.size() && !std::isspace(s[pos]) && s[pos] != '}' &&
           s[pos] != '#') {
        ++pos;
    }
    return s.substr(start, pos - start);
}

std::vector<CalibConfigParser::TextProtoNode> CalibConfigParser::parseBlock(
        const std::string& content, size_t& pos) {
    std::vector<TextProtoNode> nodes;

    while (pos < content.size()) {
        skipWhitespace(content, pos);
        if (pos >= content.size()) break;
        if (content[pos] == '}') {
            ++pos;
            break;
        }

        TextProtoNode node;
        node.name = readToken(content, pos);
        if (node.name.empty()) break;

        skipWhitespace(content, pos);
        if (pos < content.size() && content[pos] == ':') {
            ++pos;
            node.value = readValue(content, pos);
            node.is_message = false;
        } else if (pos < content.size() && content[pos] == '{') {
            ++pos;
            node.children = parseBlock(content, pos);
            node.is_message = true;
        } else {
            skipWhitespace(content, pos);
            if (pos < content.size() && content[pos] == '{') {
                ++pos;
                node.children = parseBlock(content, pos);
                node.is_message = true;
            }
        }

        nodes.push_back(node);
    }
    return nodes;
}

std::vector<CalibConfigParser::TextProtoNode> CalibConfigParser::parseTextProto(
        const std::string& content) {
    size_t pos = 0;
    return parseBlock(content, pos);
}

void CalibConfigParser::parseTransformation3(const TextProtoNode& node,
                                             Eigen::Isometry3d& transform) {
    Eigen::Quaterniond q(1, 0, 0, 0);
    Eigen::Vector3d t(0, 0, 0);

    for (const auto& child : node.children) {
        if ((child.name == "rotation" || child.name == "orientation") &&
            child.is_message) {
            double qx = 0, qy = 0, qz = 0, qw = 1;
            for (const auto& rc : child.children) {
                if (rc.name == "x" || rc.name == "qx")
                    qx = std::stod(rc.value);
                else if (rc.name == "y" || rc.name == "qy")
                    qy = std::stod(rc.value);
                else if (rc.name == "z" || rc.name == "qz")
                    qz = std::stod(rc.value);
                else if (rc.name == "w" || rc.name == "qw")
                    qw = std::stod(rc.value);
            }
            q = Eigen::Quaterniond(qw, qx, qy, qz);
        } else if ((child.name == "translation" || child.name == "position") &&
                   child.is_message) {
            for (const auto& tc : child.children) {
                if (tc.name == "x")
                    t.x() = std::stod(tc.value);
                else if (tc.name == "y")
                    t.y() = std::stod(tc.value);
                else if (tc.name == "z")
                    t.z() = std::stod(tc.value);
            }
        }
    }

    transform = Eigen::Isometry3d::Identity();
    transform.linear() = q.normalized().toRotationMatrix();
    transform.translation() = t;
}

void CalibConfigParser::parseIntrinsicNode(const TextProtoNode& node,
                                           CameraIntrinsic& intrinsic) {
    for (const auto& child : node.children) {
        if (child.name == "img_width")
            intrinsic.width = std::stoi(child.value);
        else if (child.name == "img_height")
            intrinsic.height = std::stoi(child.value);
        else if (child.name == "f_x")
            intrinsic.fx = std::stod(child.value);
        else if (child.name == "f_y")
            intrinsic.fy = std::stod(child.value);
        else if (child.name == "o_x")
            intrinsic.cx = std::stod(child.value);
        else if (child.name == "o_y")
            intrinsic.cy = std::stod(child.value);
        else if (child.name == "k_1")
            intrinsic.k1 = std::stod(child.value);
        else if (child.name == "k_2")
            intrinsic.k2 = std::stod(child.value);
        else if (child.name == "k_3")
            intrinsic.k3 = std::stod(child.value);
        else if (child.name == "k_4")
            intrinsic.k4 = std::stod(child.value);
        else if (child.name == "k_5")
            intrinsic.k5 = std::stod(child.value);
        else if (child.name == "k_6")
            intrinsic.k6 = std::stod(child.value);
        else if (child.name == "p_1")
            intrinsic.p1 = std::stod(child.value);
        else if (child.name == "p_2")
            intrinsic.p2 = std::stod(child.value);
        else if (child.name == "xi")
            intrinsic.xi = std::stod(child.value);
        else if (child.name == "model_type") {
            if (child.value == "PINHOLE" || child.value == "0")
                intrinsic.model_type = CameraIntrinsic::PINHOLE;
            else if (child.value == "KANNALA_BRANDT" || child.value == "1")
                intrinsic.model_type = CameraIntrinsic::KANNALA_BRANDT;
            else if (child.value == "MEI" || child.value == "2")
                intrinsic.model_type = CameraIntrinsic::MEI;
            else if (child.value == "FULLPINHOLE" || child.value == "3")
                intrinsic.model_type = CameraIntrinsic::FULLPINHOLE;
        }
    }
}

void CalibConfigParser::parseExtrinsicNode(const TextProtoNode& node,
                                           Eigen::Isometry3d& extrinsic) {
    for (const auto& child : node.children) {
        if ((child.name == "sensor_to_camera" ||
             child.name == "sensor_to_cam" ||
             child.name == "sensor_to_lidar") &&
            child.is_message) {
            parseTransformation3(child, extrinsic);
            return;
        }
    }
}

void CalibConfigParser::parseCameraConfigNode(const TextProtoNode& node,
                                              CameraSensorConfig& cam_config) {
    for (const auto& child : node.children) {
        if (child.name == "frame_id" || child.name == "camera_dev") {
            cam_config.frame_id = child.value;
        } else if (child.name == "width" && cam_config.intrinsic.width == 0) {
            cam_config.intrinsic.width = std::stoi(child.value);
        } else if (child.name == "height" && cam_config.intrinsic.height == 0) {
            cam_config.intrinsic.height = std::stoi(child.value);
        } else if (child.name == "parameters" && child.is_message) {
            for (const auto& param : child.children) {
                if (param.name == "extrinsic" && param.is_message) {
                    parseExtrinsicNode(param, cam_config.extrinsic);
                } else if (param.name == "intrinsic" && param.is_message) {
                    parseIntrinsicNode(param, cam_config.intrinsic);
                }
            }
        }
    }
}

void CalibConfigParser::parseLidarConfigNode(const TextProtoNode& node,
                                             LidarSensorConfig& lidar_config) {
    for (const auto& child : node.children) {
        if (child.name == "lidar_idx") {
            lidar_config.lidar_idx = std::stoi(child.value);
        } else if (child.name == "frame_id") {
            lidar_config.frame_id = child.value;
        } else if (child.name == "model") {
            lidar_config.model = child.value;
        } else if (child.name == "ring_start" ||
                   child.name == "ring_id_start") {
            lidar_config.ring_start = std::stoi(child.value);
        } else if (child.name == "ring_end" || child.name == "ring_id_end") {
            lidar_config.ring_end = std::stoi(child.value);
        } else if (child.name == "extrinsic" && child.is_message) {
            parseExtrinsicNode(child, lidar_config.extrinsic);
        } else if ((child.name == "sensor_to_lidar") && child.is_message) {
            parseTransformation3(child, lidar_config.extrinsic);
        }
    }
}

bool CalibConfigParser::loadCameraConfig(const std::string& config_file,
                                         VehicleCalibConfig& config) {
    std::ifstream ifs;
    if (!mcalib::openInputFile(ifs, config_file)) {
        CVLog::Warning("[CalibConfigParser] Cannot open: %s",
                       config_file.c_str());
        return false;
    }

    std::string content((std::istreambuf_iterator<char>(ifs)),
                        std::istreambuf_iterator<char>());
    ifs.close();

    config.camera_cfg_path = config_file;
    config.camera_cfg_raw = content;

    auto nodes = parseTextProto(content);

    for (const auto& node : nodes) {
        if (node.name == "config" && node.is_message) {
            CameraSensorConfig cam_config;
            parseCameraConfigNode(node, cam_config);
            if (!cam_config.frame_id.empty()) {
                config.cameras[cam_config.frame_id] = cam_config;
                CVLog::Print("[CalibConfigParser] camera '%s': %dx%d, model=%d",
                             cam_config.frame_id.c_str(),
                             cam_config.intrinsic.width,
                             cam_config.intrinsic.height,
                             static_cast<int>(cam_config.intrinsic.model_type));
            } else {
                CVLog::Warning(
                        "[CalibConfigParser] skipped camera node with empty "
                        "frame_id");
            }
        }
    }

    if (config.cameras.empty()) {
        CVLog::Warning("[CalibConfigParser] No camera configs found in: %s",
                       config_file.c_str());
    } else {
        CVLog::Print("[CalibConfigParser] Loaded %zu cameras from: %s",
                     config.cameras.size(), config_file.c_str());
    }

    return !config.cameras.empty();
}

bool CalibConfigParser::loadLidarConfig(const std::string& config_file,
                                        VehicleCalibConfig& config) {
    std::ifstream ifs;
    if (!mcalib::openInputFile(ifs, config_file)) {
        CVLog::Print("[CalibConfigParser] Cannot open: %s",
                     config_file.c_str());
        return false;
    }

    std::string content((std::istreambuf_iterator<char>(ifs)),
                        std::istreambuf_iterator<char>());
    ifs.close();

    config.lidar_cfg_path = config_file;
    config.lidar_cfg_raw = content;

    auto nodes = parseTextProto(content);

    for (const auto& node : nodes) {
        if (node.name == "config" && node.is_message) {
            LidarSensorConfig lidar;
            parseLidarConfigNode(node, lidar);
            config.lidars.push_back(lidar);
        } else if ((node.name == "extrinsic_vehicle_sensing" ||
                    node.name == "vehicle_to_sensing") &&
                   node.is_message) {
            parseTransformation3(node, config.iso_sensing_vehicle);
        }
    }
    return true;
}

bool CalibConfigParser::loadGroundConfig(const std::string& config_file,
                                         VehicleCalibConfig& config) {
    std::ifstream ifs;
    if (!mcalib::openInputFile(ifs, config_file)) {
        CVLog::Print("[CalibConfigParser] Cannot open ground config: %s",
                     config_file.c_str());
        return false;
    }

    std::string content((std::istreambuf_iterator<char>(ifs)),
                        std::istreambuf_iterator<char>());
    ifs.close();

    auto nodes = parseTextProto(content);

    for (const auto& node : nodes) {
        if (node.name == "ground_in_sensing" && node.is_message) {
            for (const auto& child : node.children) {
                if (child.name == "a")
                    config.ground.a = std::stod(child.value);
                else if (child.name == "b")
                    config.ground.b = std::stod(child.value);
                else if (child.name == "c")
                    config.ground.c = std::stod(child.value);
                else if (child.name == "d")
                    config.ground.d = std::stod(child.value);
            }
        }
    }
    return true;
}

bool CalibConfigParser::loadConfigDirectory(const std::string& dir,
                                            VehicleCalibConfig& config) {
    config = VehicleCalibConfig{};
    const std::string camera_cfg = dir + "/cameras.cfg";
    const std::string lidar_cfg = dir + "/lidars.cfg";
    const std::string ground_cfg = dir + "/ground.cfg";

    auto fileExists = [](const std::string& path) {
        std::ifstream ifs;
        return mcalib::openInputFile(ifs, path);
    };

    if (!fileExists(camera_cfg) || !fileExists(lidar_cfg) ||
        !fileExists(ground_cfg)) {
        CVLog::Warning(
                "[CalibConfigParser] Config directory must contain "
                "cameras.cfg, lidars.cfg and ground.cfg: %s",
                dir.c_str());
        return false;
    }

    if (!loadCameraConfig(camera_cfg, config)) return false;
    if (!loadLidarConfig(lidar_cfg, config)) return false;
    if (!loadGroundConfig(ground_cfg, config)) return false;
    return true;
}

void CalibConfigParser::alignCameraSizes(VehicleCalibConfig& config) {
    cv::Size target_size(1920, 1080);
    for (const auto& [name, cam] : config.cameras) {
        if (name == "camera_2" && cam.intrinsic.width > 0 &&
            cam.intrinsic.height > 0) {
            target_size = cv::Size(cam.intrinsic.width, cam.intrinsic.height);
            break;
        }
    }

    for (auto& [name, cam] : config.cameras) {
        if (name.compare(0, 3, "pan") == 0) continue;
        int w = cam.intrinsic.width;
        int h = cam.intrinsic.height;
        if (w <= 0 || h <= 0) continue;
        if (w == target_size.width && h == target_size.height) continue;

        double sx = static_cast<double>(target_size.width) / w;
        double sy = static_cast<double>(target_size.height) / h;
        cam.intrinsic.width = target_size.width;
        cam.intrinsic.height = target_size.height;
        cam.intrinsic.cx *= sx;
        cam.intrinsic.cy *= sy;
        cam.intrinsic.fx *= sx;
        cam.intrinsic.fy *= sy;
    }
    CVLog::Print("[CalibConfigParser] alignCameraSizes: target=%dx%d",
                 target_size.width, target_size.height);
}

bool CalibConfigParser::getCameraIntrinsic(const VehicleCalibConfig& config,
                                           const std::string& camera_name,
                                           cv::Mat& camera_matrix,
                                           cv::Mat& distor_coeffs,
                                           cv::Size& img_size) {
    auto it = config.cameras.find(camera_name);
    if (it == config.cameras.end()) return false;

    const auto& intr = it->second.intrinsic;
    camera_matrix = intr.getCameraMatrix();
    distor_coeffs = intr.getDistCoeffs();
    img_size = intr.getImageSize();
    return true;
}

bool CalibConfigParser::getCameraExtrinsics(
        const VehicleCalibConfig& config,
        std::map<std::string, Eigen::Isometry3d>& extrinsics) {
    for (const auto& [name, cam] : config.cameras) {
        extrinsics[name] = cam.extrinsic;
    }
    return !extrinsics.empty();
}

void CalibConfigParser::updateCameraExtrinsic(
        VehicleCalibConfig& config,
        const std::string& camera_name,
        const Eigen::Isometry3d& extrinsic) {
    auto it = config.cameras.find(camera_name);
    if (it != config.cameras.end()) {
        it->second.extrinsic = extrinsic;
    }
}

std::string CalibConfigParser::serializeCameraConfig(
        const VehicleCalibConfig& config) {
    std::ostringstream oss;
    oss << std::fixed;

    for (const auto& [name, cam] : config.cameras) {
        oss << "config {\n";
        oss << "  camera_dev: \"" << cam.frame_id << "\"\n";
        oss << "  parameters {\n";

        oss << "    extrinsic {\n";
        oss << "      sensor_to_cam {\n";
        Eigen::Quaterniond q(cam.extrinsic.linear());
        oss << "        position {\n";
        oss << "          x: " << cam.extrinsic.translation().x() << "\n";
        oss << "          y: " << cam.extrinsic.translation().y() << "\n";
        oss << "          z: " << cam.extrinsic.translation().z() << "\n";
        oss << "        }\n";
        oss << "        orientation {\n";
        oss << "          qx: " << q.x() << "\n";
        oss << "          qy: " << q.y() << "\n";
        oss << "          qz: " << q.z() << "\n";
        oss << "          qw: " << q.w() << "\n";
        oss << "        }\n";
        oss << "      }\n";
        oss << "    }\n";

        const auto& intr = cam.intrinsic;
        oss << "    intrinsic {\n";
        oss << "      img_width: " << intr.width << "\n";
        oss << "      img_height: " << intr.height << "\n";
        oss << "      f_x: " << intr.fx << "\n";
        oss << "      f_y: " << intr.fy << "\n";
        oss << "      o_x: " << intr.cx << "\n";
        oss << "      o_y: " << intr.cy << "\n";
        oss << "      k_1: " << intr.k1 << "\n";
        oss << "      k_2: " << intr.k2 << "\n";
        oss << "      k_3: " << intr.k3 << "\n";
        oss << "      k_4: " << intr.k4 << "\n";
        oss << "      k_5: " << intr.k5 << "\n";
        oss << "      k_6: " << intr.k6 << "\n";
        oss << "      p_1: " << intr.p1 << "\n";
        oss << "      p_2: " << intr.p2 << "\n";
        std::string model;
        switch (intr.model_type) {
            case CameraIntrinsic::PINHOLE:
                model = "PINHOLE";
                break;
            case CameraIntrinsic::KANNALA_BRANDT:
                model = "KANNALA_BRANDT";
                break;
            case CameraIntrinsic::MEI:
                model = "MEI";
                break;
            case CameraIntrinsic::FULLPINHOLE:
                model = "FULLPINHOLE";
                break;
        }
        oss << "      model_type: " << model << "\n";
        oss << "    }\n";

        oss << "  }\n";
        oss << "}\n";
    }
    return oss.str();
}

bool CalibConfigParser::saveCameraConfig(const std::string& config_file,
                                         const VehicleCalibConfig& config) {
    std::ofstream ofs;
    if (!mcalib::openOutputFile(ofs, config_file)) {
        CVLog::Error("[CalibConfigParser] Cannot write: %s",
                     config_file.c_str());
        return false;
    }
    ofs << serializeCameraConfig(config);
    return true;
}

bool CalibConfigParser::saveLidarConfig(const std::string& config_file,
                                        const VehicleCalibConfig& config) {
    std::ofstream ofs;
    if (!mcalib::openOutputFile(ofs, config_file)) {
        CVLog::Error("[CalibConfigParser] Cannot write: %s",
                     config_file.c_str());
        return false;
    }

    ofs << std::fixed << std::setprecision(12);

    auto writeTransform = [&](const Eigen::Isometry3d& iso) {
        ofs << "  extrinsic_parameter {\n";
        ofs << "    transformation3 {\n";
        Eigen::Vector3d euler = iso.rotation().eulerAngles(2, 1, 0);
        ofs << "      rotation { x: " << euler(2) << "  y: " << euler(1)
            << "  z: " << euler(0) << " }\n";
        ofs << "      translation { x: " << iso.translation().x()
            << "  y: " << iso.translation().y()
            << "  z: " << iso.translation().z() << " }\n";
        ofs << "    }\n";
        ofs << "  }\n";
    };

    for (const auto& lc : config.lidars) {
        ofs << "lidar_config {\n";
        ofs << "  frame_id: \"" << lc.frame_id << "\"\n";
        ofs << "  lidar_idx: " << lc.lidar_idx << "\n";
        ofs << "  ring_start: " << lc.ring_start << "\n";
        ofs << "  ring_end: " << lc.ring_end << "\n";
        writeTransform(lc.extrinsic);
        ofs << "}\n\n";
    }

    {
        ofs << "extrinsic_sensing_vehicle {\n";
        ofs << "  transformation3 {\n";
        Eigen::Vector3d euler =
                config.iso_sensing_vehicle.rotation().eulerAngles(2, 1, 0);
        ofs << "    rotation { x: " << euler(2) << "  y: " << euler(1)
            << "  z: " << euler(0) << " }\n";
        ofs << "    translation { x: "
            << config.iso_sensing_vehicle.translation().x()
            << "  y: " << config.iso_sensing_vehicle.translation().y()
            << "  z: " << config.iso_sensing_vehicle.translation().z()
            << " }\n";
        ofs << "  }\n";
        ofs << "}\n";
    }

    return true;
}

namespace {

std::string makeStampString() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t tt = std::chrono::system_clock::to_time_t(now);
    const std::tm local_tm = *std::localtime(&tt);
    return std::to_string(local_tm.tm_year + 1900) + "/" +
           std::to_string(local_tm.tm_mon + 1) + "/" +
           std::to_string(local_tm.tm_mday) + "/" +
           std::to_string(local_tm.tm_hour) + ":" +
           std::to_string(local_tm.tm_min) + ":" +
           std::to_string(local_tm.tm_sec);
}

void writeQuaternionTransform(std::ostream& ofs, const Eigen::Isometry3d& iso) {
    const Eigen::Quaterniond q(iso.linear());
    const Eigen::Vector3d t = iso.translation();
    ofs << "    position {\n";
    ofs << "      x: " << t.x() << "\n";
    ofs << "      y: " << t.y() << "\n";
    ofs << "      z: " << t.z() << "\n";
    ofs << "    }\n";
    ofs << "    orientation {\n";
    ofs << "      qx: " << q.x() << "\n";
    ofs << "      qy: " << q.y() << "\n";
    ofs << "      qz: " << q.z() << "\n";
    ofs << "      qw: " << q.w() << "\n";
    ofs << "    }\n";
}

size_t findMatchingBrace(const std::string& text, size_t open_brace_pos) {
    int depth = 0;
    for (size_t i = open_brace_pos; i < text.size(); ++i) {
        if (text[i] == '{') {
            ++depth;
        } else if (text[i] == '}') {
            --depth;
            if (depth == 0) return i;
        }
    }
    return std::string::npos;
}

std::string extractCameraNameFromBlock(const std::string& block) {
    static const std::regex frame_re(
            R"((?:frame_id|camera_dev)\s*:\s*\"([^\"]+)\")");
    std::smatch match;
    if (std::regex_search(block, match, frame_re) && match.size() > 1) {
        return match[1].str();
    }
    return {};
}

std::string extractLidarNameFromBlock(const std::string& block) {
    static const std::regex frame_re(R"(frame_id\s*:\s*\"([^\"]+)\")");
    std::smatch match;
    if (std::regex_search(block, match, frame_re) && match.size() > 1) {
        return match[1].str();
    }
    return {};
}

std::string formatSensorToCamBlock(const Eigen::Isometry3d& iso_sensing_cam) {
    const Eigen::Quaterniond q(iso_sensing_cam.linear());
    const Eigen::Vector3d t = iso_sensing_cam.translation();
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(12);
    oss << "sensor_to_cam {\n";
    oss << "        position {\n";
    oss << "          x: " << t.x() << "\n";
    oss << "          y: " << t.y() << "\n";
    oss << "          z: " << t.z() << "\n";
    oss << "        }\n";
    oss << "        orientation {\n";
    oss << "          qx: " << q.x() << "\n";
    oss << "          qy: " << q.y() << "\n";
    oss << "          qz: " << q.z() << "\n";
    oss << "          qw: " << q.w() << "\n";
    oss << "        }\n";
    oss << "      }";
    return oss.str();
}

std::string formatSensorToLidarBlock(
        const Eigen::Isometry3d& iso_sensing_lidar) {
    const Eigen::Quaterniond q(iso_sensing_lidar.linear());
    const Eigen::Vector3d t = iso_sensing_lidar.translation();
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(12);
    oss << "sensor_to_lidar {\n";
    oss << "    position {\n";
    oss << "      x: " << t.x() << "\n";
    oss << "      y: " << t.y() << "\n";
    oss << "      z: " << t.z() << "\n";
    oss << "    }\n";
    oss << "    orientation {\n";
    oss << "      qx: " << q.x() << "\n";
    oss << "      qy: " << q.y() << "\n";
    oss << "      qz: " << q.z() << "\n";
    oss << "      qw: " << q.w() << "\n";
    oss << "    }\n";
    oss << "  }";
    return oss.str();
}

bool replaceNamedTransformBlock(std::string& text,
                                size_t search_from,
                                const std::string& block_name,
                                const std::string& replacement) {
    const size_t pos = text.find(block_name, search_from);
    if (pos == std::string::npos) return false;

    const size_t brace_pos = text.find('{', pos + block_name.size());
    if (brace_pos == std::string::npos) return false;

    const size_t end_pos = findMatchingBrace(text, brace_pos);
    if (end_pos == std::string::npos) return false;

    text.replace(pos, end_pos - pos + 1, replacement);
    return true;
}

std::string patchCameraConfigRaw(const std::string& raw,
                                 const std::map<std::string, Eigen::Isometry3d>&
                                         extrinsics_cam_sensing) {
    std::string output = raw;
    size_t search_pos = 0;
    while (true) {
        const size_t block_pos = output.find("config", search_pos);
        if (block_pos == std::string::npos) break;

        const size_t brace_pos = output.find('{', block_pos);
        if (brace_pos == std::string::npos) break;

        const size_t end_pos = findMatchingBrace(output, brace_pos);
        if (end_pos == std::string::npos) break;

        const std::string block =
                output.substr(block_pos, end_pos - block_pos + 1);
        const std::string camera_name = extractCameraNameFromBlock(block);
        auto it = extrinsics_cam_sensing.find(camera_name);
        if (!camera_name.empty() && it != extrinsics_cam_sensing.end()) {
            const Eigen::Isometry3d iso_sensing_cam = it->second.inverse();
            const std::string replacement =
                    formatSensorToCamBlock(iso_sensing_cam);
            std::string patched_block = block;
            if (!replaceNamedTransformBlock(patched_block, 0, "sensor_to_cam",
                                            replacement) &&
                !replaceNamedTransformBlock(patched_block, 0,
                                            "sensor_to_camera", replacement)) {
                CVLog::Warning(
                        "[CalibConfigParser] camera '%s': sensor_to_cam block "
                        "not found",
                        camera_name.c_str());
            } else {
                output.replace(block_pos, end_pos - block_pos + 1,
                               patched_block);
            }
        }
        search_pos = block_pos + 6;
    }
    return output;
}

std::string patchLidarConfigRaw(
        const std::string& raw,
        const VehicleCalibConfig& config,
        const std::vector<Eigen::Isometry3d>& extrinsics_gnss_lidar) {
    std::string output = raw;
    size_t search_pos = 0;
    while (true) {
        const size_t block_pos = output.find("config", search_pos);
        if (block_pos == std::string::npos) break;

        const size_t brace_pos = output.find('{', block_pos);
        if (brace_pos == std::string::npos) break;

        const size_t end_pos = findMatchingBrace(output, brace_pos);
        if (end_pos == std::string::npos) break;

        const std::string block =
                output.substr(block_pos, end_pos - block_pos + 1);
        const std::string frame_id = extractLidarNameFromBlock(block);
        int lidar_index = -1;
        for (size_t i = 0; i < config.lidars.size(); ++i) {
            if (config.lidars[i].frame_id == frame_id) {
                lidar_index = static_cast<int>(i);
                break;
            }
        }
        if (lidar_index >= 0 &&
            lidar_index < static_cast<int>(extrinsics_gnss_lidar.size())) {
            const Eigen::Isometry3d iso_sensing_lidar =
                    config.iso_sensing_vehicle *
                    extrinsics_gnss_lidar[lidar_index];
            const std::string replacement =
                    formatSensorToLidarBlock(iso_sensing_lidar);
            std::string patched_block = block;
            if (replaceNamedTransformBlock(patched_block, 0, "sensor_to_lidar",
                                           replacement)) {
                output.replace(block_pos, end_pos - block_pos + 1,
                               patched_block);
            }
        }
        search_pos = block_pos + 6;
    }
    return output;
}

}  // namespace

bool CalibConfigParser::saveGnssMultiLidarExtrinsic(
        const std::string& config_file,
        const std::vector<Eigen::Isometry3d>& extrinsics_gnss_lidar,
        const VehicleCalibConfig& config,
        const std::string& car_id) {
    if (config.lidars.empty() || extrinsics_gnss_lidar.empty()) {
        CVLog::Warning(
                "[CalibConfigParser] saveGnssMultiLidarExtrinsic: no lidars");
        return false;
    }

    if (!config.lidar_cfg_raw.empty()) {
        const std::string body = patchLidarConfigRaw(
                config.lidar_cfg_raw, config, extrinsics_gnss_lidar);
        std::ostringstream out;
        std::string id = car_id.empty() ? "manual_calib" : car_id;
        out << "## " << id << "\n";
        out << "## " << makeStampString() << "\n\n";
        out << body;

        std::ofstream ofs;
        if (!mcalib::openOutputFile(ofs, config_file)) {
            CVLog::Error("[CalibConfigParser] Cannot write: %s",
                         config_file.c_str());
            return false;
        }
        ofs << out.str();
        return true;
    }

    std::ostringstream body;
    body << std::fixed << std::setprecision(12);

    body << "vehicle_to_sensing {\n";
    writeQuaternionTransform(body, config.iso_sensing_vehicle);
    body << "}\n";

    std::vector<int> sorted_idx;
    sorted_idx.reserve(config.lidars.size());
    for (size_t i = 0; i < config.lidars.size(); ++i) {
        sorted_idx.push_back(static_cast<int>(config.lidars[i].lidar_idx));
    }
    std::sort(sorted_idx.begin(), sorted_idx.end());

    for (int sorted_id : sorted_idx) {
        int k = -1;
        for (size_t i = 0; i < config.lidars.size(); ++i) {
            if (config.lidars[i].lidar_idx == sorted_id) {
                k = static_cast<int>(i);
                break;
            }
        }
        if (k < 0 || k >= static_cast<int>(extrinsics_gnss_lidar.size()))
            continue;

        const auto& lc = config.lidars[k];
        const Eigen::Isometry3d iso_sensing_lidar =
                config.iso_sensing_vehicle * extrinsics_gnss_lidar[k];

        body << "config {\n";
        body << "  frame_id: \"" << lc.frame_id << "\"\n";
        if (!lc.model.empty()) {
            body << "  model: " << lc.model << "\n";
        }
        body << "  sensor_to_lidar {\n";
        writeQuaternionTransform(body, iso_sensing_lidar);
        body << "  }\n";
        body << "  ring_id_start: " << lc.ring_start << "\n";
        body << "  ring_id_end: " << lc.ring_end << "\n";
        body << "  install_angle_error {\n";
        body << "    x: 0\n";
        body << "    y: 0\n";
        body << "    z: 0\n";
        body << "  }\n";
        body << "}\n";
    }

    std::string id = car_id.empty() ? "manual_calib" : car_id;
    std::ostringstream out;
    out << "## " << id << "\n";
    out << "## " << makeStampString() << "\n\n";
    out << body.str();

    std::ofstream ofs;
    if (!mcalib::openOutputFile(ofs, config_file)) {
        CVLog::Error("[CalibConfigParser] Cannot write: %s",
                     config_file.c_str());
        return false;
    }
    ofs << out.str();
    return true;
}

bool CalibConfigParser::saveMultiCameraExtrinsic(
        const std::string& output_file,
        const std::map<std::string, Eigen::Isometry3d>& extrinsics_cam_sensing,
        const VehicleCalibConfig& config) {
    std::string body;
    if (!config.camera_cfg_raw.empty()) {
        body = patchCameraConfigRaw(config.camera_cfg_raw,
                                    extrinsics_cam_sensing);
    } else {
        body = serializeCameraConfig(config);
    }

    std::ostringstream out;
    out << "## manual_calib\n";
    out << "## " << makeStampString() << "\n\n";
    out << body;

    std::ofstream ofs;
    if (!mcalib::openOutputFile(ofs, output_file)) {
        CVLog::Error("[CalibConfigParser] Cannot write: %s",
                     output_file.c_str());
        return false;
    }
    ofs << out.str();
    CVLog::Print("[CalibConfigParser] saveMultiCameraExtrinsic: %s",
                 output_file.c_str());
    return true;
}

}  // namespace mcalib
