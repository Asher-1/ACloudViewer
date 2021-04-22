// ----------------------------------------------------------------------------
// -                        cloudViewer: www.erow.cn                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.erow.cn
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

#include "PinholeCameraTrajectoryIO.h"

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <unordered_map>

#include <Console.h>
#include <FileSystem.h>
#include <IJsonConvertibleIO.h>

namespace cloudViewer {
namespace io {

bool ReadPinholeCameraTrajectoryFromJSON(
        const std::string &filename,
        camera::PinholeCameraTrajectory &trajectory) {
    return ReadIJsonConvertible(filename, trajectory);
}

bool WritePinholeCameraTrajectoryToJSON(
        const std::string &filename,
        const camera::PinholeCameraTrajectory &trajectory) {
    return WriteIJsonConvertibleToJSON(filename, trajectory);
}

static const std::unordered_map<
        std::string,
        std::function<bool(const std::string &,
                           camera::PinholeCameraTrajectory &)>>
        file_extension_to_trajectory_read_function{
                {"log", ReadPinholeCameraTrajectoryFromLOG},
                {"json", ReadPinholeCameraTrajectoryFromJSON},
                {"txt", ReadPinholeCameraTrajectoryFromTUM},
        };

static const std::unordered_map<
        std::string,
        std::function<bool(const std::string &,
                           const camera::PinholeCameraTrajectory &)>>
        file_extension_to_trajectory_write_function{
                {"log", WritePinholeCameraTrajectoryToLOG},
                {"json", WritePinholeCameraTrajectoryToJSON},
                {"txt", WritePinholeCameraTrajectoryToTUM},
        };

using namespace cloudViewer;
std::shared_ptr<camera::PinholeCameraTrajectory>
CreatePinholeCameraTrajectoryFromFile(const std::string &filename) {
    auto trajectory = cloudViewer::make_shared<camera::PinholeCameraTrajectory>();
    ReadPinholeCameraTrajectory(filename, *trajectory);
    return trajectory;
}

bool ReadPinholeCameraTrajectory(const std::string &filename,
                                 camera::PinholeCameraTrajectory &trajectory) {
    std::string filename_ext =
            utility::filesystem::GetFileExtensionInLowerCase(filename);
    if (filename_ext.empty()) {
        utility::LogWarning(
                "Read camera::PinholeCameraTrajectory failed: unknown file "
                "extension.");
        return false;
    }
    auto map_itr =
            file_extension_to_trajectory_read_function.find(filename_ext);
    if (map_itr == file_extension_to_trajectory_read_function.end()) {
        utility::LogWarning(
                "Read camera::PinholeCameraTrajectory failed: unknown file "
                "extension.");
        return false;
    }
    return map_itr->second(filename, trajectory);
}

bool WritePinholeCameraTrajectory(
        const std::string &filename,
        const camera::PinholeCameraTrajectory &trajectory) {
    std::string filename_ext =
            utility::filesystem::GetFileExtensionInLowerCase(filename);
    if (filename_ext.empty()) {
        utility::LogWarning(
                "Write camera::PinholeCameraTrajectory failed: unknown file "
                "extension.");
        return false;
    }
    auto map_itr =
            file_extension_to_trajectory_write_function.find(filename_ext);
    if (map_itr == file_extension_to_trajectory_write_function.end()) {
        utility::LogWarning(
                "Write camera::PinholeCameraTrajectory failed: unknown file "
                "extension.");
        return false;
    }
    return map_itr->second(filename, trajectory);
}

bool ReadPinholeCameraTrajectoryFromTUM(
	const std::string &filename,
	camera::PinholeCameraTrajectory &trajectory) {
	camera::PinholeCameraIntrinsic intrinsic;
	if (trajectory.parameters_.size() >= 1 &&
		trajectory.parameters_[0].intrinsic_.IsValid()) {
		intrinsic = trajectory.parameters_[0].intrinsic_;
	}
	else {
		intrinsic = camera::PinholeCameraIntrinsic(
			camera::PinholeCameraIntrinsicParameters::PrimeSenseDefault);
	}
	trajectory.parameters_.clear();
	FILE *f = utility::filesystem::FOpen(filename, "r");
	if (f == NULL) {
		utility::LogWarning("Read TUM failed: unable to open file: {}",
			filename);
		return false;
	}
	char line_buffer[DEFAULT_IO_BUFFER_SIZE];
	double ts, x, y, z, qx, qy, qz, qw;
	Eigen::Matrix4d transform;
	while (fgets(line_buffer, DEFAULT_IO_BUFFER_SIZE, f)) {
		if (strlen(line_buffer) > 0 && line_buffer[0] != '#') {
			if (sscanf(line_buffer, "%lf %lf %lf %lf %lf %lf %lf %lf", &ts, &x,
				&y, &z, &qx, &qy, &qz, &qw) != 8) {
				utility::LogWarning("Read TUM failed: unrecognized format.");
				fclose(f);
				return false;
			}

			transform.setIdentity();
			transform.topLeftCorner<3, 3>() =
				Eigen::Quaterniond(qw, qx, qy, qz).toRotationMatrix();
			transform.topRightCorner<3, 1>() = Eigen::Vector3d(x, y, z);
			auto param = camera::PinholeCameraParameters();
			param.intrinsic_ = intrinsic;
			param.extrinsic_ = transform.inverse();
			trajectory.parameters_.push_back(param);
		}
	}
	fclose(f);
	return true;
}

bool WritePinholeCameraTrajectoryToTUM(
	const std::string &filename,
	const camera::PinholeCameraTrajectory &trajectory) {
	FILE *f = utility::filesystem::FOpen(filename, "w");
	if (f == NULL) {
		utility::LogWarning("Write TUM failed: unable to open file: {}",
			filename);
		return false;
	}

	Eigen::Quaterniond q;
	fprintf(f, "# TUM trajectory, format: <t> <x> <y> <z> <qx> <qy> <qz> <qw>");
	for (size_t i = 0; i < trajectory.parameters_.size(); i++) {
		const Eigen::Matrix4d transform =
			trajectory.parameters_[i].extrinsic_.inverse();
		q = transform.topLeftCorner<3, 3>();
		fprintf(f, "%zu %lf %lf %lf %lf %lf %lf %lf\n", i, transform(0, 3),
			transform(1, 3), transform(2, 3), q.x(), q.y(), q.z(), q.w());
	}
	fclose(f);
	return true;
}

bool ReadPinholeCameraTrajectoryFromLOG(
	const std::string &filename,
	camera::PinholeCameraTrajectory &trajectory) {
	camera::PinholeCameraIntrinsic intrinsic;
	if (trajectory.parameters_.size() >= 1 &&
		trajectory.parameters_[0].intrinsic_.IsValid()) {
		intrinsic = trajectory.parameters_[0].intrinsic_;
	}
	else {
		intrinsic = camera::PinholeCameraIntrinsic(
			camera::PinholeCameraIntrinsicParameters::PrimeSenseDefault);
	}
	trajectory.parameters_.clear();
	FILE *f = utility::filesystem::FOpen(filename, "r");
	if (f == NULL) {
		utility::LogWarning("Read LOG failed: unable to open file: {}",
			filename.c_str());
		return false;
	}
	char line_buffer[DEFAULT_IO_BUFFER_SIZE];
	int i, j, k;
	int res;
	Eigen::Matrix4d trans;
	while (fgets(line_buffer, DEFAULT_IO_BUFFER_SIZE, f)) {
		if (strlen(line_buffer) > 0 && line_buffer[0] != '#') {
			if (sscanf(line_buffer, "%d %d %d", &i, &j, &k) != 3) {
				utility::LogWarning("Read LOG failed: unrecognized format.");
				fclose(f);
				return false;
			}
			if (fgets(line_buffer, DEFAULT_IO_BUFFER_SIZE, f) == 0) {
				utility::LogWarning("Read LOG failed: unrecognized format.");
				fclose(f);
				return false;
			}
			else {
				res = sscanf(line_buffer, "%lf %lf %lf %lf", &trans(0, 0),
					&trans(0, 1), &trans(0, 2), &trans(0, 3));
			}
			if (fgets(line_buffer, DEFAULT_IO_BUFFER_SIZE, f) == 0) {
				utility::LogWarning("Read LOG failed: unrecognized format.");
				fclose(f);
				return false;
			}
			else {
				res = sscanf(line_buffer, "%lf %lf %lf %lf", &trans(1, 0),
						&trans(1, 1), &trans(1, 2), &trans(1, 3));
			}
			if (fgets(line_buffer, DEFAULT_IO_BUFFER_SIZE, f) == 0) {
				utility::LogWarning("Read LOG failed: unrecognized format.");
				fclose(f);
				return false;
			}
			else {
				res = sscanf(line_buffer, "%lf %lf %lf %lf", &trans(2, 0),
						&trans(2, 1), &trans(2, 2), &trans(2, 3));
			}
			if (fgets(line_buffer, DEFAULT_IO_BUFFER_SIZE, f) == 0) {
				utility::LogWarning("Read LOG failed: unrecognized format.");
				fclose(f);
				return false;
			}
			else {
				res = sscanf(line_buffer, "%lf %lf %lf %lf", &trans(3, 0),
					&trans(3, 1), &trans(3, 2), &trans(3, 3));
			}
			auto param = camera::PinholeCameraParameters();
			param.intrinsic_ = intrinsic;
			param.extrinsic_ = trans.inverse();
			trajectory.parameters_.push_back(param);
		}
	}
	fclose(f);
	return true;
}

bool WritePinholeCameraTrajectoryToLOG(
	const std::string &filename,
	const camera::PinholeCameraTrajectory &trajectory) {
	FILE *f = utility::filesystem::FOpen(filename.c_str(), "w");
	if (f == NULL) {
		utility::LogWarning("Write LOG failed: unable to open file: {}",
			filename);
		return false;
	}
	for (size_t i = 0; i < trajectory.parameters_.size(); i++) {
		Eigen::Matrix4d_u trans =
			trajectory.parameters_[i].extrinsic_.inverse();
		fprintf(f, "%d %d %d\n", (int)i, (int)i, (int)i + 1);
		fprintf(f, "%.8f %.8f %.8f %.8f\n", trans(0, 0), trans(0, 1),
			trans(0, 2), trans(0, 3));
		fprintf(f, "%.8f %.8f %.8f %.8f\n", trans(1, 0), trans(1, 1),
			trans(1, 2), trans(1, 3));
		fprintf(f, "%.8f %.8f %.8f %.8f\n", trans(2, 0), trans(2, 1),
			trans(2, 2), trans(2, 3));
		fprintf(f, "%.8f %.8f %.8f %.8f\n", trans(3, 0), trans(3, 1),
			trans(3, 2), trans(3, 3));
	}
	fclose(f);
	return true;
}

}  // namespace io
}  // namespace cloudViewer
