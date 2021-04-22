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

#include "VoxelGridIO.h"

#include <unordered_map>

#include <Console.h>
#include <FileSystem.h>

#include <rply.h>

namespace cloudViewer {

namespace {
using namespace io;

namespace ply_voxelgrid_reader {

	struct PLYReaderState {
		cloudViewer::utility::ConsoleProgressBar *progress_bar;
		std::vector<geometry::Voxel> *voxelgrid_ptr;
		Eigen::Vector3d origin;
		double voxel_size;
		long voxel_index;
		long voxel_num;
		long color_index;
		long color_num;
	};

	int ReadOriginCallback(p_ply_argument argument) {
		PLYReaderState *state_ptr;
		long index;
		ply_get_argument_user_data(argument, reinterpret_cast<void **>(&state_ptr),
			&index);

		double value = ply_get_argument_value(argument);
		state_ptr->origin(index) = value;
		return 1;
	}

	int ReadScaleCallback(p_ply_argument argument) {
		PLYReaderState *state_ptr;
		long index;
		ply_get_argument_user_data(argument, 
			reinterpret_cast<void **>(&state_ptr), &index);

		double value = ply_get_argument_value(argument);
		state_ptr->voxel_size = value;
		return 1;
	}

	int ReadVoxelCallback(p_ply_argument argument) {
		PLYReaderState *state_ptr;
		long index;
		ply_get_argument_user_data(argument, reinterpret_cast<void **>(&state_ptr),
			&index);
		if (state_ptr->voxel_index >= state_ptr->voxel_num) {
			return 0;  // some sanity check
		}

		double value = ply_get_argument_value(argument);
		auto &ptr = *(state_ptr->voxelgrid_ptr);
		ptr[state_ptr->voxel_index].grid_index_(index) = int(value);
		if (index == 2) {  // reading 'z'
			state_ptr->voxel_index++;
			++(*state_ptr->progress_bar);
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
		auto &ptr = *(state_ptr->voxelgrid_ptr);
		ptr[state_ptr->color_index].color_(index) = value / 255.0;
		if (index == 2) {  // reading 'blue'
			state_ptr->color_index++;
			++(*state_ptr->progress_bar);
		}
		return 1;
	}

}  // namespace ply_voxelgrid_reader

static const std::unordered_map<
        std::string,
        std::function<bool(const std::string &, geometry::VoxelGrid &, bool)>>
        file_extension_to_voxelgrid_read_function {
                {"ply", ReadVoxelGridFromPLY},
        };

static const std::unordered_map<std::string,
                                std::function<bool(const std::string &,
                                                   const geometry::VoxelGrid &,
                                                   const bool,
                                                   const bool,
                                                   const bool)>>
        file_extension_to_voxelgrid_write_function{
                {"ply", WriteVoxelGridToPLY},
        };
}  // unnamed namespace

namespace io {

using namespace cloudViewer;

std::shared_ptr<geometry::VoxelGrid> CreateVoxelGridFromFile(
        const std::string &filename,
        const std::string &format,
        bool print_progress) {
    auto voxelgrid = cloudViewer::make_shared<geometry::VoxelGrid>();
    ReadVoxelGrid(filename, *voxelgrid, format, print_progress);
    return voxelgrid;
}

bool ReadVoxelGrid(const std::string &filename,
                   geometry::VoxelGrid &voxelgrid,
                   const std::string &format,
                   bool print_progress) {
    std::string filename_ext;
    if (format == "auto") {
        filename_ext =
                utility::filesystem::GetFileExtensionInLowerCase(filename);
    } else {
        filename_ext = format;
    }
    if (filename_ext.empty()) {
        utility::LogWarning(
                "Read geometry::VoxelGrid failed: unknown file extension.");
        return false;
    }
    auto map_itr = file_extension_to_voxelgrid_read_function.find(filename_ext);
    if (map_itr == file_extension_to_voxelgrid_read_function.end()) {
        utility::LogWarning(
                "Read geometry::VoxelGrid failed: unknown file extension.");
        return false;
    }
    bool success = map_itr->second(filename, voxelgrid, print_progress);
    utility::LogDebug("Read geometry::VoxelGrid: {:d} voxels.",
                      (int)voxelgrid.voxels_.size());
    return success;
}

bool WriteVoxelGrid(const std::string &filename,
                    const geometry::VoxelGrid &voxelgrid,
                    bool write_ascii /* = false*/,
                    bool compressed /* = false*/,
                    bool print_progress) {
    std::string filename_ext =
            utility::filesystem::GetFileExtensionInLowerCase(filename);
    if (filename_ext.empty()) {
        utility::LogWarning(
                "Write geometry::VoxelGrid failed: unknown file extension.");
        return false;
    }
    auto map_itr =
            file_extension_to_voxelgrid_write_function.find(filename_ext);
    if (map_itr == file_extension_to_voxelgrid_write_function.end()) {
        utility::LogWarning(
                "Write geometry::VoxelGrid failed: unknown file extension.");
        return false;
    }
    bool success = map_itr->second(filename, voxelgrid, write_ascii, compressed,
                                   print_progress);
    utility::LogDebug("Write geometry::VoxelGrid: {:d} voxels.",
                      (int)voxelgrid.voxels_.size());
    return success;
}

bool ReadVoxelGridFromPLY(const std::string &filename,
	geometry::VoxelGrid &voxelgrid,
	bool print_progress) {
	using namespace ply_voxelgrid_reader;

    p_ply ply_file = ply_open(filename.c_str(), nullptr, 0, nullptr);
	if (!ply_file) {
		utility::LogWarning("Read PLY failed: unable to open file: {}",
			filename);
		return false;
	}
	if (!ply_read_header(ply_file)) {
		utility::LogWarning("Read PLY failed: unable to parse header.");
		ply_close(ply_file);
		return false;
	}

	PLYReaderState state;
	std::vector<geometry::Voxel> voxelgrid_ptr;
	state.voxelgrid_ptr = &voxelgrid_ptr;
	state.voxel_num = ply_set_read_cb(ply_file, "vertex", "x",
		ReadVoxelCallback, &state, 0);
	ply_set_read_cb(ply_file, "vertex", "y", ReadVoxelCallback, &state, 1);
	ply_set_read_cb(ply_file, "vertex", "z", ReadVoxelCallback, &state, 2);

	if (state.voxel_num <= 0) {
		utility::LogWarning("Read PLY failed: number of vertex <= 0.");
		ply_close(ply_file);
		return false;
	}

	state.color_num = ply_set_read_cb(ply_file, "vertex", "red",
		ReadColorCallback, &state, 0);
	ply_set_read_cb(ply_file, "vertex", "green", ReadColorCallback, &state, 1);
	ply_set_read_cb(ply_file, "vertex", "blue", ReadColorCallback, &state, 2);

	ply_set_read_cb(ply_file, "origin", "x", ReadOriginCallback, &state, 0);
	ply_set_read_cb(ply_file, "origin", "y", ReadOriginCallback, &state, 1);
	ply_set_read_cb(ply_file, "origin", "z", ReadOriginCallback, &state, 2);
	ply_set_read_cb(ply_file, "voxel_size", "val", ReadScaleCallback, &state,
		0);

	state.voxel_index = 0;
	state.color_index = 0;

	voxelgrid_ptr.clear();
	voxelgrid_ptr.resize(state.voxel_num);
	utility::ConsoleProgressBar progress_bar(state.voxel_num + state.color_num,
		"Reading PLY: ", print_progress);
	state.progress_bar = &progress_bar;

	if (!ply_read(ply_file)) {
		utility::LogWarning("Read PLY failed: unable to read file: {}",
			filename);
		ply_close(ply_file);
		return false;
	}

	voxelgrid.Clear();
	for (auto &it : voxelgrid_ptr) {
		if (state.color_num > 0)
			voxelgrid.AddVoxel(geometry::Voxel(it.grid_index_, it.color_));
		else
			voxelgrid.AddVoxel(geometry::Voxel(it.grid_index_));
	}
	voxelgrid.origin_ = state.origin;
	voxelgrid.voxel_size_ = state.voxel_size;

	ply_close(ply_file);
	return true;
}

bool WriteVoxelGridToPLY(const std::string &filename,
	const geometry::VoxelGrid &voxelgrid,
	bool write_ascii /* = false*/,
	bool compressed /* = false*/,
	bool print_progress) {
	if (voxelgrid.isEmpty()) {
		utility::LogWarning("Write PLY failed: voxelgrid has 0 voxels.");
		return false;
	}

	p_ply ply_file = ply_create(filename.c_str(),
		write_ascii ? PLY_ASCII : PLY_LITTLE_ENDIAN,
        nullptr, 0, nullptr);
	if (!ply_file) {
		utility::LogWarning("Write PLY failed: unable to open file: {}",
			filename);
		return false;
	}
	ply_add_comment(ply_file, "Created by cloudViewer");
	ply_add_element(ply_file, "origin", 1);
	ply_add_property(ply_file, "x", PLY_DOUBLE, PLY_DOUBLE, PLY_DOUBLE);
	ply_add_property(ply_file, "y", PLY_DOUBLE, PLY_DOUBLE, PLY_DOUBLE);
	ply_add_property(ply_file, "z", PLY_DOUBLE, PLY_DOUBLE, PLY_DOUBLE);
	ply_add_element(ply_file, "voxel_size", 1);
	ply_add_property(ply_file, "val", PLY_DOUBLE, PLY_DOUBLE, PLY_DOUBLE);

	ply_add_element(ply_file, "vertex",
		static_cast<long>(voxelgrid.voxels_.size()));
	// PLY_UINT could be used for x, y, z but PLY_DOUBLE used instead due to
	// compatibility issue.
	ply_add_property(ply_file, "x", PLY_DOUBLE, PLY_DOUBLE, PLY_DOUBLE);
	ply_add_property(ply_file, "y", PLY_DOUBLE, PLY_DOUBLE, PLY_DOUBLE);
	ply_add_property(ply_file, "z", PLY_DOUBLE, PLY_DOUBLE, PLY_DOUBLE);
	if (voxelgrid.HasColors()) {
		ply_add_property(ply_file, "red", PLY_UCHAR, PLY_UCHAR, PLY_UCHAR);
		ply_add_property(ply_file, "green", PLY_UCHAR, PLY_UCHAR, PLY_UCHAR);
		ply_add_property(ply_file, "blue", PLY_UCHAR, PLY_UCHAR, PLY_UCHAR);
	}

	if (!ply_write_header(ply_file)) {
		utility::LogWarning("Write PLY failed: unable to write header.");
		ply_close(ply_file);
		return false;
	}

	utility::ConsoleProgressBar progress_bar(
		static_cast<size_t>(voxelgrid.voxels_.size()),
		"Writing PLY: ", print_progress);

	const Eigen::Vector3d &origin = voxelgrid.origin_;
	ply_write(ply_file, origin(0));
	ply_write(ply_file, origin(1));
	ply_write(ply_file, origin(2));
	ply_write(ply_file, voxelgrid.voxel_size_);

	for (auto &it : voxelgrid.voxels_) {
		const geometry::Voxel &voxel = it.second;
		ply_write(ply_file, voxel.grid_index_(0));
		ply_write(ply_file, voxel.grid_index_(1));
		ply_write(ply_file, voxel.grid_index_(2));

		const Eigen::Vector3d &color = voxel.color_;
		ply_write(ply_file, std::min(255.0, std::max(0.0, color(0) * 255.0)));
		ply_write(ply_file, std::min(255.0, std::max(0.0, color(1) * 255.0)));
		ply_write(ply_file, std::min(255.0, std::max(0.0, color(2) * 255.0)));

		++progress_bar;
	}

	ply_close(ply_file);
	return true;
}


}  // namespace io
}  // namespace cloudViewer
