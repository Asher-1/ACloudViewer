// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "LineSetIO.h"

#include <rply.h>

#include <unordered_map>

#include <Logging.h>
#include <FileSystem.h>

namespace cloudViewer {

namespace {
using namespace io;

namespace ply_lineset_reader {

	struct PLYReaderState {
		cloudViewer::utility::ConsoleProgressBar *progress_bar;
		geometry::LineSet *lineset_ptr;
		long vertex_index;
		long vertex_num;
		long line_index;
		long line_num;
		long color_index;
		long color_num;
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
		state_ptr->lineset_ptr->points_[state_ptr->vertex_index](index) = value;
		if (index == 2) {  // reading 'z'
			state_ptr->vertex_index++;
			++(*state_ptr->progress_bar);
		}
		return 1;
	}

	int ReadLineCallback(p_ply_argument argument) {
		PLYReaderState *state_ptr;
		long index;
		ply_get_argument_user_data(argument, reinterpret_cast<void **>(&state_ptr),
			&index);
		if (state_ptr->line_index >= state_ptr->line_num) {
			return 0;
		}

		double value = ply_get_argument_value(argument);
		state_ptr->lineset_ptr->lines_[state_ptr->line_index](index) = int(value);
		if (index == 1) {  // reading 'vertex2'
			state_ptr->line_index++;
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
		state_ptr->lineset_ptr->colors_[state_ptr->color_index](index) =
			value / 255.0;
		if (index == 2) {  // reading 'blue'
			state_ptr->color_index++;
			++(*state_ptr->progress_bar);
		}
		return 1;
	}

}  // namespace ply_lineset_reader


static const std::unordered_map<
        std::string,
        std::function<bool(const std::string &, geometry::LineSet &, bool)>>
        file_extension_to_lineset_read_function{
                {"ply", ReadLineSetFromPLY},
        };

static const std::unordered_map<std::string,
                                std::function<bool(const std::string &,
                                                   const geometry::LineSet &,
                                                   const bool,
                                                   const bool,
                                                   const bool)>>
        file_extension_to_lineset_write_function{
                {"ply", WriteLineSetToPLY},
        };
}  // unnamed namespace

namespace io {
using namespace cloudViewer;
std::shared_ptr<geometry::LineSet> CreateLineSetFromFile(
        const std::string &filename,
        const std::string &format,
        bool print_progress) {
    auto lineset = cloudViewer::make_shared<geometry::LineSet>();
    ReadLineSet(filename, *lineset, format, print_progress);
    return lineset;
}

bool ReadLineSet(const std::string &filename,
                 geometry::LineSet &lineset,
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
                "Read geometry::LineSet failed: unknown file extension.");
        return false;
    }
    auto map_itr = file_extension_to_lineset_read_function.find(filename_ext);
    if (map_itr == file_extension_to_lineset_read_function.end()) {
        utility::LogWarning(
                "Read geometry::LineSet failed: unknown file extension.");
        return false;
    }
    bool success = map_itr->second(filename, lineset, print_progress);
    utility::LogDebug("Read geometry::LineSet: {:d} vertices.",
                      (int)lineset.points_.size());
    return success;
}

bool WriteLineSet(const std::string &filename,
                  const geometry::LineSet &lineset,
                  bool write_ascii /* = false*/,
                  bool compressed /* = false*/,
                  bool print_progress) {
    std::string filename_ext =
            utility::filesystem::GetFileExtensionInLowerCase(filename);
    if (filename_ext.empty()) {
        utility::LogWarning(
                "Write geometry::LineSet failed: unknown file extension.");
        return false;
    }
    auto map_itr = file_extension_to_lineset_write_function.find(filename_ext);
    if (map_itr == file_extension_to_lineset_write_function.end()) {
        utility::LogWarning(
                "Write geometry::LineSet failed: unknown file extension.");
        return false;
    }
    bool success = map_itr->second(filename, lineset, write_ascii, compressed,
                                   print_progress);
    utility::LogDebug("Write geometry::LineSet: {:d} vertices.",
                      (int)lineset.points_.size());
    return success;
}


bool ReadLineSetFromPLY(const std::string &filename,
	geometry::LineSet &lineset,
	bool print_progress) {
	using namespace ply_lineset_reader;

	p_ply ply_file = ply_open(filename.c_str(), NULL, 0, NULL);
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
	state.lineset_ptr = &lineset;
	state.vertex_num = ply_set_read_cb(ply_file, "vertex", "x",
		ReadVertexCallback, &state, 0);
	ply_set_read_cb(ply_file, "vertex", "y", ReadVertexCallback, &state, 1);
	ply_set_read_cb(ply_file, "vertex", "z", ReadVertexCallback, &state, 2);

	state.line_num = ply_set_read_cb(ply_file, "edge", "vertex1",
		ReadLineCallback, &state, 0);
	ply_set_read_cb(ply_file, "edge", "vertex2", ReadLineCallback, &state, 1);

	state.color_num = ply_set_read_cb(ply_file, "edge", "red",
		ReadColorCallback, &state, 0);
	ply_set_read_cb(ply_file, "edge", "green", ReadColorCallback, &state, 1);
	ply_set_read_cb(ply_file, "edge", "blue", ReadColorCallback, &state, 2);

	if (state.vertex_num <= 0) {
		utility::LogWarning("Read PLY failed: number of vertex <= 0.");
		ply_close(ply_file);
		return false;
	}
	if (state.line_num <= 0) {
		utility::LogWarning("Read PLY failed: number of edges <= 0.");
		ply_close(ply_file);
		return false;
	}

	state.vertex_index = 0;
	state.line_index = 0;
	state.color_index = 0;

	lineset.clear();
	lineset.points_.resize(state.vertex_num);
	lineset.lines_.resize(state.line_num);
	lineset.colors_.resize(state.color_num);

	utility::ConsoleProgressBar progress_bar(
		state.vertex_num + state.line_num + state.color_num,
		"Reading PLY: ", print_progress);
	state.progress_bar = &progress_bar;

	if (!ply_read(ply_file)) {
		utility::LogWarning("Read PLY failed: unable to read file: {}",
			filename);
		ply_close(ply_file);
		return false;
	}

	ply_close(ply_file);
	return true;
}

bool WriteLineSetToPLY(const std::string &filename,
	const geometry::LineSet &lineset,
	bool write_ascii /* = false*/,
	bool compressed /* = false*/,
	bool print_progress) {
	if (lineset.IsEmpty()) {
		utility::LogWarning("Write PLY failed: line set has 0 points.");
		return false;
	}
	if (!lineset.HasLines()) {
		utility::LogWarning("Write PLY failed: line set has 0 lines.");
		return false;
	}

	p_ply ply_file = ply_create(filename.c_str(),
		write_ascii ? PLY_ASCII : PLY_LITTLE_ENDIAN,
		NULL, 0, NULL);
	if (!ply_file) {
		utility::LogWarning("Write PLY failed: unable to open file: {}",
			filename);
		return false;
	}
	ply_add_comment(ply_file, "Created by cloudViewer");
	ply_add_element(ply_file, "vertex",
		static_cast<long>(lineset.points_.size()));
	ply_add_property(ply_file, "x", PLY_DOUBLE, PLY_DOUBLE, PLY_DOUBLE);
	ply_add_property(ply_file, "y", PLY_DOUBLE, PLY_DOUBLE, PLY_DOUBLE);
	ply_add_property(ply_file, "z", PLY_DOUBLE, PLY_DOUBLE, PLY_DOUBLE);
	ply_add_element(ply_file, "edge", static_cast<long>(lineset.lines_.size()));
	ply_add_property(ply_file, "vertex1", PLY_INT, PLY_INT, PLY_INT);
	ply_add_property(ply_file, "vertex2", PLY_INT, PLY_INT, PLY_INT);
	if (lineset.HasColors()) {
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
		static_cast<size_t>(lineset.points_.size() + lineset.lines_.size()),
		"Writing PLY: ", print_progress);

	for (size_t i = 0; i < lineset.points_.size(); i++) {
		const Eigen::Vector3d &point = lineset.points_[i];
		ply_write(ply_file, point(0));
		ply_write(ply_file, point(1));
		ply_write(ply_file, point(2));
		++progress_bar;
	}
	bool printed_color_warning = false;
	for (size_t i = 0; i < lineset.lines_.size(); i++) {
		const Eigen::Vector2i &line = lineset.lines_[i];
		ply_write(ply_file, line(0));
		ply_write(ply_file, line(1));
		if (lineset.HasColors()) {
			const Eigen::Vector3d &color = lineset.colors_[i];
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
		++progress_bar;
	}

	ply_close(ply_file);
	return true;
}

}  // namespace io
}  // namespace cloudViewer
