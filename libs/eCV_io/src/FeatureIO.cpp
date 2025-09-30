// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "FeatureIO.h"

#include <Logging.h>
#include <FileSystem.h>

namespace cloudViewer {
namespace {
	using namespace io;

	bool ReadMatrixXdFromBINFile(FILE *file, Eigen::MatrixXd &mat) 
	{
		uint32_t rows, cols;
		if (fread(&rows, sizeof(uint32_t), 1, file) < 1) {
			cloudViewer::utility::LogWarning("Read BIN failed: unexpected EOF.");
			return false;
		}
		if (fread(&cols, sizeof(uint32_t), 1, file) < 1) {
			cloudViewer::utility::LogWarning("Read BIN failed: unexpected EOF.");
			return false;
		}
		mat.resize(rows, cols);
		if (fread(mat.data(), sizeof(double), rows * cols, file) < rows * cols) {
			cloudViewer::utility::LogWarning("Read BIN failed: unexpected EOF.");
			return false;
		}
		return true;
	}

	bool WriteMatrixXdToBINFile(FILE *file, const Eigen::MatrixXd &mat) 
	{
		uint32_t rows = (uint32_t)mat.rows();
		uint32_t cols = (uint32_t)mat.cols();
		if (fwrite(&rows, sizeof(uint32_t), 1, file) < 1) {
			cloudViewer::utility::LogWarning("Write BIN failed: unexpected error.");
			return false;
		}
		if (fwrite(&cols, sizeof(uint32_t), 1, file) < 1) {
			cloudViewer::utility::LogWarning("Write BIN failed: unexpected error.");
			return false;
		}
		if (fwrite(mat.data(), sizeof(double), rows * cols, file) < rows * cols) {
			cloudViewer::utility::LogWarning("Write BIN failed: unexpected error.");
			return false;
		}
		return true;
	}

}  // unnamed namespace

namespace io {

bool ReadFeature(const std::string &filename, utility::Feature &feature) 
{
    return ReadFeatureFromBIN(filename, feature);
}

bool WriteFeature(const std::string &filename, const utility::Feature &feature) 
{
    return WriteFeatureToBIN(filename, feature);
}

bool ReadFeatureFromBIN(const std::string &filename, utility::Feature &feature) 
{
	FILE *fid = cloudViewer::utility::filesystem::FOpen(filename, "rb");
	if (fid == NULL) {
		cloudViewer::utility::LogWarning("Read BIN failed: unable to open file: {}",
			filename);
		return false;
	}
	bool success = ReadMatrixXdFromBINFile(fid, feature.data_);
	fclose(fid);
	return success;
}

bool WriteFeatureToBIN(const std::string &filename, const utility::Feature &feature)
{
	FILE *fid = cloudViewer::utility::filesystem::FOpen(filename, "wb");
	if (fid == NULL) {
		cloudViewer::utility::LogWarning("Write BIN failed: unable to open file: {}",
			filename);
		return false;
	}
	bool success = WriteMatrixXdToBINFile(fid, feature.data_);
	fclose(fid);
	return success;
}


}  // namespace io
}  // namespace cloudViewer
