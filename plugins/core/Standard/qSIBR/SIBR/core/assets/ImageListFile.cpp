// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "core/assets/ImageListFile.hpp"

#include <boost/filesystem.hpp>
#include <fstream>

namespace sibr {
bool ImageListFile::load(const std::string& filename, bool verbose) {
    std::fstream file(filename, std::ios::in);

    _infos.clear();
    if (file) {
        while (file.eof() == false) {
            Infos i;
            file >> i.filename >> i.width >> i.height;
            if (i.filename.size()) _infos.emplace_back(std::move(i));
        }

        // store basename
        boost::filesystem::path path(filename);
        _basename = path.parent_path().string();

        if (verbose)
            SIBR_FLOG << "'" << filename << "' successfully loaded."
                      << std::endl;

        return true;
    } else
        SIBR_WRG << "file not found: '" << filename << "'" << std::endl;
    return false;
}

}  // namespace sibr
