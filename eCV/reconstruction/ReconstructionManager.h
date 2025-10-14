// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "base/reconstruction.h"

namespace colmap {
class Reconstruction;
}

namespace cloudViewer {

class OptionManager;

class ReconstructionManager {
public:
    ReconstructionManager();

    // Move constructor and assignment.
    ReconstructionManager(ReconstructionManager&& other);
    ReconstructionManager& operator=(ReconstructionManager&& other);

    // The number of reconstructions managed.
    std::size_t Size() const;

    // Get a reference to a specific reconstruction.
    const colmap::Reconstruction& Get(const size_t idx) const;
    colmap::Reconstruction& Get(const size_t idx);

    // Add a new empty reconstruction and return its index.
    std::size_t Add();

    // Delete a specific reconstruction.
    void Delete(const size_t idx);

    // Delete all reconstructions.
    void Clear();

    // Read and add a new reconstruction and return its index.
    std::size_t Read(const std::string& path);

    // Write all managed reconstructions into sub-folders "0", "1", "2", ...
    // If the option manager object is not null, the options are written
    // to each respective reconstruction folder as well.
    void Write(const std::string& path, const OptionManager* options) const;

private:
    NON_COPYABLE(ReconstructionManager)

    std::vector<std::unique_ptr<colmap::Reconstruction>> reconstructions_;
};

}  // namespace cloudViewer
