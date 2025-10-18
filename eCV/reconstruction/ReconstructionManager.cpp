// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "ReconstructionManager.h"

#include "OptionManager.h"
#include "util/misc.h"

namespace cloudViewer {

using namespace colmap;

ReconstructionManager::ReconstructionManager() {}

ReconstructionManager::ReconstructionManager(ReconstructionManager&& other)
    : ReconstructionManager() {
    reconstructions_ = std::move(other.reconstructions_);
}

ReconstructionManager& ReconstructionManager::operator=(
        ReconstructionManager&& other) {
    if (this != &other) {
        reconstructions_ = std::move(other.reconstructions_);
    }
    return *this;
}

size_t ReconstructionManager::Size() const { return reconstructions_.size(); }

const Reconstruction& ReconstructionManager::Get(const size_t idx) const {
    return *reconstructions_.at(idx);
}

Reconstruction& ReconstructionManager::Get(const size_t idx) {
    return *reconstructions_.at(idx);
}

size_t ReconstructionManager::Add() {
    const size_t idx = Size();
    reconstructions_.emplace_back(new Reconstruction());
    return idx;
}

void ReconstructionManager::Delete(const size_t idx) {
    CHECK_LT(idx, reconstructions_.size());
    reconstructions_.erase(reconstructions_.begin() + idx);
}

void ReconstructionManager::Clear() { reconstructions_.clear(); }

size_t ReconstructionManager::Read(const std::string& path) {
    const size_t idx = Add();
    reconstructions_[idx]->Read(path);
    return idx;
}

void ReconstructionManager::Write(const std::string& path,
                                  const OptionManager* options) const {
    std::vector<std::pair<size_t, size_t>> recon_sizes(reconstructions_.size());
    for (size_t i = 0; i < reconstructions_.size(); ++i) {
        recon_sizes[i] = std::make_pair(i, reconstructions_[i]->NumPoints3D());
    }
    std::sort(recon_sizes.begin(), recon_sizes.end(),
              [](const std::pair<size_t, size_t>& first,
                 const std::pair<size_t, size_t>& second) {
                  return first.second > second.second;
              });

    for (size_t i = 0; i < reconstructions_.size(); ++i) {
        const std::string reconstruction_path =
                JoinPaths(path, std::to_string(i));
        CreateDirIfNotExists(reconstruction_path);
        reconstructions_[recon_sizes[i].first]->Write(reconstruction_path);
        if (options != nullptr) {
            options->Write(JoinPaths(reconstruction_path, "project.ini"));
        }
    }
}

}  // namespace cloudViewer
